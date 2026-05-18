import os
import ctypes
import time
from collections import deque
from typing import List, Set, Tuple, cast

import numpy as np
import pygame as pg
import pygame.gfxdraw as gfxdraw
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from ai_agents import random_reveal_safe
from engine import MinesweeperEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINE_WIDTH: int = 1   # 1-px grid lines between every cell, both styles
COVERED:    int = -1

# Simple style
grid_color:     Tuple[int, int, int] = (85,  85,  85)
covered_color:  Tuple[int, int, int] = (50,  50,  50)
revealed_color: Tuple[int, int, int] = (18,  18,  18)

# Nice style
background_color: Tuple[int, int, int] = (5,   5,   5)
cell_color:       Tuple[int, int, int] = (30,  30,  30)
line_color:       Tuple[int, int, int] = (125, 125, 125)

text_color: Tuple[int, int, int] = (220, 220, 220)

NUMBER_COLORS: List[Tuple[int, int, int]] = [
    (100, 149, 237),  # 1 steel blue
    ( 60, 179,  60),  # 2 green
    (220,  80,  80),  # 3 red
    ( 65, 105, 225),  # 4 royal blue
    (180,  50,  50),  # 5 dark red
    ( 32, 178, 170),  # 6 teal
    (200, 200, 200),  # 7 light gray
    (120, 120, 120),  # 8 gray
]

IMAGE_PATH: str = "./assets/images"

levels    = MinesweeperEngine.LEVELS
game_mode = set(MinesweeperEngine.DIFFICULTIES.keys())


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------

def _prob_color(p: float) -> Tuple[int, int, int]:
    if p <= 0.5:
        return (int(510 * p), 255, 0)
    return (255, int(510 * (1.0 - p)), 0)


# ---------------------------------------------------------------------------
# Nice-style rendering utilities
# ---------------------------------------------------------------------------

def _blur_bg(screen: pg.Surface, sigma: float = 0.5) -> None:
    pixels = pg.surfarray.pixels3d(screen)
    for ch in range(3):
        gaussian_filter(pixels[:, :, ch], sigma=sigma, mode="nearest", output=pixels[:, :, ch])


def _find_clusters(board: np.ndarray, flag: int) -> List[List[Tuple[int, int]]]:
    rows, cols = board.shape
    visited    = np.zeros_like(board, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _flood(sr: int, sc: int) -> List[Tuple[int, int]]:
        cluster: List[Tuple[int, int]] = []
        q: deque[Tuple[int, int]] = deque([(sr, sc)])
        visited[sr, sc] = True
        while q:
            r, c = q.popleft()
            cluster.append((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and board[nr, nc] == flag:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        return cluster

    return [
        _flood(r, c)
        for r in range(rows)
        for c in range(cols)
        if board[r, c] == flag and not visited[r, c]
    ]


def _cluster_to_rects(cluster: List[Tuple[int, int]], cell_size: int) -> List[pg.Rect]:
    """
    Rects are expanded by LINE_WIDTH so adjacent cells share polygon edges in Shapely space,
    allowing unary_union to merge them into one blob. The extra pixel bleeds into the grid-line
    gap but is invisible — blobs are drawn before the line overlay, and cells never overlap.
    """
    stride    = cell_size + LINE_WIDTH
    blob_size = cell_size + LINE_WIDTH  # fill the gap so adjacent polys share an edge
    return [
        pg.Rect(LINE_WIDTH + c * stride, LINE_WIDTH + r * stride, blob_size, blob_size)
        for r, c in cluster
    ]


def _rects_to_polygon(rects: List[pg.Rect]) -> Polygon:
    return cast(Polygon, unary_union([
        Polygon([rect.topleft, rect.topright, rect.bottomright, rect.bottomleft])
        for rect in rects
    ]))


def _draw_polygon_with_holes(
    surface:    pg.Surface,
    polygon:    Polygon,
    fill_color: Tuple[int, int, int],
    hole_color: Tuple[int, int, int],
    cell_size:  int,
) -> None:
    resolution = max(16, int(polygon.length / 10))
    d = cell_size // 9

    smoothed = (
        polygon
        .buffer( d,      cap_style="round", join_style="round", resolution=resolution)
        .buffer(-d * 3.0, cap_style="round", join_style="round", resolution=resolution)
        .buffer( d,      cap_style="round", join_style="round", resolution=resolution)
    )
    gfxdraw.aapolygon(    surface, list(map(tuple, smoothed.exterior.coords)), fill_color)
    gfxdraw.filled_polygon(surface, list(map(tuple, smoothed.exterior.coords)), fill_color)

    for interior in polygon.interiors:
        hole = (
            Polygon(interior.coords)
            .buffer( d * 1.5, cap_style="round", join_style="round", resolution=resolution)
            .buffer(-d * 2.0, cap_style="round", join_style="round", resolution=resolution)
            .buffer( d * 1.5, cap_style="round", join_style="round", resolution=resolution)
        )
        gfxdraw.aapolygon(    surface, list(map(tuple, hole.exterior.coords)), hole_color)
        gfxdraw.filled_polygon(surface, list(map(tuple, hole.exterior.coords)), hole_color)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class GUI:
    STYLES = ("simple", "nice")

    def __init__(self, level: int = 1, style: str = "nice") -> None:
        lname = levels.get(level)
        if lname is None or lname not in game_mode:
            lname = "hard" if "hard" in game_mode else next(iter(game_mode))
        self.level:        str  = lname
        self.style:        str  = style if style in self.STYLES else "nice"
        self._initialized: bool = False

        if hasattr(ctypes, "windll"):
            user32 = ctypes.windll.user32
            sw = user32.GetSystemMetrics(0)
            sh = user32.GetSystemMetrics(1)
        else:
            pg.display.init()
            info = pg.display.Info()
            sw = info.current_w if info.current_w > 0 else 1920
            sh = info.current_h if info.current_h > 0 else 1080
        self._sw = sw
        self._sh = sh

        # Compute max cell size per level once, accounting for LINE_WIDTH grid lines.
        # Layout: (n+1) grid lines of 1px + n cells → cell = (avail - n - 1) // n
        self._cell_sizes: dict[str, int] = {
            n: max(min(
                (sh - 80 - (rows + 1) * LINE_WIDTH) // rows,
                (sw - 80 - (cols + 1) * LINE_WIDTH) // cols,
            ), 4)
            for n, (rows, cols, _) in MinesweeperEngine.DIFFICULTIES.items()
        }

        self.init_game()

    def init_game(self) -> None:
        if self._initialized:
            pg.quit()

        self.running:         bool                = True
        self.help:            bool                = True
        self.colored_numbers: bool                = True
        self.flagged:         Set[Tuple[int, int]] = set()

        _ids         = {v: k for k, v in MinesweeperEngine.LEVELS.items()}
        self.board   = MinesweeperEngine(level=_ids[self.level], seed=int(time.time() * 1000) % (2**31))
        self.probability = self.board.mine_probs
        rows, cols   = self.board.rows, self.board.cols

        self.cell_size: int = self._cell_sizes[self.level]
        stride              = self.cell_size + LINE_WIDTH

        # Window size: n cells * stride + 1 closing border
        # → [1px] cell [1px] cell ... cell [1px]  — no overlap possible
        self.width:  int = cols * stride + LINE_WIDTH
        self.height: int = rows * stride + LINE_WIDTH

        os.environ["SDL_VIDEO_WINDOW_POS"] = (
            f"{(self._sw - self.width) // 2},{(self._sh - self.height - 20) // 2}"
        )

        pg.init()
        pg.font.init()
        self.clock  = pg.time.Clock()
        self.screen = pg.display.set_mode((self.width, self.height))
        self.font   = pg.font.SysFont("Consolas", max(8, min(30, int(self.cell_size * 0.32))))

        mine_img         = pg.image.load(f"{IMAGE_PATH}/mine.png").convert_alpha()
        flag_img         = pg.image.load(f"{IMAGE_PATH}/flag.png").convert_alpha()
        half             = max(self.cell_size // 2, 2)
        self.mine_image  = mine_img
        self.flag_image  = flag_img
        self.scaled_mine = pg.transform.scale(mine_img, (half, half))
        self.scaled_flag = pg.transform.scale(flag_img, (half, half))

        pg.display.set_caption("Minesweeper")
        pg.display.set_icon(mine_img)
        self._initialized = True

    def quit(self) -> None:
        self.running = False

    def reset_game(self) -> None:
        self.init_game()

    # -----------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------

    def handle_events(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            elif event.type == pg.KEYDOWN:
                self.handle_key_event(event.key)
            elif event.type == pg.MOUSEBUTTONDOWN:
                self.handle_mouse_event(event)

    def handle_key_event(self, key: int) -> None:
        if key == pg.K_ESCAPE:
            self.quit()
        elif key == pg.K_r:
            self.reset_game()
        elif key == pg.K_h:
            self.help = not self.help
            if self.help:
                self.probability = self.board.mine_probs
        elif key == pg.K_c:
            self.colored_numbers = not self.colored_numbers
        elif key == pg.K_t:
            self.style = "nice" if self.style == "simple" else "simple"
        elif key == pg.K_z:
            random_reveal_safe(self.board)
            self.probability = self.board.mine_probs
        elif key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
            chosen = levels.get(key - pg.K_0)
            if chosen and chosen in game_mode:
                self.level = chosen
                self.reset_game()

    def handle_mouse_event(self, event: pg.event.Event) -> None:
        if self.board.state == MinesweeperEngine.OVER:
            return
        cell = self._cell_from_mouse(*event.pos)
        if cell is None:
            return
        row, col = cell
        if event.button == pg.BUTTON_LEFT:
            if (row, col) not in self.flagged:
                self.board.reveal(row, col)
                self.probability = self.board.mine_probs
                self.flagged = {f for f in self.flagged if not self.board.revealed[f[0], f[1]]}
        elif event.button == pg.BUTTON_RIGHT and not self.board.revealed[row, col]:
            self.flagged.symmetric_difference_update({(row, col)})

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------

    def _cell_from_mouse(self, mx: int, my: int) -> Tuple[int, int] | None:
        stride = self.cell_size + LINE_WIDTH
        if mx < LINE_WIDTH or my < LINE_WIDTH:
            return None
        col, col_rem = divmod(mx - LINE_WIDTH, stride)
        row, row_rem = divmod(my - LINE_WIDTH, stride)
        if col_rem >= self.cell_size or row_rem >= self.cell_size:
            return None  # click landed on a grid line
        if 0 <= row < self.board.rows and 0 <= col < self.board.cols:
            return int(row), int(col)
        return None

    def _cell_origin(self, row: int, col: int) -> Tuple[int, int]:
        stride = self.cell_size + LINE_WIDTH
        return LINE_WIDTH + col * stride, LINE_WIDTH + row * stride

    def _blit_centered(self, image: pg.Surface, row: int, col: int) -> None:
        x, y = self._cell_origin(row, col)
        self.screen.blit(image, image.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2)))

    def _tint(self, color: Tuple[int, int, int, int]) -> None:
        overlay = pg.Surface((self.width, self.height), pg.SRCALPHA)
        overlay.fill(color)
        self.screen.blit(overlay, (0, 0))

    def _draw_overlay(self, lines: List[str]) -> None:
        lh = self.font.get_height() + 4
        y  = (self.height - len(lines) * lh) // 2
        for line in lines:
            surf = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(surf, surf.get_rect(center=(self.width // 2, y)))
            y += lh

    def _level_help_lines(self) -> List[str]:
        lines = [f"Current Level: {self.level.capitalize()}"]
        for lvl, name in levels.items():
            if name in game_mode:
                lines.append(f"{lvl} = {name.capitalize()}")
        lines.append("H: toggle help")
        lines.append("C: toggle number colours")
        lines.append("T: toggle style")
        return lines

    def _number_color(self, n: int) -> Tuple[int, int, int]:
        if self.colored_numbers and 1 <= n <= 8:
            return NUMBER_COLORS[n - 1]
        return text_color

    # -----------------------------------------------------------------------
    # Draw dispatch
    # -----------------------------------------------------------------------

    def draw(self) -> None:
        if self.style == "simple":
            self._draw_simple()
        else:
            self._draw_nice()

    # -----------------------------------------------------------------------
    # Simple draw path
    # -----------------------------------------------------------------------

    def _draw_simple(self) -> None:
        self.screen.fill(grid_color)  # background shows through gaps = grid lines
        if self.board.state == MinesweeperEngine.OVER and self.board.hit_mine:
            self._simple_cells(show_mines=True)
            self._tint((128, 0, 0, 64))
            self._draw_overlay(["Game Over  |  R to restart", *self._level_help_lines()])
        elif self.board.state == MinesweeperEngine.OVER and self.board.won:
            self._simple_cells(show_flags=True)
            self._tint((0, 128, 0, 64))
            self._draw_overlay(["You Won!  |  R to restart", *self._level_help_lines()])
        else:
            self._simple_cells()

    def _simple_cells(self, show_mines: bool = False, show_flags: bool = False) -> None:
        stride   = self.cell_size + LINE_WIDTH
        revealed = self.board.revealed
        counts   = self.board._neighbor_counts
        mines    = self.board._mines
        prob     = self.probability

        for row in range(self.board.rows):
            y = LINE_WIDTH + row * stride
            for col in range(self.board.cols):
                x    = LINE_WIDTH + col * stride
                rect = pg.Rect(x, y, self.cell_size, self.cell_size)
                cx   = x + self.cell_size // 2
                cy   = y + self.cell_size // 2

                if revealed[row, col]:
                    pg.draw.rect(self.screen, revealed_color, rect)
                    n = int(counts[row, col])
                    if n > 0:
                        surf = self.font.render(str(n), True, self._number_color(n))
                        self.screen.blit(surf, surf.get_rect(center=(cx, cy)))
                else:
                    pg.draw.rect(self.screen, covered_color, rect)
                    if show_mines and mines[row, col]:
                        self._blit_centered(self.scaled_mine, row, col)
                    elif show_flags and mines[row, col]:
                        self._blit_centered(self.scaled_flag, row, col)
                    elif (row, col) in self.flagged:
                        self._blit_centered(self.scaled_flag, row, col)
                    elif self.help and not np.isnan(prob[row, col]):
                        p    = float(prob[row, col])
                        text = "S" if p == 0.0 else ("X" if p == 1.0 else f"{p:.2f}")
                        surf = self.font.render(text, True, _prob_color(p))
                        self.screen.blit(surf, surf.get_rect(center=(cx, cy)))

    # -----------------------------------------------------------------------
    # Nice draw path
    # -----------------------------------------------------------------------

    def _draw_nice(self) -> None:
        self.screen.fill(background_color)

        if self.board.state == MinesweeperEngine.OVER and self.board.hit_mine:
            self._nice_mines()
            self._nice_cells()
            self._tint((128, 0, 0, 64))
            self._nice_lines()
            _blur_bg(self.screen, sigma=2)
            self._draw_overlay(["Game Over  |  R to restart", *self._level_help_lines()])

        elif self.board.state == MinesweeperEngine.OVER and self.board.won:
            self._nice_clusters()
            self._nice_cells()
            self._nice_flags()
            self._tint((0, 128, 0, 64))
            self._nice_lines()
            _blur_bg(self.screen, sigma=2)
            self._draw_overlay(["You Won!  |  R to restart", *self._level_help_lines()])

        else:
            self._nice_clusters()
            self._nice_cells()
            self._nice_markers()
            self._nice_lines()
            _blur_bg(self.screen, sigma=0.32)

    def _nice_clusters(self) -> None:
        covered_mask = np.where(self.board.revealed, np.int8(0), np.int8(COVERED))
        for cluster in _find_clusters(covered_mask, COVERED):
            union = _rects_to_polygon(_cluster_to_rects(cluster, self.cell_size))
            polys = list(union.geoms) if isinstance(union, MultiPolygon) else [union]
            for poly in polys:
                _draw_polygon_with_holes(
                    self.screen, poly, cell_color, background_color, self.cell_size,
                )
        _blur_bg(self.screen, sigma=0.8)

    def _nice_cells(self) -> None:
        for row in range(self.board.rows):
            for col in range(self.board.cols):
                x, y = self._cell_origin(row, col)
                cx   = x + self.cell_size // 2
                cy   = y + self.cell_size // 2

                if self.board.revealed[row, col]:
                    n = int(self.board._neighbor_counts[row, col])
                    if n > 0:
                        surf = self.font.render(str(n), True, self._number_color(n))
                        self.screen.blit(surf, surf.get_rect(center=(cx, cy)))
                elif (
                    self.help
                    and (row, col) not in self.flagged
                    and not np.isnan(self.probability[row, col])
                ):
                    p    = float(self.probability[row, col])
                    text = "S" if p == 0.0 else ("X" if p == 1.0 else f"{p:.2f}")
                    surf = self.font.render(text, True, _prob_color(p))
                    self.screen.blit(surf, surf.get_rect(center=(cx, cy)))

    def _nice_mines(self) -> None:
        for row, col in np.argwhere(self.board._mines):
            self._blit_centered(self.scaled_mine, row, col)

    def _nice_flags(self) -> None:
        for row, col in np.argwhere(self.board._mines):
            self._blit_centered(self.scaled_flag, row, col)

    def _nice_markers(self) -> None:
        for row, col in self.flagged:
            self._blit_centered(self.scaled_flag, row, col)

    def _nice_lines(self) -> None:
        """Dashed grid lines at the 1-px gaps between cells.
        Segments are pinned to each cell's origin so the pattern never drifts."""
        stride = self.cell_size + LINE_WIDTH
        gap    = max(1, self.cell_size // 8)

        for col in range(self.board.cols - 1):
            x = (col + 1) * stride
            for row in range(self.board.rows):
                y0 = LINE_WIDTH + row * stride
                pg.draw.line(self.screen, line_color, (x, y0 + gap), (x, y0 + self.cell_size - gap), 1)

        for row in range(self.board.rows - 1):
            y = (row + 1) * stride
            for col in range(self.board.cols):
                x0 = LINE_WIDTH + col * stride
                pg.draw.line(self.screen, line_color, (x0 + gap, y), (x0 + self.cell_size - gap, y), 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(style: str = "nice") -> None:
    game = GUI(level=1, style=style)
    while game.running:
        game.handle_events()
        game.draw()
        pg.display.update()
    pg.quit()


if __name__ == "__main__":
    main()
