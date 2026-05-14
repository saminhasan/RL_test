import os
import ctypes
import time
from typing import Set, Tuple

import numpy as np
import pygame as pg

from engine import MinesweeperEngine

LINE_WIDTH: int = 1
CELL_SIZE: int = 48  # recalculated at init

covered_color: Tuple[int, int, int] = (50, 50, 50)
revealed_color: Tuple[int, int, int] = (18, 18, 18)
grid_color: Tuple[int, int, int] = (85, 85, 85)
text_color: Tuple[int, int, int] = (220, 220, 220)

NUMBER_COLORS = [
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

levels = {1: "easy", 2: "medium", 3: "hard"}
game_mode = set(MinesweeperEngine.DIFFICULTIES.keys())


def _prob_color(p: float) -> Tuple[int, int, int]:
    if p <= 0.5:
        return (int(510 * p), 255, 0)
    return (255, int(510 * (1.0 - p)), 0)


class GUI:
    def __init__(self, level: int = 1) -> None:
        name = levels.get(level)
        if name is None or name not in game_mode:
            name = "hard" if "hard" in game_mode else next(iter(game_mode))
        self.level = name
        self._initialized = False
        self.init_game()

    def init_game(self) -> None:
        global CELL_SIZE

        if self._initialized:
            pg.quit()

        self.running = True
        self.help = True
        self.flagged: Set[Tuple[int, int]] = set()

        level_id = {v: k for k, v in MinesweeperEngine.LEVELS.items()}[self.level]
        self.board = MinesweeperEngine(level=level_id, seed=int(time.time() * 1000) % (2**31))
        self.probability = self.board.get_mine_probabilities()

        rows, cols = self.board.rows, self.board.cols

        user32 = ctypes.windll.user32
        sw = user32.GetSystemMetrics(0)
        sh = user32.GetSystemMetrics(1)

        avail_w = sw - 80
        avail_h = sh - 80
        # total pixels = n*CELL_SIZE + (n+1)*LINE_WIDTH  =>  CELL_SIZE = (avail - (n+1)*LW) // n
        CELL_SIZE = min(
            (avail_h - (rows + 1) * LINE_WIDTH) // rows,
            (avail_w - (cols + 1) * LINE_WIDTH) // cols,
        )
        CELL_SIZE = max(CELL_SIZE, 4)

        stride = CELL_SIZE + LINE_WIDTH
        self.width = cols * stride + LINE_WIDTH
        self.height = rows * stride + LINE_WIDTH

        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{(sw - self.width) // 2},{(sh - self.height - 20) // 2}"

        pg.init()
        pg.font.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((self.width, self.height))
        self.font = pg.font.SysFont("Consolas", max(5, CELL_SIZE * 2 // 5))

        mine_img = pg.image.load(f"{IMAGE_PATH}/mine.png").convert_alpha()
        flag_img = pg.image.load(f"{IMAGE_PATH}/flag.png").convert_alpha()
        half = max(CELL_SIZE // 2, 2)
        self.scaled_mine = pg.transform.scale(mine_img, (half, half))
        self.scaled_flag = pg.transform.scale(flag_img, (half, half))

        pg.display.set_caption("Minesweeper")
        pg.display.set_icon(mine_img)
        self._initialized = True

    def _cell_from_mouse(self, mx: int, my: int) -> Tuple[int, int] | None:
        stride = CELL_SIZE + LINE_WIDTH
        if mx < LINE_WIDTH or my < LINE_WIDTH:
            return None
        col, col_rem = divmod(mx - LINE_WIDTH, stride)
        row, row_rem = divmod(my - LINE_WIDTH, stride)
        if col_rem >= CELL_SIZE or row_rem >= CELL_SIZE:
            return None  # on a grid line
        if 0 <= row < self.board.rows and 0 <= col < self.board.cols:
            return int(row), int(col)
        return None

    def handle_events(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.KEYDOWN:
                self._handle_key(event.key)
            elif event.type == pg.MOUSEBUTTONDOWN:
                self._handle_mouse(event)

    def _handle_key(self, key: int) -> None:
        if key == pg.K_ESCAPE:
            self.running = False
        elif key == pg.K_r:
            self.init_game()
        elif key == pg.K_h:
            self.help = not self.help
            if self.help:
                self.probability = self.board.get_mine_probabilities()
        elif key == pg.K_z:
            self.board.random_reveal(safe=True)
            self.probability = self.board.get_mine_probabilities()
        elif key in (pg.K_1, pg.K_2, pg.K_3):
            chosen = levels.get(key - pg.K_0)
            if chosen and chosen in game_mode:
                self.level = chosen
                self.init_game()

    def _handle_mouse(self, event: pg.event.Event) -> None:
        if self.board.state == MinesweeperEngine.OVER:
            return
        cell = self._cell_from_mouse(*event.pos)
        if cell is None:
            return
        row, col = cell
        if event.button == pg.BUTTON_LEFT:
            if (row, col) not in self.flagged:
                self.board.reveal(row, col)
                if self.board.state != MinesweeperEngine.OVER:
                    self.probability = self.board.get_mine_probabilities()
                    self.flagged = {f for f in self.flagged if not self.board.revealed[f[0], f[1]]}
        elif event.button == pg.BUTTON_RIGHT and not self.board.revealed[row, col]:
            self.flagged.symmetric_difference_update({(row, col)})

    def draw(self) -> None:
        # grid lines are just the background color showing through
        self.screen.fill(grid_color)
        if self.board.state == MinesweeperEngine.OVER and self.board.hit_mine:
            self._draw_cells(show_mines=True)
            self._tint((128, 0, 0, 64))
            self._draw_overlay("Game Over  |  R to restart")
        elif self.board.state == MinesweeperEngine.OVER and self.board.won:
            self._draw_cells(show_flags=True)
            self._tint((0, 128, 0, 64))
            self._draw_overlay("You Won!  |  R to restart")
        else:
            self._draw_cells()

    def _tint(self, color: Tuple[int, int, int, int]) -> None:
        overlay = pg.Surface((self.width, self.height), pg.SRCALPHA)
        overlay.fill(color)
        self.screen.blit(overlay, (0, 0))

    def _draw_cells(self, show_mines: bool = False, show_flags: bool = False) -> None:
        stride = CELL_SIZE + LINE_WIDTH
        revealed = self.board.revealed
        counts = self.board._neighbor_counts
        mines = self.board._mines
        prob = self.probability

        for row in range(self.board.rows):
            y = LINE_WIDTH + row * stride
            for col in range(self.board.cols):
                x = LINE_WIDTH + col * stride
                rect = pg.Rect(x, y, CELL_SIZE, CELL_SIZE)
                cx = x + CELL_SIZE // 2
                cy = y + CELL_SIZE // 2

                if revealed[row, col]:
                    pg.draw.rect(self.screen, revealed_color, rect)
                    n = int(counts[row, col])
                    if n > 0:
                        color = NUMBER_COLORS[n - 1] if n <= 8 else text_color
                        surf = self.font.render(str(n), True, color)
                        self.screen.blit(surf, surf.get_rect(center=(cx, cy)))
                else:
                    pg.draw.rect(self.screen, covered_color, rect)
                    if show_mines and mines[row, col]:
                        self.screen.blit(self.scaled_mine, self.scaled_mine.get_rect(center=(cx, cy)))
                    elif show_flags and mines[row, col]:
                        self.screen.blit(self.scaled_flag, self.scaled_flag.get_rect(center=(cx, cy)))
                    elif (row, col) in self.flagged:
                        self.screen.blit(self.scaled_flag, self.scaled_flag.get_rect(center=(cx, cy)))
                    elif self.help and not np.isnan(prob[row, col]):
                        p = float(prob[row, col])
                        text = "S" if p == 0.0 else ("X" if p == 1.0 else f"{p:.2f}")
                        surf = self.font.render(text, True, _prob_color(p))
                        self.screen.blit(surf, surf.get_rect(center=(cx, cy)))

    def _draw_overlay(self, message: str) -> None:
        lines = [
            message,
            f"Level: {self.level.capitalize()}",
            "1=Easy  2=Medium  3=Hard",
            "H=Toggle help  Z=Safe reveal  R=Restart",
        ]
        lh = self.font.get_height() + 4
        y = (self.height - len(lines) * lh) // 2
        for line in lines:
            surf = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(surf, surf.get_rect(center=(self.width // 2, y)))
            y += lh


def main() -> None:
    game = GUI(1)
    while game.running:
        game.handle_events()
        game.draw()
        pg.display.update()
        game.clock.tick(120)


if __name__ == "__main__":
    main()
