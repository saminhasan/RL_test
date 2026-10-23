# Double DQN for Minesweeper.
#
# Adapted from CleanRL's dqn.py: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
# (MIT License, Copyright (c) 2019 CleanRL developers; docs https://docs.cleanrl.dev/rl-algorithms/dqn/).
# The training loop is CleanRL's. Every change is marked "MINESWEEPER:":
#   * Double DQN target: the online net picks the next cell, the target net scores it
#   * cells already revealed are never chosen: not when acting, exploring, or in the target
#   * convolutional Q-network, one value per cell (rl/qnet.py), instead of CleanRL's MLP
#   * many games stepped together (CleanRL's dqn.py allows one)
#   * speed-ups, each switchable:
#       - curriculum: easy -> medium -> hard boards, all drawn on the hard board's canvas
#       - multi-step returns: each click learns from the next `n_step` rewards
#       - prioritized replay (OpenAI Baselines' implementation, rl/per.py) with
#         importance weights; per_alpha 0 turns it off
#   * Huber loss and gradient clipping for stability; bf16 maths for speed
#   * evaluation against bayes on fixed boards (measuring only), checkpoints,
#     resume, and clean interruption
#   * for long unattended runs: the learning rate halves when evaluations stop
#     improving, a guard slows down while the GPU is too hot, and the run's
#     progress is written to runs/<run>/ for the dashboard (rl/dashboard.py)
#
#     python -m rl.dqn --total-timesteps 300000
#     python -m rl.dqn --curriculum 4 --per-alpha 0 --n-step 1     # plain Double DQN on hard
#     python -m rl.dqn --resume models/<run>-latest.pt --total-timesteps 5000000
#
# Stop a run cleanly at any time: Ctrl+C, or create the file models/<run>.stop
# (useful for runs in the background). Either way the latest state is saved first.
# Resuming with the same learning rate on the command line carries on with the
# learning rate the run had reached; a different one replaces it.
import copy
import dataclasses
import math
import os
import random
import socket
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from rl.evaluate import bayes_summary, eval_board_seeds, format_row, play_q, summarize
from rl.gpu import ThermalGuard
from rl.gpu import query as gpu_query
from rl.journal import Journal, read_status
from rl.minesweeper_env import MinesweeperEnv, RewardConfig, board_shape
from rl.per import PrioritizedReplayBuffer, ReplayBuffer
from rl.qnet import MODELS_DIR, QNetwork, count_params, greedy, load_checkpoint, save_checkpoint

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class Args:
    exp_name: str = "dqn"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    bf16: bool = True
    """MINESWEEPER: bf16 maths on the GPU (needs an RTX 30-series or newer); --no-bf16 for older cards"""

    # MINESWEEPER: the game
    curriculum: str = "2:200000,3:600000,4"
    """board levels and the timestep each one ends at; the last level is the target and runs to the end.
    1 test 6x6, 2 easy 9x9, 3 medium 16x16, 4 hard 16x30. "4" = hard only"""
    advance_at: float = 0.0
    """move to the next level early once an evaluation's win rate reaches this share of bayes' (0 = off)"""
    reward_win: float = 1.0
    """reward for clearing the board, the true goal (see RewardConfig)"""
    reward_progress: float = 1.0
    """paid for uncovering cells, as a share of the board's safe cells; a mine takes back the unbanked part"""
    reward_credit: float = 0.0
    """share of the progress a lost game keeps (0 = winning is the only goal)"""
    reward_mine: float = 0.0
    """penalty for hitting a mine (with gamma 0 and no other terms: learn each cell's chance of being a mine)"""
    reward_reveal: float = 0.0
    """paid on each safe click: this times the share of the board's safe cells it uncovered"""
    reveal_from: int = 0
    """MINESWEEPER: the click count at which the reveal reward starts (a second stage; 0 = from the start).
    At the switch the replay memory is emptied (its clicks were scored without it), the learning rate
    restarts, and the network as it was is kept as models/<run>-stage1.pt"""
    stage2_mine: float = 0.0
    """MINESWEEPER: the mine penalty from --reveal-from on (0 = keep --reward-mine)"""
    reward_safe: float = 0.0
    """paid on each safe click, a flat amount (negative = a small cost per click)"""
    reward_late: float = 0.0
    """paid on each safe click: this times (g_after^late_power - g_before^late_power), g = share of safe cells uncovered"""
    late_power: float = 1.0
    """how strongly the late reward favours the end of the game (1 = evenly)"""
    reward_burst: float = 0.0
    """paid on each safe click, growing exponentially with the share of the board it opens (all of it: this much)"""
    burst_rate: float = 5.0
    """how steeply the burst reward grows"""
    wall: float = 0.0
    """MINESWEEPER: refuse to train unless a mine costs at least this many times the most any click can earn (0 = no check)"""
    skip_coin_flips: bool = False
    """MINESWEEPER: clicks on coin flips (forced guesses with no better cell, found with the exact solver) cost
    nothing and are left out of the replay memory; needs --n-step 1"""
    stage2_from: int = 0
    """MINESWEEPER: the click count at which --stage2 takes over (0 = no second stage); at the switch the replay
    memory is emptied, the learning rate restarts, and the network as it was is kept as models/<run>-stage1.pt"""
    stage2: str = ""
    """MINESWEEPER: the reward settings that change at --stage2-from, e.g. "mine=100,safe=-0.01,late=10,late_power=10" """

    # MINESWEEPER: the network
    channels: int = 64
    """channels per conv layer"""
    layers: int = 16
    """conv layers (all but the first have a residual shortcut)"""
    pool_every: int = 0
    """every this many residual layers also see the whole board (global pooling, as in KataGo); 0 = off"""
    loss: str = "huber"
    """huber, or bce: each click's value is a logit between the worst and the best outcome of a game
    (minus reward_mine and reward_win), learned with cross-entropy; needs reward_progress 0"""

    # Algorithm specific arguments
    total_timesteps: int = 300_000
    """total timesteps (clicks) of the experiment"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    buffer_size: int = 200_000
    """the replay memory buffer size"""
    gamma: float = 0.999
    """the discount factor gamma (MINESWEEPER: near 1, so winning counts the same however late it comes)"""
    n_step: int = 3
    """MINESWEEPER: rewards summed per learning target (1 = plain one-step DQN)"""
    per_alpha: float = 0.6
    """MINESWEEPER: how strongly replay favours surprising clicks (0 = uniform replay)"""
    per_beta_start: float = 0.4
    """MINESWEEPER: importance-weight correction at the start; rises to 1 by the end"""
    per_eps: float = 1e-3
    """MINESWEEPER: added to every priority so every click can still be replayed"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 5_000
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.02
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.4
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10_000
    """timestep to start learning"""
    train_frequency: int = 16
    """timesteps per gradient step"""

    # MINESWEEPER: measuring, saving, stopping
    eval_every: int = 50_000
    """timesteps between evaluations on fixed boards"""
    eval_games: int = 300
    """games per evaluation"""
    checkpoint_every: int = 25_000
    """timesteps between saves of the latest state"""
    status_every: float = 5.0
    """seconds between updates of the run's status for the dashboard"""
    resume: str = ""
    """continue from this checkpoint (the replay buffer starts empty again)"""
    init: str = ""
    """MINESWEEPER: start a new run from this checkpoint's network (weights only)"""
    lr_patience: int = 0
    """MINESWEEPER: evaluations without a new best before the learning rate drops (0 = never)"""
    lr_decay: float = 0.5
    """MINESWEEPER: the learning rate is multiplied by this when it drops"""
    lr_min: float = 1e-6
    """MINESWEEPER: the learning rate never drops below this"""
    lr_mode: str = "halve"
    """MINESWEEPER: halve (after lr_patience evaluations without a new best), or restart: the learning rate decays
    exponentially from --learning-rate towards --lr-floor, and jumps back up after lr_patience evaluations
    without a new best"""
    lr_floor: float = 1e-5
    """MINESWEEPER: where the restart schedule decays to and holds"""
    lr_decay_clicks: int = 200_000
    """MINESWEEPER: the restart schedule's decay time (clicks to fall about two thirds of the way to the floor)"""
    ema: float = 0.0
    """MINESWEEPER: act, evaluate and save a moving average of the weights, updated by this
    much per gradient step (e.g. 0.999; 0 = off)"""
    augment: bool = False
    """MINESWEEPER: also learn from flipped (and, on square boards, rotated) copies of replayed boards"""
    eval_breakdown: bool = False
    """MINESWEEPER: evaluations also judge every lost game with the exact solver (missed safe cell, riskier guess,
    coin flip, bad luck) and report the win rate without coin flips; measuring only, a little slower"""
    max_gpu_temp: float = 0
    """MINESWEEPER: slow down while the GPU is hotter than this, in degrees C (0 = off)"""


def parse_curriculum(text: str) -> list[tuple[int, int | None]]:
    """"2:200000,3:600000,4" -> [(2, 200000), (3, 600000), (4, None)]"""
    stages = []
    for part in str(text).split(","):
        level, _, until = part.strip().partition(":")
        stages.append((int(level), int(until) if until else None))
    if stages[-1][1] is not None:
        raise ValueError("the last curriculum level runs to the end; give it no end step")
    return stages


class NStep:
    """MINESWEEPER: turns each game's stream of one-step transitions into n-step ones:
    (obs, action, r_0 + g r_1 + ... + g^(k-1) r_(k-1), obs after k clicks, g^k, or 0 if the game ended)."""

    def __init__(self, n: int, gamma: float, num_envs: int) -> None:
        self.n, self.gamma = n, gamma
        self.queues = [deque() for _ in range(num_envs)]

    def push(self, i: int, obs, action, reward, next_obs, terminated: bool, truncated: bool) -> list[tuple]:
        q = self.queues[i]
        q.append((obs, action, reward))
        ended = terminated or truncated
        out = []
        while q and (ended or len(q) == self.n):
            ret = sum(r * self.gamma ** k for k, (_, _, r) in enumerate(q))
            discount = 0.0 if terminated else self.gamma ** len(q)
            out.append((q[0][0], q[0][1], ret, next_obs, discount))
            q.popleft()
        return out


def mirror(obs, next_obs, actions, rng):
    """MINESWEEPER: a board seen in a mirror is just as valid a game. Flips the whole batch
    up-down and/or left-right, and swaps rows and columns of square boards, at random
    (8 ways on a square board, 4 otherwise); the clicked cell moves with the board."""
    rows, cols = obs.shape[-2:]
    r, c = actions // cols, actions % cols
    if rng.random() < 0.5:
        obs, next_obs, r = obs.flip(-2), next_obs.flip(-2), rows - 1 - r
    if rng.random() < 0.5:
        obs, next_obs, c = obs.flip(-1), next_obs.flip(-1), cols - 1 - c
    if rows == cols and rng.random() < 0.5:
        obs, next_obs, r, c = obs.transpose(-2, -1), next_obs.transpose(-2, -1), c, r
    return obs.contiguous(), next_obs.contiguous(), r * cols + c


def value_range(reward: RewardConfig) -> tuple[float, float]:
    """MINESWEEPER: the worst and the best a click can score under `reward` (the cross-entropy loss maps
    values into this range): a mine at worst, at best a win plus everything a safe click can earn."""
    return min(-reward.mine, reward.safe), reward.win + reward.reveal + reward.late + max(reward.safe, 0.0)


def make_env(level, seed, reward, canvas, skip_coin_flips=False):
    def thunk():
        return MinesweeperEnv(level=level, reward=reward, seed=seed, canvas=canvas,   # MINESWEEPER
                              skip_coin_flips=skip_coin_flips)

    return thunk


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def train(args: Args) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and args.bf16)
    torch.backends.cudnn.benchmark = True

    # MINESWEEPER: resume, or a new run
    ckpt = None
    if args.resume:
        _, ckpt = load_checkpoint(args.resume, device)
        run_name = ckpt["run_name"]
    else:
        run_name = f"{args.exp_name}-c{args.channels}x{args.layers}-{time.strftime('%Y%m%d-%H%M%S')}"
    latest, best = MODELS_DIR / f"{run_name}-latest.pt", MODELS_DIR / f"{run_name}-best.pt"
    stop_file = MODELS_DIR / f"{run_name}.stop"
    stop_file.unlink(missing_ok=True)

    # MINESWEEPER: curriculum; every board is drawn on the largest board's canvas
    stages = parse_curriculum(args.curriculum)
    target_level = stages[-1][0]
    canvas = tuple(max(board_shape(lv)[k] for lv, _ in stages) for k in (0, 1))
    stage = ckpt.get("stage", 0) if ckpt else 0
    # MINESWEEPER: on resume, exploration carries on from where it was instead of restarting
    # with the new total, and the games are dealt from fresh seeds instead of replaying the first ones
    if ckpt:
        explore_steps = ckpt.get("explore_steps") or ckpt["args"]["exploration_fraction"] * ckpt["args"]["total_timesteps"]
    else:
        explore_steps = args.exploration_fraction * args.total_timesteps
    seed_offset = ckpt["global_step"] if ckpt else 0

    # MINESWEEPER: the run's journal for the dashboard; a run without one starts it from its history
    journal = Journal(RUNS_DIR / run_name)
    if journal.new and ckpt:
        journal.backfill(ckpt.get("eval_history", []), RUNS_DIR / run_name)
    journal.status(**{**(read_status(journal.dir) or {}), "state": "starting", "run": run_name, "pid": os.getpid(),
                      "host": socket.gethostname(), "started": time.time()})

    writer = SummaryWriter(str(RUNS_DIR / run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # env setup
    # MINESWEEPER: the reward; with --reveal-from the reveal part starts later (a second stage)
    base = RewardConfig(win=args.reward_win, progress=args.reward_progress, credit=args.reward_credit,
                        mine=args.reward_mine, reveal=args.reward_reveal, safe=args.reward_safe, late=args.reward_late,
                        late_power=args.late_power, gamma=args.gamma)
    if args.stage2:                    # --stage2-from with --stage2 "name=value,..."
        switch_at = args.stage2_from
        stage1, full_reward = base, dataclasses.replace(base, **{
            k.strip(): float(v) for k, v in (item.split("=") for item in args.stage2.split(","))})
    else:                              # the older form: --reveal-from (and --stage2-mine)
        switch_at = args.reveal_from
        stage1 = dataclasses.replace(base, reveal=0.0) if args.reveal_from else base
        full_reward = dataclasses.replace(base, mine=args.stage2_mine or base.mine)
    reward = full_reward if seed_offset >= switch_at else stage1
    for r in (stage1, full_reward):    # MINESWEEPER: the wall rule
        if args.wall and r.mine < args.wall * r.max_click():
            raise ValueError(f"--wall {args.wall:g}: a mine costs {r.mine:g}, but a click can earn up to "
                             f"{r.max_click():g} (needs at least {args.wall * r.max_click():g})")
    if args.skip_coin_flips and args.n_step != 1:
        raise ValueError("--skip-coin-flips needs --n-step 1")
    envs = gym.vector.SyncVectorEnv(
        [make_env(stages[stage][0], args.seed * 10_000 + i + seed_offset, reward, canvas, args.skip_coin_flips)
         for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # MINESWEEPER: a new run can start from another run's network; layers it lacks (global pooling) start switched off
    init = load_checkpoint(args.init, "cpu")[1] if args.init and not ckpt else None
    if ckpt:
        config = ckpt["config"]
    elif init:
        config = {**init["config"], **({"pool_every": args.pool_every} if args.pool_every else {})}
    else:
        config = {"channels": args.channels, "layers": args.layers, "pool_every": args.pool_every}
    q_network = QNetwork(config).to(device)
    if init:
        added = q_network.load_state_dict(init["state_dict"], strict=False).missing_keys
        if added:
            print(f"started from {args.init}; new layers, switched off at first: {', '.join(added)}", flush=True)

    # MINESWEEPER: with the cross-entropy loss the network's outputs are logits of the value,
    # which runs from the worst outcome (a mine) to the best (a win)
    if args.loss == "bce":
        if args.reward_progress != 0:
            raise ValueError("--loss bce needs --reward-progress 0 (values must stay between -mine and win)")
        lo, hi = value_range(reward)
    elif args.loss != "huber":
        raise ValueError(f"unknown --loss {args.loss!r}")

    def to_value(z: torch.Tensor) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(z) if args.loss == "bce" else z
    if amp["enabled"]:   # MINESWEEPER: channels-last memory helps bf16 on newer GPUs, but is 1.7x slower in fp32 on a GTX 1050
        q_network = q_network.to(memory_format=torch.channels_last)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = copy.deepcopy(q_network)
    # MINESWEEPER: the network that plays (and is saved and evaluated): the weights' moving average, or the weights
    policy = copy.deepcopy(q_network) if args.ema > 0 else q_network
    global_step, eval_history, best_eval = 0, [], None
    restart_step = 0   # MINESWEEPER: where the restart learning-rate schedule last jumped up
    lr, evals_since_best = args.learning_rate, 0
    if ckpt:
        q_network.load_state_dict(ckpt.get("online") or ckpt["state_dict"])
        policy.load_state_dict(ckpt["state_dict"])
        target_network.load_state_dict(ckpt["target"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # MINESWEEPER: the same learning rate on the command line carries on with the one the run
        # had reached (it may have dropped since); a different one replaces it
        if ckpt.get("lr") is not None and args.learning_rate == ckpt["args"]["learning_rate"]:
            lr, evals_since_best = ckpt["lr"], ckpt.get("evals_since_best", 0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        global_step, eval_history = ckpt["global_step"], ckpt.get("eval_history", [])
        restart_step = ckpt.get("restart_step", global_step)
        # MINESWEEPER: the best so far counts only evaluations comparable with the ones to come:
        # same boards, same reward
        comparable = [e for e in eval_history if e.get("level") == target_level and e.get("games") == args.eval_games
                      and e.get("reward") == asdict(reward)]
        best_eval = max(comparable, key=lambda e: (e["win"], e["revealed"]), default=None)
    best_score = (best_eval["win"], best_eval["revealed"]) if best_eval else (-1.0, -1.0)
    learning_starts = global_step + args.learning_starts
    print(f"run {run_name}: {count_params(q_network):,} weights, {args.num_envs} games at once, on {device}; "
          f"curriculum {stages}, canvas {canvas}, {args.n_step}-step returns, "
          f"replay {'prioritized (alpha %.2f)' % args.per_alpha if args.per_alpha > 0 else 'uniform'}"
          + (f", resumed at step {global_step:,}" if ckpt else ""), flush=True)

    # MINESWEEPER: OpenAI Baselines' prioritized replay; multi-step transitions go in
    rb = PrioritizedReplayBuffer(args.buffer_size, alpha=args.per_alpha) if args.per_alpha > 0 else ReplayBuffer(args.buffer_size)
    nstep = NStep(args.n_step, args.gamma, args.num_envs)

    # MINESWEEPER: fixed evaluation boards per level, and bayes on them (measuring only)
    bayes: dict[int, dict] = {k: v for k, v in (ckpt or {}).get("bayes", {}).items() if isinstance(k, int)}

    def bayes_on(level: int) -> dict:
        if bayes.get(level, {}).get("games") != args.eval_games:
            bayes[level] = bayes_summary(level, args.eval_games)
            print(format_row(f"bayes, level {level} (same boards)", bayes[level]), flush=True)
        return bayes[level]

    bayes_on(target_level)

    def save(path: Path) -> None:
        save_checkpoint(path, policy, {
            "target": target_network.state_dict(), "optimizer": optimizer.state_dict(),
            **({"online": q_network.state_dict()} if policy is not q_network else {}),
            "global_step": global_step, "run_name": run_name, "args": vars(args), "stage": stage,
            "canvas": canvas, "reward": asdict(reward), "eval_history": eval_history, "bayes": bayes,
            "explore_steps": explore_steps, "lr": lr, "evals_since_best": evals_since_best, "restart_step": restart_step,
        })

    # MINESWEEPER: the state the dashboard shows, rewritten at every progress line
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    session_start = time.time()

    def report(state: str, **extra) -> None:
        journal.status(
            state=state, run=run_name, pid=os.getpid(), host=socket.gethostname(), device=gpu_name,
            started=session_start, session_step=start_step, step=global_step, total_steps=args.total_timesteps,
            level=stages[stage][0], target_level=target_level, epsilon=epsilon, lr=lr, lr_patience=args.lr_patience,
            lr_decay=args.lr_decay, lr_min=args.lr_min, evals_since_best=evals_since_best, sps=sps,
            sps_recent=(global_step - pace[0][1]) / (time.time() - pace[0][0]) if len(pace) > 1 else None,
            games={k: float(np.mean(v)) for k, v in episodes.items() if v}, td=td_loss, q=q_mean,
            gpu_temp=gpu_temp, gpu_pause=guard.pause, max_gpu_temp=args.max_gpu_temp,
            last_eval=eval_history[-1] if eval_history else None, best=best_eval, bayes=bayes,
            eval_games=args.eval_games, eval_every=args.eval_every, next_eval=next_eval,
            reward=asdict(reward), bf16=args.bf16, model={"channels": config["channels"], "layers": config["layers"],
                                                         "weights": count_params(q_network)},
            checkpoints={"latest": str(latest), "best": str(best)}, **extra)

    def evaluate(level: int) -> dict:
        nonlocal best_score, best_eval, evals_since_best, lr, restart_step
        base = bayes_on(level)
        report("evaluating")
        res = summarize(play_q(policy, level, eval_board_seeds(level, args.eval_games), device, canvas=canvas,
                               bf16=args.bf16, breakdown=args.eval_breakdown))
        res.update(step=global_step, level=level, lr=lr, reward=asdict(reward))
        eval_history.append(res)
        journal.log("eval", global_step, **{k: v for k, v in res.items() if k != "step"})
        print(format_row(f"eval L{level} at step {global_step:,}", res) + f"   (bayes {base['win']*100:.1f}%)",
              flush=True)
        for k in ("win", "efficiency", "revealed"):
            writer.add_scalar(f"eval_L{level}/{k}", res[k], global_step)
        writer.add_scalar(f"eval_L{level}/win_minus_bayes", res["win"] - base["win"], global_step)
        if level == target_level:
            if (res["win"], res["revealed"]) > best_score:
                best_score = (res["win"], res["revealed"])   # win rate first, then how much of the board it clears
                best_eval, evals_since_best = res, 0
                save(best)
            else:
                # MINESWEEPER: a plateau; each earlier drop of the learning rate brought a new climb
                evals_since_best += 1
                if args.lr_mode == "restart" and args.lr_patience and evals_since_best >= args.lr_patience:
                    restart_step, evals_since_best = global_step, 0     # back up to the peak, to decay again
                    print(f"learning rate back up to {args.learning_rate:.3g} from step {global_step:,}: no new best in "
                          f"{args.lr_patience} evaluations", flush=True)
                    journal.event(global_step, f"learning rate boost {args.learning_rate:.2g}", "lr")
                elif args.lr_mode != "restart" and args.lr_patience and evals_since_best >= args.lr_patience \
                        and lr > args.lr_min:
                    lr, evals_since_best = max(args.lr_min, lr * args.lr_decay), 0
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    print(f"learning rate {lr:.3g} from step {global_step:,}: no new best in "
                          f"{args.lr_patience} evaluations", flush=True)
                    journal.event(global_step, f"learning rate {lr:.2g}", "lr")
        report("training")
        return res

    def advance_stage() -> None:
        nonlocal stage
        stage += 1
        level = stages[stage][0]
        envs.call("set_level", level)       # each game switches once its current game ends
        print(f"curriculum: level {level} from step {global_step:,}", flush=True)
        journal.event(global_step, f"level {level}", "level")

    episodes = {k: deque(maxlen=1000) for k in ("return", "won", "revealed", "length")}
    ep_return = np.zeros(args.num_envs)
    start_time, start_step = time.time(), global_step
    next_eval = (global_step // args.eval_every + 1) * args.eval_every
    next_ckpt = (global_step // args.checkpoint_every + 1) * args.checkpoint_every
    next_print = global_step + 10_000
    next_status = time.time() + args.status_every
    pace = deque([(time.time(), global_step)], maxlen=max(2, int(60 / args.status_every) + 1))   # speed over the last minute
    next_target = global_step + args.target_network_frequency
    td_loss = q_mean = float("nan")
    stopped = "finished"
    # MINESWEEPER: GPU temperature guard; readings for the dashboard even when it is off
    guard = ThermalGuard(args.max_gpu_temp if device.type == "cuda" else 0)
    gpu_temp, sps = None, 0
    epsilon = linear_schedule(args.start_e, args.end_e, explore_steps, global_step)

    journal.event(global_step, f"{'resumed' if ckpt else 'started'} on {gpu_name}, learning rate {lr:.2g}", "session")
    if ckpt and ckpt.get("reward") != asdict(reward):
        journal.event(global_step, "new reward: " + ", ".join(f"{k} {v:g}" for k, v in asdict(reward).items()), "reward")
    if ckpt and lr != ckpt.get("lr", ckpt["args"]["learning_rate"]):
        journal.event(global_step, f"learning rate {lr:.2g}", "lr")
    report("training")

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed + seed_offset)
    try:
        while global_step < args.total_timesteps:
            # MINESWEEPER: curriculum by step count
            while stages[stage][1] is not None and global_step >= stages[stage][1]:
                advance_stage()
            # MINESWEEPER: second stage of the reward
            if reward != full_reward and global_step >= switch_at:
                save(MODELS_DIR / f"{run_name}-stage1.pt")
                reward = full_reward
                envs.call("set_reward", reward)
                if args.loss == "bce":
                    lo, hi = value_range(reward)
                rb = PrioritizedReplayBuffer(args.buffer_size, alpha=args.per_alpha) if args.per_alpha > 0 \
                    else ReplayBuffer(args.buffer_size)
                learning_starts = global_step + args.learning_starts
                lr, evals_since_best, best_eval, best_score = args.learning_rate, 0, None, (-1.0, -1.0)
                restart_step = global_step
                for group in optimizer.param_groups:
                    group["lr"] = lr
                changed = ", ".join(f"{k} {v:g}" for k, v in asdict(reward).items() if v != asdict(stage1)[k])
                print(f"stage 2 from step {global_step:,}: {changed}; replay memory emptied, learning rate {lr:.2g}; "
                      f"stage 1 kept as {run_name}-stage1.pt", flush=True)
                journal.event(global_step, f"stage 2: {changed}", "reward")

            if args.lr_mode == "restart":   # MINESWEEPER: decay from the peak towards the floor
                lr = args.lr_floor + (args.learning_rate - args.lr_floor) * math.exp(
                    -(global_step - restart_step) / args.lr_decay_clicks)
                for group in optimizer.param_groups:
                    group["lr"] = lr

            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(args.start_e, args.end_e, explore_steps, global_step)
            # MINESWEEPER: greedy over legal cells; exploring picks a random legal cell
            obs_t = torch.from_numpy(obs).to(device)
            with torch.no_grad(), torch.autocast(**amp):
                actions = greedy(policy(obs_t).float(), obs_t).cpu().numpy()
            for i in np.flatnonzero(rng.random(args.num_envs) < epsilon):
                actions[i] = rng.choice(np.flatnonzero(obs[i, 0].ravel() == 0))

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            ep_return += rewards

            # MINESWEEPER: episode statistics from the games that just ended
            if "final_info" in infos:
                fi = infos["final_info"]
                for i in np.flatnonzero(terminations | truncations):
                    episodes["return"].append(ep_return[i])
                    episodes["won"].append(fi["won"][i])
                    episodes["revealed"].append(fi["revealed_fraction"][i])
                    episodes["length"].append(fi["moves"][i])
                    ep_return[i] = 0.0

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_obs`
            real_next_obs = next_obs.copy()
            for idx in np.flatnonzero(terminations | truncations):
                real_next_obs[idx] = infos["final_obs"][idx]
            for i in range(args.num_envs):   # MINESWEEPER: multi-step transitions
                if args.skip_coin_flips:     # MINESWEEPER: a coin flip teaches nothing; leave it out
                    ended = terminations[i] or truncations[i]
                    if (infos["final_info"]["coin_flip"][i] if ended else infos["coin_flip"][i]):
                        continue
                for transition in nstep.push(i, obs[i], int(actions[i]), float(rewards[i]), real_next_obs[i],
                                             bool(terminations[i]), bool(truncations[i])):
                    rb.add(*transition)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            global_step += args.num_envs

            # ALGO LOGIC: training.
            if global_step > learning_starts:
                beta = min(1.0, args.per_beta_start + (1.0 - args.per_beta_start) * global_step / args.total_timesteps)
                for _ in range(max(1, args.num_envs // args.train_frequency)):
                    data = rb.sample(args.batch_size, beta=beta) if args.per_alpha > 0 else rb.sample(args.batch_size)
                    b_obs      = torch.from_numpy(data["obs"]).to(device)
                    b_next_obs = torch.from_numpy(data["next_obs"]).to(device)
                    b_actions  = torch.from_numpy(data["action"]).to(device)
                    if args.augment:   # MINESWEEPER
                        b_obs, b_next_obs, b_actions = mirror(b_obs, b_next_obs, b_actions, rng)
                    b_rewards  = torch.from_numpy(data["reward"]).to(device)
                    b_discount = torch.from_numpy(data["discount"]).to(device)
                    with torch.no_grad(), torch.autocast(**amp):
                        if args.gamma > 0:
                            # MINESWEEPER: Double DQN -- online net picks the next cell, target net scores it
                            next_actions = greedy(q_network(b_next_obs).float(), b_next_obs)
                            next_q = target_network(b_next_obs).float().gather(1, next_actions[:, None]).squeeze(1)
                            td_target = b_rewards + b_discount * to_value(next_q)
                        else:   # MINESWEEPER: no discount, nothing to look ahead to
                            td_target = b_rewards
                    with torch.autocast(**amp):
                        old_val = q_network(b_obs).float().gather(1, b_actions[:, None]).squeeze(1)
                    # MINESWEEPER: Huber (or cross-entropy), weighted for prioritized replay; new priorities = |TD error|
                    if args.loss == "bce":
                        share = ((td_target - lo) / (hi - lo)).clamp(0, 1)   # where the target sits between worst and best
                        elementwise = F.binary_cross_entropy_with_logits(old_val, share, reduction="none")
                    else:
                        elementwise = F.smooth_l1_loss(old_val, td_target, reduction="none")
                    if args.per_alpha > 0:
                        loss = (torch.from_numpy(data["weights"]).to(device) * elementwise).mean()
                        rb.update_priorities(data["indexes"],
                                             (td_target - to_value(old_val)).abs().detach().cpu().numpy() + args.per_eps)
                    else:
                        loss = elementwise.mean()

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)   # MINESWEEPER
                    optimizer.step()
                    if policy is not q_network:   # MINESWEEPER: the moving average follows
                        with torch.no_grad():
                            torch._foreach_lerp_(list(policy.parameters()), list(q_network.parameters()), 1 - args.ema)
                td_loss, q_mean = loss.item(), to_value(old_val.detach()).mean().item()

                # update target network
                if global_step >= next_target:
                    next_target += args.target_network_frequency
                    for target_network_param, q_network_param in zip(target_network.parameters(),
                                                                     q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )

            # MINESWEEPER: logging, evaluation, checkpoints, stop file
            if global_step >= next_print:
                next_print += 10_000
                sps = int((global_step - start_step) / (time.time() - start_time))
                if args.max_gpu_temp > 0:   # MINESWEEPER: GPU temperature for the log and the dashboard
                    gpu_temp = guard.temp
                elif device.type == "cuda":
                    gpu_temp = (gpu_query() or {}).get("temp")
                n = len(episodes["won"])
                if n:
                    print(f"step {global_step:>9,}  L{stages[stage][0]}  eps {epsilon:.2f}  return {np.mean(episodes['return']):+.3f}"
                          f"  won {np.mean(episodes['won'])*100:5.1f}%  revealed {np.mean(episodes['revealed'])*100:5.1f}%"
                          f"  clicks {np.mean(episodes['length']):5.1f}  td {td_loss:.4f}  q {q_mean:+.3f}  {sps} steps/s"
                          + (f"  gpu {gpu_temp:.0f}C" if gpu_temp is not None else "")
                          + (f" (slowed {guard.pause:.1f}s/step)" if guard.pause else ""),
                          flush=True)
                    writer.add_scalar("charts/episodic_return", np.mean(episodes["return"]), global_step)
                    writer.add_scalar("charts/win_rate", np.mean(episodes["won"]), global_step)
                    writer.add_scalar("charts/revealed_fraction", np.mean(episodes["revealed"]), global_step)
                    writer.add_scalar("charts/episodic_length", np.mean(episodes["length"]), global_step)
                    journal.log("train", global_step, win=np.mean(episodes["won"]),   # MINESWEEPER: the dashboard's copy
                                revealed=np.mean(episodes["revealed"]), clicks=np.mean(episodes["length"]),
                                td=td_loss, q=q_mean, sps=sps, lr=lr, epsilon=epsilon, level=stages[stage][0],
                                gpu_temp=gpu_temp, gpu_pause=guard.pause, **{"return": np.mean(episodes["return"])})
                writer.add_scalar("charts/level", stages[stage][0], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/learning_rate", lr, global_step)
                writer.add_scalar("losses/td_loss", td_loss, global_step)
                writer.add_scalar("losses/q_values", q_mean, global_step)
                report("training")
            if global_step >= next_eval:
                next_eval += args.eval_every
                level = stages[stage][0]
                res = evaluate(level)
                if level != target_level:
                    evaluate(target_level)
                    if args.advance_at > 0 and res["win"] >= args.advance_at * bayes[level]["win"]:
                        advance_stage()
            if global_step >= next_ckpt:
                next_ckpt += args.checkpoint_every
                save(latest)
            if stop_file.exists():
                stopped = "stopped by stop file"
                break
            if time.time() >= next_status:   # MINESWEEPER: the dashboard's live numbers
                next_status = time.time() + args.status_every
                pace.append((time.time(), global_step))
                report("training")
            guard()   # MINESWEEPER
    except KeyboardInterrupt:
        stopped = "interrupted"

    report("saving", stop_reason=stopped)
    save(latest)
    stop_file.unlink(missing_ok=True)
    envs.close()
    writer.close()
    journal.event(global_step, stopped, "stop")
    report("stopped", stop_reason=stopped)
    print(f"{stopped} at step {global_step:,}. latest: {latest}"
          + (f"  best: {best} ({best_score[0]*100:.1f}% wins, {best_score[1]*100:.1f}% revealed on the "
             f"level {target_level} eval boards)" if best_eval else ""), flush=True)
    return latest


if __name__ == "__main__":
    train(tyro.cli(Args))
