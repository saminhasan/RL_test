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
#
#     python -m rl.dqn --total-timesteps 300000
#     python -m rl.dqn --curriculum 4 --per-alpha 0 --n-step 1     # plain Double DQN on hard
#     python -m rl.dqn --resume models/<run>-latest.pt --total-timesteps 5000000
#
# Stop a run cleanly at any time: Ctrl+C, or create the file models/<run>.stop
# (useful for runs in the background). Either way the latest state is saved first.
import copy
import random
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

from rl.evaluate import eval_board_seeds, format_row, play_bayes, play_q, summarize
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
    reward_progress: float = 0.5
    """reward for uncovering cells, as a share of the board's safe cells (see RewardConfig)"""
    reward_click: float = 0.25
    """cost per click, as a share of the board's safe cells"""
    reward_win: float = 1.0
    """reward for clearing the board"""
    reward_mine: float = 0.5
    """penalty for hitting a mine"""

    # MINESWEEPER: the network
    channels: int = 64
    """channels per conv layer"""
    layers: int = 16
    """conv layers (all but the first have a residual shortcut)"""

    # Algorithm specific arguments
    total_timesteps: int = 300_000
    """total timesteps (clicks) of the experiment"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    buffer_size: int = 200_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
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
    resume: str = ""
    """continue from this checkpoint (the replay buffer starts empty again)"""


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


def make_env(level, seed, reward, canvas):
    def thunk():
        return MinesweeperEnv(level=level, reward=reward, seed=seed, canvas=canvas)   # MINESWEEPER

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
    reward = RewardConfig(args.reward_progress, args.reward_click, args.reward_win, args.reward_mine)
    envs = gym.vector.SyncVectorEnv(
        [make_env(stages[stage][0], args.seed * 10_000 + i + seed_offset, reward, canvas) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    config = ckpt["config"] if ckpt else {"channels": args.channels, "layers": args.layers}
    q_network = QNetwork(config).to(device).to(memory_format=torch.channels_last)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = copy.deepcopy(q_network)
    global_step, eval_history, best_score = 0, [], (-1.0, -1.0)
    if ckpt:
        q_network.load_state_dict(ckpt["state_dict"])
        target_network.load_state_dict(ckpt["target"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for group in optimizer.param_groups:      # MINESWEEPER: the learning rate given now wins on resume
            group["lr"] = args.learning_rate
        global_step, eval_history = ckpt["global_step"], ckpt.get("eval_history", [])
        best_score = max([(e["win"], e["revealed"]) for e in eval_history if e.get("level") == target_level],
                         default=(-1.0, -1.0))
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
        if level not in bayes:
            bayes[level] = summarize(play_bayes(level, eval_board_seeds(level, args.eval_games)))
            print(format_row(f"bayes, level {level} (same boards)", bayes[level]), flush=True)
        return bayes[level]

    bayes_on(target_level)

    def save(path: Path) -> None:
        save_checkpoint(path, q_network, {
            "target": target_network.state_dict(), "optimizer": optimizer.state_dict(),
            "global_step": global_step, "run_name": run_name, "args": vars(args), "stage": stage,
            "canvas": canvas, "reward": asdict(reward), "eval_history": eval_history, "bayes": bayes,
            "explore_steps": explore_steps,
        })

    def evaluate(level: int) -> dict:
        nonlocal best_score
        base = bayes_on(level)
        res = summarize(play_q(q_network, level, eval_board_seeds(level, args.eval_games), device, canvas=canvas,
                               bf16=args.bf16))
        res.update(step=global_step, level=level)
        eval_history.append(res)
        print(format_row(f"eval L{level} at step {global_step:,}", res) + f"   (bayes {base['win']*100:.1f}%)",
              flush=True)
        for k in ("win", "efficiency", "revealed"):
            writer.add_scalar(f"eval_L{level}/{k}", res[k], global_step)
        writer.add_scalar(f"eval_L{level}/win_minus_bayes", res["win"] - base["win"], global_step)
        if level == target_level and (res["win"], res["revealed"]) > best_score:
            best_score = (res["win"], res["revealed"])   # win rate first, then how much of the board it clears
            save(best)
        return res

    def advance_stage() -> None:
        nonlocal stage
        stage += 1
        level = stages[stage][0]
        envs.call("set_level", level)       # each game switches once its current game ends
        print(f"curriculum: level {level} from step {global_step:,}", flush=True)

    episodes = {k: deque(maxlen=1000) for k in ("return", "won", "revealed", "length")}
    ep_return = np.zeros(args.num_envs)
    start_time, start_step = time.time(), global_step
    next_eval = (global_step // args.eval_every + 1) * args.eval_every
    next_ckpt = (global_step // args.checkpoint_every + 1) * args.checkpoint_every
    next_print = global_step + 10_000
    next_target = global_step + args.target_network_frequency
    td_loss = q_mean = float("nan")
    stopped = "finished"

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed + seed_offset)
    try:
        while global_step < args.total_timesteps:
            # MINESWEEPER: curriculum by step count
            while stages[stage][1] is not None and global_step >= stages[stage][1]:
                advance_stage()

            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(args.start_e, args.end_e, explore_steps, global_step)
            # MINESWEEPER: greedy over legal cells; exploring picks a random legal cell
            obs_t = torch.from_numpy(obs).to(device)
            with torch.no_grad(), torch.autocast(**amp):
                actions = greedy(q_network(obs_t).float(), obs_t).cpu().numpy()
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
                    b_rewards  = torch.from_numpy(data["reward"]).to(device)
                    b_discount = torch.from_numpy(data["discount"]).to(device)
                    with torch.no_grad(), torch.autocast(**amp):
                        # MINESWEEPER: Double DQN -- online net picks the next cell, target net scores it
                        next_actions = greedy(q_network(b_next_obs).float(), b_next_obs)
                        next_q = target_network(b_next_obs).float().gather(1, next_actions[:, None]).squeeze(1)
                        td_target = b_rewards + b_discount * next_q
                    with torch.autocast(**amp):
                        old_val = q_network(b_obs).float().gather(1, b_actions[:, None]).squeeze(1)
                    # MINESWEEPER: Huber, weighted for prioritized replay; new priorities = |TD error|
                    elementwise = F.smooth_l1_loss(old_val, td_target, reduction="none")
                    if args.per_alpha > 0:
                        loss = (torch.from_numpy(data["weights"]).to(device) * elementwise).mean()
                        rb.update_priorities(data["indexes"],
                                             (td_target - old_val).abs().detach().cpu().numpy() + args.per_eps)
                    else:
                        loss = elementwise.mean()

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)   # MINESWEEPER
                    optimizer.step()
                td_loss, q_mean = loss.item(), old_val.mean().item()

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
                n = len(episodes["won"])
                if n:
                    print(f"step {global_step:>9,}  L{stages[stage][0]}  eps {epsilon:.2f}  return {np.mean(episodes['return']):+.3f}"
                          f"  won {np.mean(episodes['won'])*100:5.1f}%  revealed {np.mean(episodes['revealed'])*100:5.1f}%"
                          f"  clicks {np.mean(episodes['length']):5.1f}  td {td_loss:.4f}  q {q_mean:+.3f}  {sps} steps/s",
                          flush=True)
                    writer.add_scalar("charts/episodic_return", np.mean(episodes["return"]), global_step)
                    writer.add_scalar("charts/win_rate", np.mean(episodes["won"]), global_step)
                    writer.add_scalar("charts/revealed_fraction", np.mean(episodes["revealed"]), global_step)
                    writer.add_scalar("charts/episodic_length", np.mean(episodes["length"]), global_step)
                writer.add_scalar("charts/level", stages[stage][0], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("losses/td_loss", td_loss, global_step)
                writer.add_scalar("losses/q_values", q_mean, global_step)
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
    except KeyboardInterrupt:
        stopped = "interrupted"

    save(latest)
    stop_file.unlink(missing_ok=True)
    envs.close()
    writer.close()
    print(f"{stopped} at step {global_step:,}. latest: {latest}"
          + (f"  best: {best} ({best_score[0]*100:.1f}% wins, {best_score[1]*100:.1f}% revealed on the "
             f"level {target_level} eval boards)" if best.exists() else ""), flush=True)
    return latest


if __name__ == "__main__":
    train(tyro.cli(Args))
