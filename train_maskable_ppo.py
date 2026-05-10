from __future__ import annotations

from dataclasses import asdict
import pprint

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from torch.utils.tensorboard import SummaryWriter

from engine import MinesweeperEngine
from minesweeper_env import (
    MinesweeperMaskEnv,
    encode_board,
    level_name,
    level_to_id,
    make_env,
    make_train_env,
)
from models import (
    LEVEL_NETWORKS,
    MaskablePPOConfig,
    best_model_path,
    latest_model_path,
    load_or_create_maskable_ppo,
    ppo_tensorboard_dir,
    rebuild_model_from_checkpoint,
    save_metadata,
)


TRAINING_CONFIG = MaskablePPOConfig(
    level="medium",
    seed=42,
    target_win_rate_pct=69.3,
    eval_games=4096,
    eval_every_timesteps=8192,

    learning_rate=1.0e-5,
    n_steps=2048,
    batch_size=512,
    n_epochs=1,

    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.03,
    ent_coef=0.0,
    vf_coef=0.25,
    max_grad_norm=0.25,

    n_envs=8,
    use_subproc_env=True,
    freeze_features_extractor=True,
    load_best_checkpoint=True,

    rollback_drop_pct=10.0,
    rollback_patience=3,
    device="auto",
)
ASK_FOR_LEVEL = True

RUN_ORACLE_WARMSTART = True
ORACLE_WARMSTART_MAX_BEST_WIN_RATE = 1.0
ORACLE_WARMSTART_EPOCHS = 0
ORACLE_WARMSTART_GAMES_PER_EPOCH = 200
ORACLE_WARMSTART_BATCH_SIZE = 512
ORACLE_WARMSTART_LEARNING_RATE = 1.0e-3
ORACLE_WARMSTART_EVAL_EVERY = 2
ORACLE_WARMSTART_EVAL_GAMES = 512


def pick_level(default_level: int | str) -> str:
    choices = ", ".join(f"{idx}:{name}" for idx, name in MinesweeperEngine.LEVELS.items())
    print(f"available levels: {choices}")
    answer = input(f"select level [{level_name(default_level)}]: ").strip()
    if not answer:
        return level_name(default_level)
    return level_name(level_to_id(answer))


def build_config() -> MaskablePPOConfig:
    level = pick_level(TRAINING_CONFIG.level) if ASK_FOR_LEVEL else level_name(TRAINING_CONFIG.level)
    return MaskablePPOConfig(
        level=level,
        seed=TRAINING_CONFIG.seed,
        target_win_rate_pct=TRAINING_CONFIG.target_win_rate_pct,
        eval_games=TRAINING_CONFIG.eval_games,
        eval_every_timesteps=TRAINING_CONFIG.eval_every_timesteps,
        max_timesteps=TRAINING_CONFIG.max_timesteps,
        learning_rate=TRAINING_CONFIG.learning_rate,
        n_steps=TRAINING_CONFIG.n_steps,
        batch_size=TRAINING_CONFIG.batch_size,
        n_epochs=TRAINING_CONFIG.n_epochs,
        gamma=TRAINING_CONFIG.gamma,
        gae_lambda=TRAINING_CONFIG.gae_lambda,
        clip_range=TRAINING_CONFIG.clip_range,
        ent_coef=TRAINING_CONFIG.ent_coef,
        vf_coef=TRAINING_CONFIG.vf_coef,
        max_grad_norm=TRAINING_CONFIG.max_grad_norm,
        n_envs=TRAINING_CONFIG.n_envs,
        use_subproc_env=TRAINING_CONFIG.use_subproc_env,
        freeze_features_extractor=TRAINING_CONFIG.freeze_features_extractor,
        load_best_checkpoint=TRAINING_CONFIG.load_best_checkpoint,
        rollback_drop_pct=TRAINING_CONFIG.rollback_drop_pct,
        rollback_patience=TRAINING_CONFIG.rollback_patience,
        device=TRAINING_CONFIG.device,
    )


def evaluate_model(
    model: MaskablePPO,
    config: MaskablePPOConfig,
    games: int,
    deterministic: bool = True,
) -> dict[str, float]:
    wins = 0
    losses = 0
    moves = 0
    won_moves = 0
    lost_moves = 0
    returns = 0.0

    for game in range(games):
        env = make_env(config, seed_offset=50_000_000 + game)
        obs, _ = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=env.action_masks(),
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_return += float(reward)
            done = terminated or truncated

        won = bool(info["won"])
        episode_moves = int(info["moves"])
        wins += int(won)
        losses += int(not won)
        moves += episode_moves
        if won:
            won_moves += episode_moves
        else:
            lost_moves += episode_moves
        returns += episode_return

    return {
        "win_rate": wins / games if games else 0.0,
        "loss_rate": losses / games if games else 0.0,
        "avg_moves": moves / games if games else 0.0,
        "avg_won_moves": won_moves / wins if wins else 0.0,
        "avg_lost_moves": lost_moves / losses if losses else 0.0,
        "wins": float(wins),
        "losses": float(losses),
        "avg_return": returns / games if games else 0.0,
    }


def print_training_setup(config: MaskablePPOConfig, env: MinesweeperMaskEnv) -> None:
    net = LEVEL_NETWORKS[config.level_name]
    print("MaskablePPO setup:")
    print(f"  level              : {config.level_name}")
    print(f"  board              : {env.engine.rows}x{env.engine.cols}")
    print(f"  input shape         : {env.observation_space.shape}")
    print(f"  action count        : {env.action_space.n}")
    print(f"  architecture        : {net.name}")
    print(f"  trunk channels      : {net.channels}")
    print(f"  plain conv layers   : {net.plain_conv_layers}")
    print(f"  residual blocks     : {net.residual_blocks}")
    print(f"  block dilations     : {net.block_dilations or 'none'}")
    print(f"  global context      : {net.global_context} (avg pool broadcast, no FC)")
    print(f"  policy head         : conv 1x1 -> board logits")
    print(f"  risk head           : conv 1x1 -> mine-risk auxiliary")
    print(f"  value head          : conv 1x1 -> spatial mean -> scalar")
    print(f"  training envs       : {config.n_envs}")
    print(f"  subproc envs        : {config.use_subproc_env and config.n_envs > 1}")
    print(f"  freeze cnn          : {config.freeze_features_extractor}")
    print(f"  load best           : {config.load_best_checkpoint}")
    print(f"  rollback drop       : {config.rollback_drop_pct:.1f}%")
    print(f"  rollback patience   : {config.rollback_patience}")
    print(f"  model dir           : {latest_model_path(config.level).parent}")
    print(f"  tensorboard         : {ppo_tensorboard_dir(config.level)}")


def write_eval_scalars(
    writer: SummaryWriter,
    prefix: str,
    stats: dict[str, float],
    step: int,
) -> None:
    for key, value in stats.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)
    if "win_rate" in stats:
        writer.add_scalar(f"{prefix}/win_rate_pct", stats["win_rate"] * 100.0, step)
    if "loss_rate" in stats:
        writer.add_scalar(f"{prefix}/loss_rate_pct", stats["loss_rate"] * 100.0, step)


def oracle_action(engine: MinesweeperEngine, rng: np.random.Generator) -> int:
    hidden = ~engine.revealed
    hidden_choices = np.flatnonzero(hidden.reshape(-1))
    if len(hidden_choices) == 0:
        return 0

    if not engine.started:
        return (engine.rows // 2) * engine.cols + (engine.cols // 2)

    probabilities = engine.get_mine_probabilities()
    finite_hidden = hidden & np.isfinite(probabilities)
    safe_hidden = finite_hidden & ~engine._mines

    if safe_hidden.any():
        scores = np.where(safe_hidden, probabilities, np.inf)
        lowest = float(np.nanmin(scores))
        choices = np.flatnonzero((scores == lowest).reshape(-1))
        return int(choices[int(rng.integers(len(choices)))])

    hidden_safe = np.flatnonzero((hidden & ~engine._mines).reshape(-1))
    if len(hidden_safe) > 0:
        return int(hidden_safe[int(rng.integers(len(hidden_safe)))])

    return int(hidden_choices[int(rng.integers(len(hidden_choices)))])


def collect_oracle_examples(
    config: MaskablePPOConfig,
    epoch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed + epoch * 1_000_003)
    level_id = level_to_id(config.level)
    states: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []

    for game in range(ORACLE_WARMSTART_GAMES_PER_EPOCH):
        seed = config.seed + epoch * 1_000_003 + game
        engine = MinesweeperEngine(level=level_id, seed=seed, headless=True)

        while engine.state != MinesweeperEngine.OVER:
            action = oracle_action(engine, rng)
            states.append(encode_board(engine.get_public_view(reveal_mines_on_loss=False)))
            masks.append((~engine.revealed).reshape(-1).astype(bool))
            actions.append(action)

            row, col = divmod(action, engine.cols)
            engine.reveal_count(row, col)

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(masks, dtype=bool),
        np.asarray(actions, dtype=np.int64),
    )


def imitation_epoch(
    model: MaskablePPO,
    optimizer: torch.optim.Optimizer,
    states: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
) -> tuple[float, float]:
    device = model.policy.device
    model.policy.train()
    order = np.random.permutation(len(actions))
    total_loss = 0.0
    correct = 0
    seen = 0

    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        obs_t = torch.as_tensor(states[idx], device=device)
        action_t = torch.as_tensor(actions[idx], device=device)
        mask_t = torch.as_tensor(masks[idx], device=device)

        try:
            _, log_prob, _ = model.policy.evaluate_actions(
                obs_t,
                action_t,
                action_masks=mask_t,
            )
            distribution = model.policy.get_distribution(obs_t, action_masks=mask_t)
        except TypeError:
            distribution = model.policy.get_distribution(obs_t, action_masks=mask_t)
            log_prob = distribution.log_prob(action_t)

        loss = -log_prob.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            predicted = distribution.distribution.probs.argmax(dim=1)
            batch_size_seen = int(action_t.numel())
            correct += int((predicted == action_t).sum().item())
            seen += batch_size_seen
            total_loss += float(loss.item()) * batch_size_seen

    return total_loss / max(seen, 1), correct / max(seen, 1)


def oracle_warmstart(
    model: MaskablePPO,
    config: MaskablePPOConfig,
    timesteps: int,
    best_win_rate: float,
    writer: SummaryWriter | None = None,
) -> float:
    if not RUN_ORACLE_WARMSTART or best_win_rate >= ORACLE_WARMSTART_MAX_BEST_WIN_RATE:
        return best_win_rate

    latest_path = latest_model_path(config.level)
    best_path = best_model_path(config.level)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=ORACLE_WARMSTART_LEARNING_RATE)

    print(
        "oracle warm-start: "
        f"epochs={ORACLE_WARMSTART_EPOCHS} "
        f"games/epoch={ORACLE_WARMSTART_GAMES_PER_EPOCH} "
        f"lr={ORACLE_WARMSTART_LEARNING_RATE:.1e}"
    )

    for epoch in range(1, ORACLE_WARMSTART_EPOCHS + 1):
        states, masks, actions = collect_oracle_examples(config, epoch)
        loss, accuracy = imitation_epoch(
            model,
            optimizer,
            states,
            masks,
            actions,
            ORACLE_WARMSTART_BATCH_SIZE,
        )

        eval_text = ""
        if epoch % ORACLE_WARMSTART_EVAL_EVERY == 0 or epoch == ORACLE_WARMSTART_EPOCHS:
            eval_stats = evaluate_model(
                model,
                config,
                ORACLE_WARMSTART_EVAL_GAMES,
                deterministic=True,
            )
            if eval_stats["win_rate"] > best_win_rate:
                best_win_rate = eval_stats["win_rate"]
                model.save(best_path)
            eval_text = (
                f" | eval_win={eval_stats['win_rate'] * 100.0:.2f}%"
                f" best={best_win_rate * 100.0:.2f}%"
                f" eval_moves={eval_stats['avg_moves']:.2f}"
            )

        model.save(latest_path)
        save_metadata(config, timesteps, best_win_rate)
        print(
            f"oracle epoch {epoch:04d} | examples={len(actions):6d} "
            f"loss={loss:.4f} oracle_acc={accuracy * 100.0:.2f}%{eval_text}"
        )
        if writer is not None:
            writer.add_scalar("oracle/loss", loss, epoch)
            writer.add_scalar("oracle/accuracy_pct", accuracy * 100.0, epoch)
            writer.add_scalar("oracle/examples", len(actions), epoch)
            if eval_text:
                write_eval_scalars(writer, "oracle_eval", eval_stats, epoch)
                writer.add_scalar("oracle_eval/best_win_rate_pct", best_win_rate * 100.0, epoch)
            writer.flush()

    return best_win_rate


def main(config: MaskablePPOConfig | None = None) -> None:
    if config is None:
        config = build_config()

    probe_env = make_env(config)
    train_env = make_train_env(config)
    print_training_setup(config, probe_env)
    model, timesteps, best_win_rate = load_or_create_maskable_ppo(config, train_env)
    writer = SummaryWriter(log_dir=str(ppo_tensorboard_dir(config.level) / "eval"))
    writer.add_text("config", pprint.pformat(asdict(config)), 0)
    best_win_rate = oracle_warmstart(model, config, timesteps, best_win_rate, writer)

    latest_path = latest_model_path(config.level)
    best_path = best_model_path(config.level)
    target = config.target_win_rate_pct / 100.0
    weak_eval_streak = 0
    rollback_drop = config.rollback_drop_pct / 100.0

    print(f"Starting training with config:\n{pprint.pformat(asdict(config))}")

    try:
        while best_win_rate < target:
            if config.max_timesteps is not None and timesteps >= config.max_timesteps:
                print(f"stopped at max_timesteps={config.max_timesteps}")
                break

            chunk = config.eval_every_timesteps
            if config.max_timesteps is not None:
                chunk = min(chunk, config.max_timesteps - timesteps)
                if chunk <= 0:
                    break

            before_steps = int(model.num_timesteps)
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                progress_bar=True,
                tb_log_name="ppo",
            )
            timesteps += int(model.num_timesteps) - before_steps
            model.save(latest_path)

            greedy_eval = evaluate_model(model, config, config.eval_games, deterministic=True)
            sample_eval = evaluate_model(model, config, max(200, config.eval_games // 5), deterministic=False)

            improved = greedy_eval["win_rate"] > best_win_rate
            if improved:
                best_win_rate = greedy_eval["win_rate"]
                model.save(best_path)
                weak_eval_streak = 0
            elif (
                best_path.exists()
                and best_win_rate > 0.0
                and greedy_eval["win_rate"] <= best_win_rate - rollback_drop
            ):
                weak_eval_streak += 1
            else:
                weak_eval_streak = 0

            rolled_back = False
            if config.rollback_patience > 0 and weak_eval_streak >= config.rollback_patience:
                print(
                    "rollback: current greedy win "
                    f"{greedy_eval['win_rate'] * 100.0:.2f}% is more than "
                    f"{config.rollback_drop_pct:.1f}% below best "
                    f"{best_win_rate * 100.0:.2f}%; reloading {best_path}"
                )
                model = rebuild_model_from_checkpoint(best_path, config, train_env, timesteps)
                model.save(latest_path)
                weak_eval_streak = 0
                rolled_back = True

            save_metadata(config, timesteps, best_win_rate)
            write_eval_scalars(writer, "eval/greedy", greedy_eval, timesteps)
            write_eval_scalars(writer, "eval/sample", sample_eval, timesteps)
            writer.add_scalar("eval/best_win_rate_pct", best_win_rate * 100.0, timesteps)
            writer.add_scalar("eval/weak_eval_streak", weak_eval_streak, timesteps)
            writer.add_scalar("train/timesteps", timesteps, timesteps)
            writer.flush()
            print(
                f"steps={timesteps:09d} | "
                f"greedy_win={greedy_eval['win_rate'] * 100.0:.2f}% "
                f"greedy_loss={greedy_eval['loss_rate'] * 100.0:.2f}% "
                f"sample_win={sample_eval['win_rate'] * 100.0:.2f}% "
                f"sample_loss={sample_eval['loss_rate'] * 100.0:.2f}% "
                f"best={best_win_rate * 100.0:.2f}% "
                f"avg_moves={greedy_eval['avg_moves']:.2f} "
                f"won_moves={greedy_eval['avg_won_moves']:.2f} "
                f"lost_moves={greedy_eval['avg_lost_moves']:.2f} "
                f"return={greedy_eval['avg_return']:.3f}"
                f"{' rollback=best' if rolled_back else ''}"
            )

    except KeyboardInterrupt:
        print("\nCtrl+C received. Saving latest MaskablePPO checkpoint before exit...")
    finally:
        model.save(latest_path)
        save_metadata(config, timesteps, best_win_rate)
        print(f"saved latest checkpoint: {latest_path}")
        if best_path.exists():
            print(f"best checkpoint: {best_path} ({best_win_rate * 100.0:.2f}%)")
        writer.flush()
        writer.close()
        train_env.close()


if __name__ == "__main__":
    main()
