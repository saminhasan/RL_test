from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from torch.utils.tensorboard import SummaryWriter

from engine import MinesweeperEngine
from minesweeper_env import encode_board, level_name, level_to_id, make_env, make_train_env
from models import (
    MaskablePPOConfig,
    best_model_path,
    latest_model_path,
    load_or_create_maskable_ppo,
    ppo_tensorboard_dir,
    print_model_architecture,
    save_metadata,
)


BAYES_WIN_TARGETS = {
    # "test": 0.567,
    # "easy": 0.846,
    "test": 0.6,
    "easy": 0.9,
    "medium": 0.693,
    "hard": 0.331,
}


@dataclass(frozen=True)
class TeacherLevelConfig:
    games_per_epoch: int
    batch_size: int
    learning_rate: float
    eval_every_epochs: int
    eval_games: int
    max_epochs: int
    policy_temperature: float = 0.03
    risk_loss_weight: float = 1.0
    policy_kl_weight: float = 0.75
    policy_margin_weight: float = 0.5
    policy_margin: float = 1.0
    margin_target_gap: float = 0.01
    value_loss_weight: float = 0.05


TEACHER_LEVEL_CONFIGS = {
    "test": TeacherLevelConfig(
        games_per_epoch=500,
        batch_size=512,
        learning_rate=1.0e-3,
        eval_every_epochs=2,
        eval_games=1_000,
        max_epochs=300,
    ),
    "easy": TeacherLevelConfig(
        games_per_epoch=500,
        batch_size=512,
        learning_rate=1.0e-3,
        eval_every_epochs=2,
        eval_games=1_000,
        max_epochs=500,
    ),
    "medium": TeacherLevelConfig(
        games_per_epoch=1000,
        batch_size=1024,
        learning_rate=3.0e-4,
        eval_every_epochs=1,
        eval_games=4096,
        max_epochs=1000,
        policy_temperature=0.025,
        risk_loss_weight=1.0,
        policy_kl_weight=1.0,
        policy_margin_weight=0.75,
        policy_margin=1.25,
        margin_target_gap=0.01,
        value_loss_weight=0.05,
    ),
    "hard": TeacherLevelConfig(
        games_per_epoch=150,
        batch_size=1_024,
        learning_rate=5.0e-4,
        eval_every_epochs=2,
        eval_games=1_000,
        max_epochs=900,
        policy_temperature=0.03,
        risk_loss_weight=1.0,
        policy_kl_weight=0.75,
        policy_margin_weight=0.5,
        policy_margin=1.0,
        margin_target_gap=0.01,
        value_loss_weight=0.05,
    ),
}


BASE_MODEL_CONFIG = MaskablePPOConfig(
    level="medium",
    seed=42,
    n_envs=4,
    use_subproc_env=True,
    load_best_checkpoint=True,
    device="auto",
)

ASK_FOR_LEVEL = True


def pick_level(default_level: int | str) -> str:
    choices = ", ".join(f"{idx}:{name}" for idx, name in MinesweeperEngine.LEVELS.items())
    print(f"available levels: {choices}")
    answer = input(f"select level [{level_name(default_level)}]: ").strip()
    if not answer:
        return level_name(default_level)
    return level_name(level_to_id(answer))


def build_model_config() -> MaskablePPOConfig:
    level = pick_level(BASE_MODEL_CONFIG.level) if ASK_FOR_LEVEL else level_name(BASE_MODEL_CONFIG.level)
    return MaskablePPOConfig(
        level=level,
        seed=BASE_MODEL_CONFIG.seed,
        target_win_rate_pct=BAYES_WIN_TARGETS[level] * 100.0,
        learning_rate=BASE_MODEL_CONFIG.learning_rate,
        n_steps=BASE_MODEL_CONFIG.n_steps,
        batch_size=BASE_MODEL_CONFIG.batch_size,
        n_epochs=BASE_MODEL_CONFIG.n_epochs,
        gamma=BASE_MODEL_CONFIG.gamma,
        gae_lambda=BASE_MODEL_CONFIG.gae_lambda,
        clip_range=BASE_MODEL_CONFIG.clip_range,
        ent_coef=BASE_MODEL_CONFIG.ent_coef,
        vf_coef=BASE_MODEL_CONFIG.vf_coef,
        max_grad_norm=BASE_MODEL_CONFIG.max_grad_norm,
        n_envs=1,
        use_subproc_env=False,
        freeze_features_extractor=False,
        load_best_checkpoint=BASE_MODEL_CONFIG.load_best_checkpoint,
        device=BASE_MODEL_CONFIG.device,
    )


def bayes_dense_targets(
    engine: MinesweeperEngine,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    hidden_mask = (~engine.revealed).reshape(-1)
    hidden_actions = np.flatnonzero(hidden_mask)
    action_count = engine.rows * engine.cols

    risk_target = np.zeros(action_count, dtype=np.float32)
    policy_target = np.zeros(action_count, dtype=np.float32)
    best_candidates = np.array([0], dtype=np.int64)
    value_target = 0.0

    if hidden_actions.size == 0:
        return risk_target, policy_target, best_candidates, value_target

    probabilities = engine.get_mine_probabilities().reshape(-1)
    hidden_risk = probabilities[hidden_actions].astype(np.float32)
    risk_target[hidden_actions] = hidden_risk

    min_risk = float(hidden_risk.min())
    value_target = 1.0 - min_risk
    best_candidates = hidden_actions[hidden_risk == min_risk].astype(np.int64)

    logits = -hidden_risk / max(temperature, 1e-6)
    logits -= float(logits.max())
    weights = np.exp(logits).astype(np.float32)
    weights /= float(weights.sum())
    policy_target[hidden_actions] = weights
    return risk_target, policy_target, best_candidates, value_target


def oracle_play_action(engine: MinesweeperEngine, candidates: np.ndarray, rng: np.random.Generator) -> int:
    if not engine.started:
        return (engine.rows // 2) * engine.cols + (engine.cols // 2)

    safe_candidates = candidates[~engine._mines.reshape(-1)[candidates]]
    if safe_candidates.size:
        return int(safe_candidates[int(rng.integers(safe_candidates.size))])

    hidden_safe = np.flatnonzero(((~engine.revealed) & ~engine._mines).reshape(-1))
    if len(hidden_safe) > 0:
        return int(hidden_safe[int(rng.integers(len(hidden_safe)))])

    hidden = np.flatnonzero((~engine.revealed).reshape(-1))
    if len(hidden) > 0:
        return int(hidden[int(rng.integers(len(hidden)))])
    return 0


def collect_teacher_examples(
    model_config: MaskablePPOConfig,
    teacher_config: TeacherLevelConfig,
    epoch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(model_config.seed + epoch * 1_000_003)
    level_id = level_to_id(model_config.level)

    states: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    risk_targets: list[np.ndarray] = []
    policy_targets: list[np.ndarray] = []
    value_targets: list[float] = []

    for game in range(teacher_config.games_per_epoch):
        seed = model_config.seed + epoch * 1_000_003 + game
        engine = MinesweeperEngine(level=level_id, seed=seed, headless=True)

        while engine.state != MinesweeperEngine.OVER:
            risk_target, policy_target, candidates, value_target = bayes_dense_targets(
                engine,
                teacher_config.policy_temperature,
            )

            states.append(encode_board(engine.get_public_view(reveal_mines_on_loss=False)))
            masks.append((~engine.revealed).reshape(-1).astype(bool))
            risk_targets.append(risk_target)
            policy_targets.append(policy_target)
            value_targets.append(value_target)

            action = oracle_play_action(engine, candidates, rng)
            row, col = divmod(action, engine.cols)
            engine.reveal_count(row, col)

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(masks, dtype=bool),
        np.asarray(risk_targets, dtype=np.float32),
        np.asarray(policy_targets, dtype=np.float32),
        np.asarray(value_targets, dtype=np.float32),
    )


def teacher_epoch(
    model: MaskablePPO,
    optimizer: torch.optim.Optimizer,
    states: np.ndarray,
    masks: np.ndarray,
    risk_targets: np.ndarray,
    policy_targets: np.ndarray,
    value_targets: np.ndarray,
    batch_size: int,
    teacher_config: TeacherLevelConfig,
) -> dict[str, float]:
    device = model.policy.device
    model.policy.train()
    order = np.random.permutation(len(states))
    total_loss = 0.0
    total_risk_loss = 0.0
    total_policy_kl = 0.0
    total_policy_margin = 0.0
    total_value_loss = 0.0
    total_policy_entropy = 0.0
    total_best_bayes_risk = 0.0
    total_chosen_bayes_risk = 0.0
    correct = 0
    seen = 0

    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        obs_t = torch.as_tensor(states[idx], device=device)
        mask_t = torch.as_tensor(masks[idx], device=device)
        risk_t = torch.as_tensor(risk_targets[idx], device=device)
        policy_target_t = torch.as_tensor(policy_targets[idx], device=device)
        value_target_t = torch.as_tensor(value_targets[idx], device=device)

        policy_logits, values, risk_logits = model.policy.forward_heads(obs_t)
        mask_float = mask_t.float()
        risk_loss_map = F.binary_cross_entropy_with_logits(risk_logits, risk_t, reduction="none")
        risk_loss = (risk_loss_map * mask_float).sum() / mask_float.sum().clamp_min(1.0)

        masked_logits = policy_logits.masked_fill(~mask_t, -1.0e9)
        log_probs = F.log_softmax(masked_logits, dim=1)
        target_log = policy_target_t.clamp_min(1e-8).log()
        policy_kl = (policy_target_t * (target_log - log_probs)).sum(dim=1).mean()

        visible_risk = risk_t.masked_fill(~mask_t, float("inf"))
        best_risk = visible_risk.min(dim=1).values
        best_mask = mask_t & torch.isclose(risk_t, best_risk[:, None], atol=1e-6, rtol=0.0)
        ranked_mask = mask_t & (risk_t > best_risk[:, None] + teacher_config.margin_target_gap)
        best_logit = masked_logits.masked_fill(~best_mask, -1.0e9).max(dim=1).values
        policy_margin_terms = F.relu(
            teacher_config.policy_margin - best_logit[:, None] + masked_logits
        )
        policy_margin = (
            policy_margin_terms[ranked_mask].mean()
            if ranked_mask.any()
            else policy_margin_terms.new_tensor(0.0)
        )

        value_loss = F.mse_loss(values.squeeze(1), value_target_t)
        loss = (
            teacher_config.risk_loss_weight * risk_loss
            + teacher_config.policy_kl_weight * policy_kl
            + teacher_config.policy_margin_weight * policy_margin
            + teacher_config.value_loss_weight * value_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            predicted = masked_logits.argmax(dim=1)
            predicted_risk = risk_t.gather(1, predicted[:, None]).squeeze(1)
            hit = torch.isclose(predicted_risk, best_risk, atol=1e-6, rtol=0.0)
            probs = torch.softmax(masked_logits, dim=1)
            policy_entropy = -(probs * log_probs).sum(dim=1).mean()
            batch_seen = int(len(idx))
            correct += int(hit.sum().item())
            seen += batch_seen
            total_loss += float(loss.item()) * batch_seen
            total_risk_loss += float(risk_loss.item()) * batch_seen
            total_policy_kl += float(policy_kl.item()) * batch_seen
            total_policy_margin += float(policy_margin.item()) * batch_seen
            total_value_loss += float(value_loss.item()) * batch_seen
            total_policy_entropy += float(policy_entropy.item()) * batch_seen
            total_best_bayes_risk += float(best_risk.mean().item()) * batch_seen
            total_chosen_bayes_risk += float(predicted_risk.mean().item()) * batch_seen

    denom = max(seen, 1)
    return {
        "loss": total_loss / denom,
        "risk_loss": total_risk_loss / denom,
        "policy_kl": total_policy_kl / denom,
        "policy_margin": total_policy_margin / denom,
        "value_loss": total_value_loss / denom,
        "policy_entropy": total_policy_entropy / denom,
        "best_bayes_risk": total_best_bayes_risk / denom,
        "chosen_bayes_risk": total_chosen_bayes_risk / denom,
        "best_action_acc": correct / denom,
    }


def evaluate_model(
    model: MaskablePPO,
    model_config: MaskablePPOConfig,
    games: int,
    deterministic: bool = True,
) -> dict[str, float]:
    wins = 0
    losses = 0
    moves = 0
    returns = 0.0

    for game in range(games):
        env = make_env(model_config, seed_offset=50_000_000 + game)
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

        wins += int(bool(info["won"]))
        losses += int(not bool(info["won"]))
        moves += int(info["moves"])
        returns += episode_return

    return {
        "win_rate": wins / games if games else 0.0,
        "loss_rate": losses / games if games else 0.0,
        "avg_moves": moves / games if games else 0.0,
        "avg_return": returns / games if games else 0.0,
    }


def count_parameters(model: MaskablePPO) -> tuple[int, int]:
    total = sum(param.numel() for param in model.policy.parameters())
    trainable = sum(param.numel() for param in model.policy.parameters() if param.requires_grad)
    return total, trainable


def print_teacher_setup(
    model: MaskablePPO,
    model_config: MaskablePPOConfig,
    teacher_config: TeacherLevelConfig,
    target_win_rate: float,
    best_win_rate: float,
) -> None:
    total_params, trainable_params = count_parameters(model)
    print_model_architecture(model_config.level)
    print("oracle teacher training:")
    print(f"  target bayes       : {target_win_rate * 100.0:.2f}%")
    print(f"  start best         : {best_win_rate * 100.0:.2f}%")
    print(f"  games/epoch        : {teacher_config.games_per_epoch}")
    print(f"  batch size         : {teacher_config.batch_size}")
    print(f"  learning rate      : {teacher_config.learning_rate:.1e}")
    print(f"  eval every epochs  : {teacher_config.eval_every_epochs}")
    print(f"  eval games         : {teacher_config.eval_games}")
    print(f"  max epochs         : {teacher_config.max_epochs}")
    print(f"  policy temperature : {teacher_config.policy_temperature:.3f}")
    print(f"  risk loss weight   : {teacher_config.risk_loss_weight:.2f}")
    print(f"  policy KL weight   : {teacher_config.policy_kl_weight:.2f}")
    print(f"  margin loss weight : {teacher_config.policy_margin_weight:.2f}")
    print(f"  policy margin      : {teacher_config.policy_margin:.2f}")
    print(f"  margin target gap  : {teacher_config.margin_target_gap:.3f}")
    print(f"  value loss weight  : {teacher_config.value_loss_weight:.2f}")
    print(f"  device             : {model.policy.device}")
    print(f"  policy params      : {total_params:,}")
    print(f"  trainable params   : {trainable_params:,}")
    print(f"  model dir          : {latest_model_path(model_config.level).parent}")
    print(f"  tensorboard        : {ppo_tensorboard_dir(model_config.level) / 'teacher'}")
    print(model.policy)


def write_teacher_scalars(
    writer: SummaryWriter,
    stats: dict[str, float],
    epoch: int,
) -> None:
    for key, value in stats.items():
        writer.add_scalar(f"teacher/{key}", value, epoch)
    writer.add_scalar("teacher/best_action_acc_pct", stats["best_action_acc"] * 100.0, epoch)


def main(model_config: MaskablePPOConfig | None = None) -> None:
    if model_config is None:
        model_config = build_model_config()

    teacher_config = TEACHER_LEVEL_CONFIGS[model_config.level_name]
    target_win_rate = BAYES_WIN_TARGETS[model_config.level_name]
    env = make_train_env(model_config)
    model, timesteps, best_win_rate = load_or_create_maskable_ppo(model_config, env)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=teacher_config.learning_rate)

    latest_path = latest_model_path(model_config.level)
    best_path = best_model_path(model_config.level)
    writer = SummaryWriter(log_dir=str(ppo_tensorboard_dir(model_config.level) / "teacher"))
    writer.add_text("model_config", str(model_config), 0)
    writer.add_text("teacher_config", str(teacher_config), 0)

    print_teacher_setup(model, model_config, teacher_config, target_win_rate, best_win_rate)

    epoch = 0
    try:
        while best_win_rate < target_win_rate and epoch < teacher_config.max_epochs:
            epoch += 1
            states, masks, risk_targets, policy_targets, value_targets = collect_teacher_examples(
                model_config,
                teacher_config,
                epoch,
            )
            stats = teacher_epoch(
                model,
                optimizer,
                states,
                masks,
                risk_targets,
                policy_targets,
                value_targets,
                teacher_config.batch_size,
                teacher_config,
            )
            write_teacher_scalars(writer, stats, epoch)

            eval_text = ""
            if epoch % teacher_config.eval_every_epochs == 0 or epoch == 1:
                greedy_eval = evaluate_model(
                    model,
                    model_config,
                    teacher_config.eval_games,
                    deterministic=True,
                )
                sample_eval = evaluate_model(
                    model,
                    model_config,
                    max(200, teacher_config.eval_games // 5),
                    deterministic=False,
                )
                if greedy_eval["win_rate"] > best_win_rate:
                    best_win_rate = greedy_eval["win_rate"]
                    model.save(best_path)

                save_metadata(model_config, timesteps=timesteps, best_win_rate=best_win_rate)
                target_progress = greedy_eval["win_rate"] / target_win_rate if target_win_rate > 0 else 0.0
                target_gap = max(target_win_rate - greedy_eval["win_rate"], 0.0)
                eval_text = (
                    f" | greedy_win={greedy_eval['win_rate'] * 100.0:.2f}%"
                    f" sample_win={sample_eval['win_rate'] * 100.0:.2f}%"
                    f" best={best_win_rate * 100.0:.2f}%"
                    f" target={target_win_rate * 100.0:.2f}%"
                    f" progress={target_progress * 100.0:.1f}%"
                    f" gap={target_gap * 100.0:.2f}%"
                    f" moves={greedy_eval['avg_moves']:.2f}"
                    f" return={greedy_eval['avg_return']:.3f}"
                )
                writer.add_scalar("eval/greedy_win_rate_pct", greedy_eval["win_rate"] * 100.0, epoch)
                writer.add_scalar("eval/greedy_loss_rate_pct", greedy_eval["loss_rate"] * 100.0, epoch)
                writer.add_scalar("eval/greedy_avg_moves", greedy_eval["avg_moves"], epoch)
                writer.add_scalar("eval/greedy_avg_return", greedy_eval["avg_return"], epoch)
                writer.add_scalar("eval/sample_win_rate_pct", sample_eval["win_rate"] * 100.0, epoch)
                writer.add_scalar("eval/sample_loss_rate_pct", sample_eval["loss_rate"] * 100.0, epoch)
                writer.add_scalar("eval/sample_avg_moves", sample_eval["avg_moves"], epoch)
                writer.add_scalar("eval/sample_avg_return", sample_eval["avg_return"], epoch)
                writer.add_scalar("eval/best_win_rate_pct", best_win_rate * 100.0, epoch)
                writer.add_scalar("eval/target_win_rate_pct", target_win_rate * 100.0, epoch)
                writer.add_scalar("eval/target_progress_pct", target_progress * 100.0, epoch)
                writer.add_scalar("eval/target_gap_pct", target_gap * 100.0, epoch)
            writer.flush()

            model.save(latest_path)
            print(
                f"epoch {epoch:05d} | examples={len(states):7d} "
                f"loss={stats['loss']:.4f} "
                f"risk_bce={stats['risk_loss']:.4f} "
                f"kl={stats['policy_kl']:.4f} "
                f"margin={stats['policy_margin']:.4f} "
                f"value={stats['value_loss']:.4f} "
                f"entropy={stats['policy_entropy']:.3f} "
                f"chosen_risk={stats['chosen_bayes_risk']:.3f} "
                f"best_risk={stats['best_bayes_risk']:.3f} "
                f"best_action_acc={stats['best_action_acc'] * 100.0:.2f}%{eval_text}"
            )

    except KeyboardInterrupt:
        print("\nCtrl+C received. Saving oracle-teacher checkpoint before exit...")
    finally:
        model.save(latest_path)
        save_metadata(model_config, timesteps=timesteps, best_win_rate=best_win_rate)
        print(f"saved latest checkpoint: {latest_path}")
        if best_path.exists():
            print(f"best checkpoint: {best_path} ({best_win_rate * 100.0:.2f}%)")
        if best_win_rate >= target_win_rate:
            print(f"bayes-level target reached: {best_win_rate * 100.0:.2f}%")
        writer.flush()
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
