"""Reinforcement learning for Minesweeper: Double DQN (PyTorch), from the visible board only.

minesweeper_env.py  gymnasium environment and the reward function (RewardConfig)
qnet.py             convolutional Q-network, one value per cell; checkpoints
dqn.py              Double DQN training, adapted from CleanRL's dqn.py
evaluate.py         measuring only: the trained net next to bayes on the same boards

    python -m rl.dqn --total-timesteps 300000
    python -m rl.evaluate --model models/<run>-best.pt --level 4 --breakdown
"""
