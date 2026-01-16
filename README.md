# Mini RL

Minimal single file implementations of various reinforcement learning algorithms

## Run

Installation. Assume that `uv` is available:

```bash
uv sync
```

Execution (e.g., REINFORCE):

```bash
uv run reinforce.py
```

Check the result with TensorBoard:

```bash
uv run tensorboard --logdir runs
```

and go to `http://localhost:6006` with your browser.

## Algorithms

- REINFORCE (Cart Pole)

## Todo

- A2C
- DQN
- PPO
- DDPG
- TD3
- SAC
- BC
- CQL
- IQL
- DT
- Diffusion policy
- Diffusion-QL
- Diffuser