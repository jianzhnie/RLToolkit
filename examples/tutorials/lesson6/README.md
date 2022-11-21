# minimal-marl

Minimal implementation of multi-agent reinforcement learning algorithms(marl). This repo
complements [`ma-gym`](https://github.com/koulanurag/ma-gym) and is is inspired
by [`minimalRl`](https://github.com/seungeunrho/minimalRL) which provides minimal implementation for RL algorithms for
the ease of understanding.

## Installation

```bash
  pip install ma-gym>=0.0.7 torch>=1.8 wandb
```

## Usage

```bash
python train.py # such as `vdn.py`
```

## Algorithms

- [ ] IDQN \[DQN version of [IQL](https://web.media.mit.edu/~cynthiab/Readings/tan-MAS-reinfLearn.pdf)\]
- [ ] [VDN](https://arxiv.org/abs/1706.05296) (Value Decomposition Network)
- [ ] [QMIX](https://arxiv.org/pdf/1803.11485.pdf)
- [ ] [MADDPG](https://arxiv.org/abs/1706.02275) (Multi Agent Deep Deterministic Policy Gradient)
  - `Status: Not converging at the moment`
