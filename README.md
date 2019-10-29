# FuturesEnvironment
This project implements an tf-agent environment of futures market

## purpose
the environment is a challenge for reinforcement learning. the environment is developed according to gym environment interface.

## prerequisite packages
install prerequisite package with

```bash
pip3 install -U gym tf-agents-nightly
```

## environment specification
the observation is a vector of length 2 which represents the sell price and buy price by minites respectively.

the action is a vector of length 3 which represents the lever, stop-profit price, stop-loss price respectively.

the reward is accumulated profit till current timestamp.

