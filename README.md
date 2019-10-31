# FuturesEnvironment
This project implements an tf-agent environment of futures market

## purpose
the environment is a challenge for reinforcement learning. the environment is developed according to gym environment interface.

## prerequisite packages
install prerequisite package with

```bash
pip3 install -U gym tf-agents-nightly
```

you also need to install MetaTrader 5 to execute the dataset download script. fortunately, MetaTrader 5 can run on wine perfectly.

## download the dataset

edit the download_dataset.mq5 file with MetaEditor. change the symbol to whatever you want to deal with. execute the script to download the dataset in csv format into dataset directory under MQL5 director of MetaTrader 5.

## convert the dataset format

the MQL5 downloaded file was encoded in UTF16. you need to convert it into utf8 and parse it into simpler format. the project provides a converter which can do the tricks. compile it with the following command.

```bash
make -C cc
```

run the converter with command

```bash
./cc/convert -i <path/to/dataset/file> -o <path/to/output>
```

## environment specification
the observation is a vector of length 2 which represents the sell price and buy price by minites respectively.

the action is a vector of length 3 which represents the lever, stop-profit price, stop-loss price respectively.

the reward is accumulated profit till current timestamp.

