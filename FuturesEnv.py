#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
from tf_agents.environments import py_environment;
from tf_agents.environments import tf_environment;
from tf_agents.environments import tf_py_environment;
from tf_agents.environments import utils;
from tf_agents.specs import array_spec;
from tf_agents.environments import wrappers;
from tf_agents.environments import suite_gym;
from tf_agents.trajectories import time_step as ts;

class FuturesEnv(py_environment.PyEnvironment):

    def __init__(self, capital = 10000.0, dataset = None):
        assert type(capital) is float and capital > 0;
        self._action_spec = array_spec.BoundedArraySpec(
            (4,),
            dtype = np.int32,
            # transaction: sell(0), buy(1), none(2)
            # lever: n * 0.01
            # stop-profit price: when buy: price + n * 0.01; when sell: price - n * 0.01
            # stop-loss price: when buy: price - n * 0.01; when sell: price + n * 0.01
            minimum = [0, 1, 1, 1],
            maximum = [2, 100, np.inf, np.inf],
            name = 'action');
        self._observation_spec = array_spec.BoundedArraySpec(
            (2,),
            dtype = np.int32,
            # buying price: n * 0.01
            # selling price: n * 0.01 
            minimum = [0, 0],
            maximum = [np.inf, np.inf],
            name = 'observation');
        self._positions = list();
        self._profit = 0;
        self._episode_ended = False;
        # customized member
        self.capital = capital;
        self.dataset = dataset;
        self.index = 0;
    def action_spec(self):
        return self._action_spec;
    def observation_spec(self):
        return self._observation_spec;
    def _reset(self):
        self._positions = list();
        self._profit = 0;
        self._episode_ended = False;
        # customized member
        self.index = 0;
        return ts.restart((self._positions,self._profit));
    def _step(self, action):
        # 1) reset condition
        if self._episode_ended:
            return self.reset();
        # 2) state transition
        sell_price = self.dataset[self.index, 0];
        buy_price = self.dataset[self.index, 1];
        # add position into state
        if action[0] != 2:
            # sell or buy
            assert sell_price <= buy_price;
            if action[0] == 0:
                # when sell, stop profit/loss price for buy price
                stop_profit_price = buy_price - 0.01 * action[2];
                stop_loss_price = buy_price + 0.01 * action[3];
                # (sell, lever scale, sell price, stop_profit_price, stop_loss_price)
                self._state.append((action[0], action[1], sell_price, stop_profit_price, stop_loss_price));
            else:
                # when buy, stop profit/loss price for sell price
                stop_profit_price = sell_price + 0.01 * action[2];
                stop_loss_price = sell_price - 0.01 * action[3];
                # (buy, lever scale, buy price, stop_profit_price, stop_loss_price)
                self._state.append((action[0], action[1], buy_price, stop_profit_price, stop_loss_price));
        # check whether a position need to be closed out
        left_positions = list();
        unsettled_profit = 0;
        for position in self._positions:
            stop_profit_price = position[3];
            stop_loss_price = position[4];
            if position[0] == 0:
                # sell
                prev_sell_price = position[2];
                if buy_price <= stop_profit_price or buy_price >= stop_loss_price:
                    # settled profit by short selling
                    self._profit += position[1] * (prev_sell_price - buy_price);
                    close_out = True;
                else:
                    # unsettled profit by short selling
                    unsettled_profit += position[1] * (prev_sell_price - buy_price);
                    close_out = False;
            else position[0] == 1:
                # buy
                prev_buy_price = position[2];
                if sell_price >= stop_profit_price or sell_price <= stop_loss_price:
                    # settled profit by going long
                    self._profit += position[1] * (sell_price - prev_buy_price);
                    close_out = True;
                else:
                    # unsettled profit by going long
                    unsettled_profit += position[1] * (sell_price - prev_buy_price);
                    close_out = False;
            if close_out == False:
                left_positions.append(position);
        self._positions = left_positions;
        # 3) end episode conditions
        if self.index == len(self.dataset):
            # end of dataset
            self._episode_ended = True;
        if self.capital + self._profit + unsettled_profit <= 0:
            # mandatory liquidation
            self._episode_ended = True;
        # 4) update dataset iterator
        self.index += 1;
        # 5) return
        if self._episode_ended:
            return ts.termination((self._positions,self._profit), self._profit + unsettled_profit);
        else:
            return ts.transition((self._positions,self._profit), reward = self._profit + unsettled_profit, discount = 0.3);

