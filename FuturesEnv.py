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
            (3,),
            dtype = np.int32,
            # transaction: sell(0), buy(1), none(2)
            # lever: n * 0.01; sell(n>0), buy(n<0), none(n=0)
            # stop-profit price: when buy: price + n * 0.01; when sell: price - n * 0.01
            # stop-loss price: when buy: price - n * 0.01; when sell: price + n * 0.01
            minimum = [-100, 1, 1],
            maximum = [100, np.inf, np.inf],
            name = 'action');
        self._observation_spec = array_spec.BoundedArraySpec(
            (2,),
            dtype = np.float32,
            # sell price: n
            # buy price: n 
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
        return ts.restart((self.dataset[self.index] ,self._profit));

    def _step(self, action):

        # 1) reset condition
        if self._episode_ended:
            return self.reset();
        # 2) state transition
        sell_price = self.dataset[self.index, 0];
        buy_price = self.dataset[self.index, 1];
        # add position into state
        if action[0] != 0:
            # sell or buy
            assert sell_price <= buy_price;
            if action[0] > 0:
                # when sell, stop profit/loss price for buy price
                stop_profit_price = buy_price - 0.01 * action[1];
                stop_loss_price = buy_price + 0.01 * action[2];
                # (sell, lever scale, sell price, stop_profit_price, stop_loss_price)
                self._state.append({'lever': action[0], 'position': sell_price, 'stop profit price': stop_profit_price, 'stop loss price': stop_loss_price});
            else:
                # when buy, stop profit/loss price for sell price
                stop_profit_price = sell_price + 0.01 * action[1];
                stop_loss_price = sell_price - 0.01 * action[2];
                # (buy, lever scale, buy price, stop_profit_price, stop_loss_price)
                self._state.append({'lever': action[0], 'position': buy_price, 'stop profit price': stop_profit_price, 'stop loss price': stop_loss_price});
        # check whether a position need to be closed out
        left_positions = list();
        unsettled_profit = 0;
        for position in self._positions:
            stop_profit_price = position['stop profit price'];
            stop_loss_price = position['stop loss price'];
            if position['lever'] > 0:
                # sell
                prev_sell_price = position['position'];
                if buy_price <= stop_profit_price or buy_price >= stop_loss_price:
                    # settled profit by short selling
                    self._profit += abs(position['lever']) * (prev_sell_price - buy_price);
                    close_out = True;
                else:
                    # unsettled profit by short selling
                    unsettled_profit += abs(position['lever']) * (prev_sell_price - buy_price);
                    close_out = False;
            elif position['lever'] < 0:
                # buy
                prev_buy_price = position['position'];
                if sell_price >= stop_profit_price or sell_price <= stop_loss_price:
                    # settled profit by going long
                    self._profit += abs(position['lever']) * (sell_price - prev_buy_price);
                    close_out = True;
                else:
                    # unsettled profit by going long
                    unsettled_profit += abs(position['lever']) * (sell_price - prev_buy_price);
                    close_out = False;
            else:
                raise "invalid action!";
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
            return ts.termination(self.dataset[self.index], self._profit + unsettled_profit);
        else:
            return ts.transition(self.dataset[self.index], reward = self._profit + unsettled_profit, discount = 0.3);

if __name__ == "__main__":

    assert True == tf.executing_eagerly();
    env = tf_py_environment.TFPyEnvironment(FuturesEnv(dataset = np.random.normal(size = (100,2))));

