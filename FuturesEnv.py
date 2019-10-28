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
        self._state = capital;
        self._episode_ended = False;
        # customized member
        self.capital = capital;
        self.dataset = dataset;
        self.index = 0;
        self.positions = list();
    def action_spec(self):
        return self._action_spec;
    def observation_spec(self):
        return self._observation_spec;
    def _reset(self):
        self._state = self.capital;
        self._episode_ended = False;
        # customized member
        self.index = 0;
        self.positions = list();
        return ts.restart(np.array([self._state], dtype = np.int32));
    def _step(self, action):
        # auto reset
        if self._episode_ended:
            return self.reset();
        # end episode conditions
        if self._state <= 0 or self.index == len(self.dataset):
            self._episode_ended = True;
        # state transition
        if action[0] == 0:
            # if buy
            # TODO


