#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;
from tf_agents.environments import tf_py_environment, suite_gym; # environment and problem
from tf_agents.agents.ppo import ppo_agent; # ppo agent
from tf_agents.networks import network, utils;
from tf_agents.specs import tensor_spec, distribution_spec;
from tf_agents.trajectories import time_step, trajectory;
from tf_agents.replay_buffers import tf_uniform_replay_buffer; # replay buffer
from tf_agents.policies import random_tf_policy, policy_saver; # random policy
from FuturesEnv import FuturesEnv;

batch_size = 64;

class ActorNetwork(network.DistributionNetwork):

    def __init__(self, obs_spec, action_spec, logits_init_output_factor = 0.1, name = 'ActorNetwork'):

        input_param_shapes = tf.TensorSpec((3,), dtype = tf.int32);
        output_spec = distribution_spec.DistributionSpec(
            tfp.distributions.JointDistributionCoroutine,
            tf.nest.map_structure(lambda tensor_shape: tensor_spec.TensorSpec(shape = tensor_shape, dtype = action_spec.dtype), input_param_shapes),
            sample_spec = action_spec
        );
        super(ActorNetwork, self).__init__(
            input_tensor_spec = obs_spec,
            state_spec = (),
            output_spec = output_spec,
            name = name);
        num_actions = action_spec.maximum - action_spec.minimum + 1;
        self.denses = {
            'lever': tf.keras.layers.Dense(
                         num_actions[0],
                         kernel_initializer = tf.keras.initializers.VarianceScaling(scale = logits_init_output_factor),
                         bias_initializer = tf.keras.initializers.Zeros(),
                         name = 'lever_logits'),
            'sellprice': tf.keras.layers.Dense(
                              num_actions[1],
                              kernel_initializer = tf.keras.initializers.VarianceScaling(scale = logits_init_output_factor),
                              bias_initializer = tf.keras.initializers.Zeros(),
                              name = 'sell_price_logits'),
            'buyprice': tf.keras.layers.Dense(
                             num_actions[2],
                             kernel_initializer = tf.keras.initializers.VarianceScaling(scale = logits_init_output_factor),
                             bias_initializer = tf.keras.initializers.Zeros(),
                             name = 'buy_price_logits')};

    def call(self, inputs, step_type = None, network_state = ()):

        flatten = tf.keras.layers.Flatten()(inputs);
        flatten = tf.keras.layers.Lambda(lambda x: tf.cast(x, dtype = tf.float32))(flatten);
        logits = {key: dense(flatten) for key, dense in self.denses.items()};
        def model():
            lever = yield tfp.distributions.JointDistributionCoroutine.Root(tfp.distributions.Categorical(logits['lever']));
            sellprice = yield tfp.distributions.JointDistributionCoroutine.Root(tfp.distributions.Categorical(logits['sellprice']));
            buyprice = yield tfp.distributions.JointDistributionCoroutine.Root(tfp.distributions.Categorical(logits['buyprice']));
        action = tfp.distributions.JointDistributionCoroutine(model);
        return action;

class ValueNetwork(network.Network):

    def __init__(self, obs_spec, name = "ValueNetwork"):

        super(ValueNetwork, self).__init__(
            input_tensor_spec = obs_spec,
            state_spec = (),
            name = name);
        self.dense = tf.keras.layers.Dense(
            1, 
            kernel_initializer = tf.keras.initializers.Constant([2,1]),
            bias_initializer = tf.keras.initializers.Constant([5]));

    def call(self, inputs, step_type = None, network_state = ()):

        flatten = tf.keras.layers.Flatten()(inputs);
        flatten = tf.keras.layers.Lambda(lambda x: tf.cast(x, dtype = tf.float32))(flatten);
        logits = self.dense(flatten);
        return logits, network_state;

def main():

    # environment serves as the dataset in reinforcement learning
    train_env = tf_py_environment.TFPyEnvironment(FuturesEnv());
    eval_env = tf_py_environment.TFPyEnvironment(FuturesEnv());
    # create agent
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 1e-3);
    tf_agent = ppo_agent.PPOAgent(
        time_step.time_step_spec(train_env.observation_spec()),
        train_env.action_spec(),
        optimizer = optimizer,
        actor_net = ActorNetwork(train_env.observation_spec(),
                                 train_env.action_spec()),
        value_net = ValueNetwork(train_env.observation_spec()),
        normalize_observations = False,
        use_gae = True,
        lambda_value = 0.95
    );
    tf_agent.initialize();
    # replay buffer 
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec = tf_agent.collect_data_spec,
        batch_size = train_env.batch_size,
        max_length = 100000);
    # shape = batch x 2 x
    dataset = replay_buffer.as_dataset(num_parallel_calls = 3, sample_batch_size = batch_size, num_steps = 2).prefetch(3);
    iterator = iter(dataset);
    # policy saver
    saver = policy_saver.PolicySaver(tf_agent.policy);
    # training
    print("training...");
    for train_iter in range(20000):
        # collect initial trajectory to avoid a cold start
        if train_iter == 0:
            random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec());
            for _ in range(1000):
                status = train_env.current_time_step();
                action = random_policy.action(status);
                next_status = train_env.step(action.action);
                traj = trajectory.from_transition(status, action, next_status);
                replay_buffer.add_batch(traj);
        # collect trajectory for some step every training iteration
        for _ in range(1):
            status = train_env.current_time_step();
            action = tf_agent.collect_policy.action(status);
            next_status = train_env.step(action.action);
            traj = trajectory.from_transition(status, action, next_status);
            replay_buffer.add_batch(traj);
        # get a batch of dataset
        experience, unused_info = next(iterator);
        train_loss = tf_agent.train(experience);
        if tf_agent.train_step_counter.numpy() % 200 == 0:
            print('step = {0}: loss = {1}'.format(tf_agent.train_step_counter.numpy(), train_loss.loss));
        if tf_agent.train_step_counter.numpy() % 1000 == 0:
            # save policy
            saver.save('checkpoints/policy_%d' % tf_agent.train_step_counter.numpy());
            # get the average return for the updated policy
            total_return = 0.0;
            for _ in range(10):
                status = eval_env.reset();
                episode_return = 0.0;
                while not status.is_last():
                    action = tf_agent.policy.action(status);
                    status = eval_env.step(action.action);
                    episode_return += status.reward;
                total_return += episode_return;
            avg_return = total_return / 10;
            print('step = {0}: Average Return = {1}'.format(tf_agent.train_step_counter.numpy(), avg_return));
    # play futures environment for the last time
    print("evaluating...");
    status = eval_env.reset();
    total_reward = 0;
    while not status.is_last():
        action = tf_agent.policy.action(status);
        status = eval_env.step(action.action);
        total_reward += status.reward;
    print("total reward = " + str(total_reward));

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
