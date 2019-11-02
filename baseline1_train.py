#!/usr/bin/python3

import tensorflow as tf;
from tf_agents.environments import tf_py_environment, suite_gym; # environment and problem
from tf_agents.agents.ppo import ppo_agent; # ppo agent
from tf_agents.networks import actor_distribution_network; # network structure
from tf_agents.trajectories import time_step, trajectory;
from tf_agents.replay_buffers import tf_uniform_replay_buffer; # replay buffer
from tf_agents.policies import random_tf_policy, policy_saver; # random policy
from FuturesEnv import FuturesEnv;

batch_size = 64;

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
        actor_net = actor_distribution_network.ActorDistributionNetwork(train_env.observation_spec(),
                                                                        train_env.action_spec()),
        check_numerics = True
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
