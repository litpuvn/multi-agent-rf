#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from multiagent.dqn_policy import DQNPolicy

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='volunteer_victim.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world=world, reset_callback=scenario.reset_world, reward_callback=scenario.reward,
                        observation_callback=scenario.observation, info_callback=None, shared_viewer = False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [DQNPolicy(env,i) for i in range(env.n)]

    N_episodes_train = 1000
    N_episodes_test = 20

    for episode in range(N_episodes_train + N_episodes_test):
        if episode >= N_episodes_train:
            print('no exploration, perform with what agent learnt')
            # agent.epsilon = 0  # set no exploration for test episodes
        # execution loop
        obs_n = env.reset()
        n_iter_episode = 0
        while not env.is_terminated():
            # query for action from each agent's policy
            act_n = []
            for i, policy in enumerate(policies):
                act_n.append(policy.action(obs_n[i]))
            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)
            # render all agent views
            env.render()
            # display rewards
            #for agent in env.world.agents:
            #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))

            n_iter_episode +=1
            if (episode >= N_episodes_train) and (n_iter_episode > 2000):
                raise IOError("Bad policy found! Non-terminal episode!")

        print('Episode', episode, ' done in', n_iter_episode, ' iterations')