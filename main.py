import time
import numpy as np

import config
import data_loader
import env
import agent
import policy

DL = data_loader.DataLoader()
DL.read_data(split='test')

env = env.ENV(DL)

greedy = policy.MaskedGreedyPolicy()
multi = policy.MaskedMultiDisPolicy()

agent = agent.pgAgent(env=env, nb_warm_up=400, policy=multi, testPolicy=greedy, gamma=0.95, lr=0.0001,
                      memory_limit=2000, batchsize=64, train_interval=200)

time1 = time.time()
for round in range(1000):
    print("------------------------------------------------------")
    print('\n\n train ' + str(round) + '_' + str((time.time()-time1)/60))
    print("------------------------------------------------------")

    agent.fit(10000)
    # env.nb_steps_warmup = 0
    # env.test(env_gym, nb_episodes=1, visualize=False, verbose=2) # for dqn
    agent.test()