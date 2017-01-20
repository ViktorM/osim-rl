import opensim as osim
import numpy as np
import sys
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge, GaussianNoise
from keras.optimizers import Nadam, Adam, Adadelta, RMSprop

import numpy as np

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from environments.arm import ArmEnv
from environments.human import *


import argparse
import math

# Some meta parameters
nallsteps = 10000000

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--output', dest='output', action='store', default=None)
parser.add_argument('--env', dest='env', action='store', default="Arm")
parser.add_argument('--sigma', dest='sigma', action='store', default=0.3)
parser.add_argument('--theta', dest='theta', action='store', default=0.15)
args = parser.parse_args()


#args.sigma = 0.2 # Defaut
#args.theta = 0.1 # Defaut

args.sigma = 0.21
args.theta = 0.12
args.env = "Stand"


if args.env == "Gait":
    env = GaitEnv(args.visualize)
elif args.env == "Stand":
    env = StandEnv(args.visualize)
elif args.env == "Hop":
    env = HopEnv(args.visualize)
elif args.env == "Leg":
    env = LegEnv(args.visualize)
else:
    env = ArmEnv(args.visualize)

nb_actions = env.action_space.shape[0]
print (env.observation_space.shape)

init_name='lecun_uniform'
# 32/64 dense default

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(24, init = init_name))
actor.add(Activation('relu'))
#actor.add(GaussianNoise(0.008))
actor.add(Dense(24, init = init_name))
actor.add(Activation('relu'))
actor.add(Dense(24, init = init_name))
actor.add(Activation('relu'))
actor.add(Dense(24, init = init_name))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions, init = init_name))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = merge([action_input, flattened_observation], mode='concat')
x = Dense(48, init = init_name)(x)
x = Activation('relu')(x)
#x = GaussianNoise(0.007)(x)
x = Dense(48, init = init_name)(x)
x = Activation('relu')(x)
x = Dense(48, init = init_name)(x)
x = Activation('relu')(x)
x = Dense(48, init = init_name)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=[action_input, observation_input], output=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=float(args.theta), mu=0., sigma=float(args.sigma), size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.995, target_model_update=1e-3,
                  delta_range=(-100., 100.))

agent.compile([Nadam(lr=0.0015), Nadam(lr=0.0015)], metrics=['mae'])

prefix = args.output if args.output else "%s_s%f_t%f" % (args.env ,float(args.sigma), float(args.theta))

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if args.train:
    #agent.load_weights("%s.h5f" % args.output)
    agent.fit(env, nb_steps=nallsteps, visualize=True, verbose=1, nb_max_episode_steps=env.timestep_limit, log_interval=10000, prefix=prefix)
    # After training is done, we save the final weights.
    agent.save_weights("%s.h5f" % args.output, overwrite=True)

if not args.train:
    agent.load_weights("%s.h5f" % args.output)
    # Finally, evaluate our algorithm for 5 episodes.
    if args.env != "Arm":
        agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=10000)
    else:
        for i in range(10000):
            if i % 300 == 0:
                env.new_target()
                print("\n\nTarget shoulder = %f, elbow = %f" % (env.shoulder,env.elbow)) 

            obs = env.get_observation()
            print "Actual shoulder = %f, elbow = %f\r" % (obs[2],obs[3]),
            env.step(agent.forward(obs))


