import opensim as osim
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam, Adadelta, RMSprop, Nadam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *


import argparse
import math


# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--output', dest='output', action='store', default=None)
parser.add_argument('--env', dest='env', action='store', default="Arm")
parser.add_argument('--sigma', dest='sigma', action='store', default=0.25)
parser.add_argument('--theta', dest='theta', action='store', default=0.15)
parser.add_argument('--gamma', dest='gamma', action='store', default=0.99)
parser.add_argument('--target_model_update', dest='target_model_update', action='store', default=1e-3, type=float)
parser.add_argument('--nb_steps', dest='nb_steps', action='store', default=1000000, type=int)
parser.add_argument('--rseed', dest='rseed', action='store', default=random.randint(0, 4294967295), type=int)
args = parser.parse_args()

print "Random seed: %i\n" % args.rseed
np.random.seed(args.rseed)
random.seed(args.rseed)


ENVS = {"Gait": GaitEnv,
        "Stand": StandEnv,
        "Hop": HopEnv,
        "Crouch": CrouchEnv,
        "Arm": ArmEnv}

if not args.env in ENVS.keys():
    print("Environment %s does not exist" % args.env)

env = ENVS[args.env](args.visualize)

nb_actions = env.action_space.shape[0]

init_name='lecun_uniform'

# Next, we build a simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(30, init = init_name))
actor.add(Activation('relu'))
actor.add(Dense(30, init = init_name))
actor.add(Activation('relu'))
actor.add(Dense(30, init = init_name))
actor.add(Activation('relu'))
actor.add(Dense(30, init = init_name))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions, init = init_name))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = merge([action_input, flattened_observation], mode='concat')
x = Dense(60, init = init_name)(x)
x = Activation('relu')(x)
x = Dense(60, init = init_name)(x)
x = Activation('relu')(x)
x = Dense(60, init = init_name)(x)
x = Activation('relu')(x)
x = Dense(60, init = init_name)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=[action_input, observation_input], output=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=200000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=float(args.theta), mu=0., sigma=float(args.sigma), size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=float(args.gamma), batch_size=32, target_model_update=float(args.target_model_update),
                  delta_range=(-100., 100.))

agent.compile([Nadam(lr=0.0005), Nadam(lr=0.001)], metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if args.train:
    agent.fit(env, nb_steps=env.nb_steps, visualize=True, verbose=1, nb_max_episode_steps=env.timestep_limit, log_interval=10000)
    # After training is done, we save the final weights.
    agent.save_weights(args.output, overwrite=True)

if not args.train:
    agent.load_weights(args.output)
    # Finally, evaluate our algorithm for 5 episodes.

    if args.env != "Arm":
        agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=10000)
    else:
        for i in range(10000):
            if i % 300 == 0:
                env.new_target()
                print("Target shoulder = %f, elbow = %f" % (env.shoulder,env.elbow)) 
            
            env.step(agent.forward(env.get_observation()))
