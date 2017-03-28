# Derived from keras-rl
#import opensim as osim
import osim
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.optimizers import Adam, Nadam

import numpy as np

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from osim.env import *

import argparse
import math

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=200000)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--start_weights', dest='start_weights', action='store', default="weights/ddpg_elu_rew2_best.h5f")
parser.add_argument('--model', dest='model', action='store', default="weights/ddpg_elu_rew3.h5f")
args = parser.parse_args()

# Load walking environment
env = GaitEnv(args.visualize)

nb_actions = env.action_space.shape[0]

# Total number of steps in training
nallsteps = args.steps

init = 'lecun_uniform'

# Create networks for DDPG
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(32, init = init))
actor.add(ELU())
actor.add(Dense(32, init = init))
actor.add(ELU())
actor.add(Dense(32, init = init))
actor.add(ELU())
actor.add(Dense(nb_actions, init = init))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = merge([action_input, flattened_observation], mode='concat')
x = Dense(64, init = init)(x)
x = ELU()(x)
x = Dense(64, init = init)(x)
x = ELU()(x)
x = Dense(64, init = init)(x)
x = ELU()(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=[action_input, observation_input], output=x)
print(critic.summary())

# Set up the agent for training
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.12, mu=0., sigma=.11, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)

agent.compile([Nadam(lr=0.001, clipnorm=1.), Nadam(lr=0.001, clipnorm=1.)], metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if args.train:
    agent.load_weights(args.start_weights)
    checkpoint_weights_filename = 'ddpg_elu_best_Gait_{step}.h5f'
  #  log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=20000)]
    agent.fit(env, callbacks=callbacks, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=env.timestep_limit, log_interval=10000)
    # After training is done, we save the final weights.
    agent.save_weights(args.model, overwrite=True)

if not args.train:
    agent.load_weights(args.model)
    # Finally, evaluate our algorithm for 1 episode.
    agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=500)
