import opensim as osim
from osim.http.client import Client
from osim.env import *
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam, Nadam

from rl.agents import ContinuousDQNAgent
from rl.memory import SequentialMemory
from rl.core import Processor

import numpy as np
import argparse

# Settings
CROWDAI_TOKEN = "ef125dcc4a82b5f162cc7f401c4c58a1"
remote_base = 'http://54.154.84.135:80'

# Command line parameters
parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--model', dest='model', action='store', default="large.h5f")
parser.add_argument('--token', dest='token', action='store', default=CROWDAI_TOKEN)
args = parser.parse_args()

env = GaitEnv(visualize=False)

nb_actions = env.action_space.shape[0]
                
# Build all necessary models: V, mu, and L networks.
V_model = Sequential()
V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
V_model.add(Dense(32))
V_model.add(Activation('relu'))
V_model.add(Dense(32))
V_model.add(Activation('relu'))
V_model.add(Dense(32))
V_model.add(Activation('relu'))
V_model.add(Dense(1))
V_model.add(Activation('linear'))

mu_model = Sequential()
mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
mu_model.add(Dense(32))
mu_model.add(Activation('relu'))
mu_model.add(Dense(32))
mu_model.add(Activation('relu'))
mu_model.add(Dense(32))
mu_model.add(Activation('relu'))
mu_model.add(Dense(nb_actions))
mu_model.add(Activation('linear'))

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
x = merge([action_input, Flatten()(observation_input)], mode='concat')
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(((nb_actions * nb_actions + nb_actions) / 2))(x)
x = Activation('linear')(x)
L_model = Model(input=[action_input, observation_input], output=x)

memory = SequentialMemory(limit=10000, window_length=1)
agent = ContinuousDQNAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model = mu_model, memory = memory)
agent.compile(Nadam(lr=.001, clipnorm=1.), metrics=['mae'])
agent.load_weights('cdqn_gait_weights.h5f')
#agent.load_weights(args.model)

client = Client(remote_base)

# Create environment
env_id = "Gait"
observation = client.env_create(args.token)

# Run a single step
for i in range(501):
    v = np.array(observation).reshape((-1, 1 ,env.observation_space.shape[0]))
    [observation, reward, done, info] = client.env_step(args.token, agent.select_action()[0].tolist(), True)
    if done:
        break

client.submit(args.token)

