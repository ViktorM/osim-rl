import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam, Nadam

from rl.agents import ContinuousDQNAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor

from osim.env import *
import argparse


class PendulumProcessor(Processor):
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=10000)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="large.h5f")
args = parser.parse_args()

ENV_NAME = 'Pendulum-v0'
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = GaitEnv(args.visualize)
#env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
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
print(V_model.summary())

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
print(mu_model.summary())

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
print(L_model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
processor = PendulumProcessor()
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.25, size=nb_actions)
agent = ContinuousDQNAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                           memory=memory, nb_steps_warmup=100, random_process=random_process,
                           gamma=.99, target_model_update=1e-3, processor=processor)

agent.compile(Nadam(lr=.001, clipnorm=1.), metrics=['mae'])
agent.fit(env, nb_steps=1000000, visualize=False, verbose=1, nb_max_episode_steps=200)

agent.save_weights('cdqn_{}_weights.h5f'.format("gait"), overwrite=True)
agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=200) 
