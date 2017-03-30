import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.layers.noise import GaussianNoise
from keras.optimizers import RMSprop, Adam, Nadam

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
parser = argparse.ArgumentParser(description='Train or test deep neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=500000)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--start_weights', dest='start_weights', action='store', default="best/ddpg_elu_rew2_best_actor.h5f")
parser.add_argument('--model', dest='model', action='store', default="CDQN/cdqn_grand.h5f")
parser.add_argument('--sigma', dest='sigma', action='store', default=0.25)
parser.add_argument('--theta', dest='theta', action='store', default=0.15)
parser.add_argument('--gamma', dest='gamma', action='store', default=0.99)
parser.add_argument('--rseed', dest='rseed', action='store', default=53, type=int)
args = parser.parse_args()

ENV_NAME = 'Pendulum-v0'
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = GaitEnv(args.visualize)
#env = gym.make(ENV_NAME)
print ("Random seed: %i\n", args.rseed)
np.random.seed(args.rseed)
random.seed(args.rseed)
env.seed(args.rseed)

assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Total number of steps in training
nallsteps = args.steps

init = 'lecun_uniform'
                
# Build all necessary models: V, mu, and L networks.
V_model = Sequential()
V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
V_model.add(GaussianNoise(0.01)) # add to the command line!
V_model.add(Dense(32, init = init))
V_model.add(ELU())
V_model.add(Dense(32, init = init))
V_model.add(ELU())
V_model.add(Dense(32, init = init))
V_model.add(ELU())
V_model.add(Dense(1))
V_model.add(Activation('linear'))
print(V_model.summary())

mu_model = Sequential()
mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
mu_model.add(GaussianNoise(0.01)) # add to the command line!
mu_model.add(Dense(32, init = init))
mu_model.add(ELU())
mu_model.add(Dense(32, init = init))
mu_model.add(ELU())
mu_model.add(Dense(32, init = init))
mu_model.add(ELU())
mu_model.add(Dense(nb_actions, init = init))
mu_model.add(GaussianNoise(0.01))
mu_model.add(Activation('linear'))
print(mu_model.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
x = merge([action_input, Flatten()(observation_input)], mode='concat')
x = GaussianNoise(0.01)(x)
x = Dense(64, init = init)(x)
x = ELU()(x)
x = Dense(64, init = init)(x)
x = ELU()(x)
x = Dense(64, init = init)(x)
x = ELU()(x)
x = Dense(((nb_actions * nb_actions + nb_actions) / 2))(x)
x = Activation('linear')(x)
L_model = Model(input=[action_input, observation_input], output=x)
print(L_model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
processor = PendulumProcessor()
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=float(args.theta), mu=0., sigma=float(args.sigma), size=nb_actions)
agent = ContinuousDQNAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                           memory=memory, nb_steps_warmup=100, random_process=random_process,
                           gamma=float(args.gamma), target_model_update=1e-3, processor=processor)
agent.compile(Nadam(lr=.001, clipnorm=2.), metrics=['mae'])

if args.train:
#    agent.load_weights(args.start_weights)
    checkpoint_weights_filename = 'CDQN/cdqn_grand_Gait_{step}.h5f'
    log_filename = 'CDQN/cdqn_gauss_rand_{}.json'.format('Gait')
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
    callbacks += [FileLogger(log_filename, interval=10000)]
    agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=1000)
    agent.save_weights('CDQN/cdqn_{}_grand.h5f'.format("Gait"), overwrite=True)

if not args.train:
    agent.load_weights(args.model)
    agent.test(env, nb_episodes=3, visualize=False, nb_max_episode_steps=500) 
