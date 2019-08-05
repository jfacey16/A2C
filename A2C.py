
# coding: utf-8

# In[1]:


import gym

import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K

import itertools
import random

from tqdm import tqdm_notebook as tqdm # progress meter
import matplotlib.pyplot as plt

class A2C(object):
    """
    An A2C (advantage actor critic) model. 
    """
    def __init__(self, game = 'CartPole-v1'):
        self.env = gym.make(game)
        # four-parameter state space
        self.state_size = self.env.observation_space.shape[0]
        # Discrete two-element action space (left or right)
        self.action_size = self.env.action_space.n

        # Lists for tracking how the actor and critic are performing
        # Results can be graphed using self.diagnostics()
        self.policyrecord = []
        self.weightentryrecord = []
        self.criticerrors = []
        self.movequalities = []

        # hyperparameters
        self.discount_factor = 0.99
        self.actor_lr = 5e-3
        self.critic_lr = 1e-3
        self.momentum = 0.8

        # Keras models for actor and critic
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # Keras functions for computing gradients of log policy values
        # For performance reasons, these are instantiated in init
        # rather than in the method that calls them 
        self.gradfuncs = [self._build_gradient_func(action) for action in range(self.action_size)]

    def _build_actor(self):
        """
        A two-layer MLP with 24 units in the middle layer and relu activation

        Takes in state vectors and returns two-entry vectors which represent the 
        mass assigned to each action by the policy
        """
        actor = keras.models.Sequential()
        actor.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        actor.add(keras.layers.Dense(2, activation='softmax'))
        actor.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=self.actor_lr))
        
        for layer in actor.layers:
            layer.prevgrads = [0,0]
        return actor

    def _build_critic(self):
        """
        A two-layer MLP with 24 units in the middle layer, relu activation, and 
        mean-squared-error loss

        Takes in state vectors and returns a scalar which represents the value of 
        the state.
        """
        critic = keras.models.Sequential()
        critic.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        critic.add(keras.layers.Dense(1, activation='linear'))
        critic.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(lr=self.critic_lr))
        return critic

    def _get_actor_weights(self):
        """
        Return actor model's weight (and bias) tensors, as Keras `Variable` objects
        """
        return list(itertools.chain(*[layer.weights for layer in self.actor.layers]))

    def _build_gradient_func(self, action):
        """
        Make a function which returns gradients of the `action`th component of the output
        of the actor model, with respect to the input and weights of the actor model
        """
        predicted_action_mass = K.dot(K.one_hot([action],self.action_size), # select output component
                                      K.transpose(K.log(self.actor.output)))
        grads = K.gradients([predicted_action_mass], self._get_actor_weights())
        return K.function(self.actor.inputs, grads)
    
    def get_action(self, state):
        """
        Return an action sampled from the distribution specified by the policy
        """
        policy = self.actor.predict(state[np.newaxis,:], batch_size=1).flatten()
        self.policyrecord.append(policy[0])
        self.weightentryrecord.append(self.actor.layers[0].get_weights()[0][0,0])
        return np.random.choice([0,1],p=policy)

    def grad_log_policy(self, state, action):
        """
        returns the value of ∂/∂θ(log π(θ, s_t, a_t)) (used in formula for ∂J/∂θ)
        """
        return self.gradfuncs[action]([state[np.newaxis,:]])

    def _re_pair(self, v):
        """
        The inverse of the flattening-a-list-of-pairs operation
        """
        return [[v[2*k], v[2*k+1]] for k in range(len(v)//2)]

    def advantage(self, state, action, reward, next_state, done):
        """
        Return an approximation of the advantage of the given action

        If the game is over, returns the reward minus the predicted reward
        from the current state. Otherwise, returns reward plus discounted 
        value of next state minus value of current state
        """
        if done:
            # I couldn't figure out how to get predicted reward from the current state
            # this is the last thing to add, and would be added to reward in this function
            x = reward + self.critic.predict(state[np.newaxis,:], batch_size=1)
            x = np.array([[x]])
        else:
            x = reward + self.discount_factor * self.critic.predict(next_state[np.newaxis,:], batch_size=1) - self.critic.predict(state[np.newaxis,:], batch_size=1)

        return x.reshape(1,)
    def update_actor(self, state, action, reward, next_state, done):
        """
        Update actor weights
        """
        weights = [layer.get_weights() for layer in self.actor.layers]
        grad_log_policy = self._re_pair(self.grad_log_policy(state, action))
        alpha = self.momentum

        advantage = self.advantage(state, action, reward, next_state, done)

        for (weightpair,gradpair,layer) in zip(weights, grad_log_policy, self.actor.layers):
            layer.set_weights([weight + self.actor_lr * advantage * (alpha * prevgrad + (1-alpha) * grad) 
                               for (weight, grad, prevgrad) in zip(weightpair, gradpair, layer.prevgrads)])
            layer.prevgrads = gradpair

    def critic_target(self, state, reward, next_state, done):
        """
        Returns target for updating critic weights 

        If the game is over, returns the reward (as a single-entry, rank-2 ndarray). 
        Otherwise, returns the sum of the reward and the discounted value of the next state
        """
        if done:
            x = reward
            x = np.array([[x]])
        else:
            x = reward + self.discount_factor * self.critic.predict(next_state[np.newaxis,:], batch_size=1)
        return x

    def update_critic(self, state, reward, next_state, done):
        """
        Update critic weights
        """
        target = self.critic_target(state, reward, next_state, done)
        self.criticerrors.append(self.critic.predict(state[np.newaxis,:]) - target)
        self.critic.fit(state[np.newaxis,:], target, epochs=1, verbose=0)
        
    def explore_and_train(self, num_episodes = 600, learn = True):
        """
        Interact with the environment and (optionally) learn. 

        progress: show a progress bar
        show: render the environment (so you can watch the cart and pole during training)
        learn: whether to update actor and critic weights
        """
        episodes = []
        self.scores = []
        R = range(num_episodes)
        for episode_index in R:
            score = 0
            state = self.env.reset()
            while True: 
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.movequalities.append(self.advantage(state, action, reward, next_state, done))
                reward = reward if not done else -20 # extra penalty for a move that ends the game
                score += reward
                state = next_state
                if learn:
                    self.update_actor(state, action, reward, next_state, done)
                    self.update_critic(state,        reward, next_state, done)
                if done:
                    break
            self.scores.append(score + (20 if score != 500 else 0))
        plt.plot(self.scores)
        plt.show()

    def diagnostics(self):
        plt.figure(figsize=(12,3))
        plt.subplot(1,4,1)
        plt.title("Policy probabilities")
        plt.plot(self.policyrecord)
        plt.subplot(1,4,2)
        plt.title("Example actor weight")
        plt.plot(self.weightentryrecord)
        plt.subplot(1,4,3)
        plt.title("Errors in critic predictions")
        plt.plot(np.squeeze(self.criticerrors))
        plt.subplot(1,4,4)
        plt.title("Move quality")
        plt.plot(np.squeeze(self.movequalities))
        plt.show()

def test_a2c():
    A = A2C()
    dummy_state = np.array([ 2.21575524e-02, -1.64592822e-02,  6.30291863e-05,  1.03607773e-02])
    dummy_action = 0
    dummy_reward = 1.0
    dummy_next_state = dummy_state + np.full(4,0.01)
    dummy_done = False
    A.get_action(dummy_state)
    A.grad_log_policy(dummy_state, dummy_action)
    assert A.advantage(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done).shape == (1,)
    assert A.advantage(dummy_state, dummy_action, dummy_reward, dummy_next_state, not dummy_done).shape == (1,)
    A.update_actor(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)
    assert A.critic_target(dummy_state, dummy_reward, dummy_next_state, dummy_done).shape == (1,1)
    A.explore_and_train(progress = True, learn = True, show = False)

if __name__ == "__main__":
    test_a2c()

