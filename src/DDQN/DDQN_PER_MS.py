import gym
from gym import spaces
import numpy as np
import itertools
import time
import torch
import pylab as plt
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import FinalProject.DDQN.src.memory as mem   
import FinalProject.DDQN.src.memory_per_ms as mem_per_ms
from FinalProject.DDQN.src.feedforward import Feedforward
from FinalProject.DDQN.src.dueling import Dueling

class QFunction(Dueling):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100], 
                 learning_rate = 0.0002):
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes, 
                         output_size=action_dim)
        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr=learning_rate, 
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss() # MSELoss()
    
    def fit(self, observations, actions, targets, weights):
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        acts = torch.from_numpy(actions)
        pred = self.Q_value(torch.from_numpy(observations).float(), acts)
        targets = torch.from_numpy(targets).float()
        weights = torch.Tensor(weights)
        # Compute Loss
        loss = 0
        for i in range(pred.shape[0]):
            loss += self.loss(pred[i], targets[i]) * weights[i]
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def Q_value(self, observations, actions):
        if actions[:,None].dtype == torch.int32:
            actions = actions.type(torch.int64)
        return self.forward(observations).gather(1, actions[:,None])
    
    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1, keepdims=True)
        
    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)
    
class DQNAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.    
    """
    def __init__(self, observation_space, action_space, action_space_size, **userconfig):
        
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.discrete.Discrete):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Reqire Discrete.)'.format(action_space, self))
        
        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space_size
        self._config = {
            "eps": 0.05,            # Epsilon in epsilon greedy policies                        
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate": 0.0002,
            "update_target_every": 20,
            "use_target_net":True,
            "per_eps": 0.01,
            "alpha": 0.5,
            "beta": 0.4,
            "beta_inc": 0.001,
            "n_step": 1
            
        }
        self._config.update(userconfig)        
        self._eps = self._config['eps']
        
        self.buffer = mem_per_ms.Memory_PER_MS(max_size=self._config["buffer_size"], n_step=self._config["n_step"], gamma=self._config['discount'])
                
        # Q Network
        self.Q = QFunction(observation_dim=self._observation_space.shape[0], 
                           action_dim=self._action_n,
                           learning_rate = self._config["learning_rate"])
        # Q Network
        self.Q_target = QFunction(observation_dim=self._observation_space.shape[0], 
                                  action_dim=self._action_n,
                                  learning_rate = 0)
        self._update_target_net()
        self.train_iter = 0
            
    def _update_target_net(self):        
        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else: 
            action = self._action_space.sample()        
        return action
    
    def store_transition(self, transition, reward):
        self.buffer.add_transition(transition, self._config['per_eps'], self._config['alpha'], reward)
            
    def train(self, iter_fit=32):
        losses = []
        self.train_iter += 1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net()                
        for i in range(iter_fit):

            # sample from the replay buffer
            data, inds = self.buffer.sample(batch=self._config['batch_size'])
            s = np.stack(data[:,0]) # s_t
            a = np.stack(data[:,1]) # a_t
            rew = np.stack(data[:,2])[:,None] # rew  (batchsize,1)
            s_prime = np.stack(data[:,3]) # s_t+1
            done = np.stack(data[:,4])[:,None] # done signal  (batchsize,1)
            
            if self._config["use_target_net"]:
                v_prime = self.Q_target.maxQ(s_prime)
            else:
                v_prime = self.Q.maxQ(s_prime)
            # target
            gamma=self._config['discount']                                                
            td_target = rew + (gamma ** self._config["n_step"]) * (1.0-done) * v_prime
            self.buffer.update_priority(inds, td_target - v_prime, self._config['per_eps'], self._config['per_eps'])
            
            # optimize the lsq objective
            weights = self.buffer.get_IS_w(inds, self._config['beta'])
            self._config['beta'] = min(1, self._config['beta'] + self._config['beta_inc'])
            
            fit_loss = self.Q.fit(s, a, td_target, weights)
            
            
            losses.append(fit_loss)
                
        return losses