import numpy as np
from FinalProject.DDQN.src.SumTree import SumTree
import math

# class to store transitions
class Memory_PER_MS():
    def __init__(self, max_size=100000, n_step=1, gamma=0.9):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size=max_size
        self.n_step = n_step
        self.gamma=gamma
        # keep track of priorities
        self.priorities = np.zeros(max_size)
        self.tree = SumTree(self.max_size)

    def add_transition(self, transitions_new, eps, alpha, reward):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx,:] = np.asarray(transitions_new, dtype=object)
        # initialize the priority using error = reward
        self.priorities[self.current_idx] = np.asarray(self.get_priority(reward, eps, alpha), dtype=float)
        # add the transition to the sum tree
        self.size = min(self.size + 1, self.max_size)
        self.tree.add_leaf(self.transitions[self.current_idx,:], self.priorities[self.current_idx])
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        # pick the indicies with probabilities equal to the priorities
        inds = np.zeros(batch)
        seg = self.tree.get_total() / batch
        for i in range(0, batch):
            low = seg * i
            high = seg * (i + 1)
            inds[i] = int(self.tree.get_leaf(low, high))
        inds = inds.astype(int)
        # look ahead the correct number of steps
        s, a, ns, r, d = [], [], [], [], []
        for i in inds:
            rew_sum = 0
            state_look = self.transitions[i][3]
            done_look = self.transitions[i][4]
            for n in range(self.n_step):
                # protect against leaving the end of the buffer
                if len(self.transitions) <= i + n:
                    break
                rew_sum += (self.gamma ** n) * (1.0 - done_look) * self.transitions[i + n][2]
                state_look = self.transitions[i + n][3]
                if self.transitions[i + n][4] == True:
                    done_look = True
                    break
                done_look = False
            s.append(self.transitions[i][0])
            a.append(self.transitions[i][1])
            ns.append(state_look)
            r.append(rew_sum)
            d.append(done_look)
        return np.asarray([s, a, r, ns, d], dtype=object).transpose(), inds

    def get_all_transitions(self):
        return self.transitions[0:self.size]
    
    # function to return the priority = (error - eps) ^ a
    def get_priority(self, error, eps, alpha):
        return abs(error - eps) ** alpha
    
    def update_priority(self, ind, error, eps, alpha):
        for i in range(ind.size):
            self.priorities[ind[i]] = self.get_priority(error[i][0], eps, alpha).squeeze()
            self.tree.update_leaf(ind[i], self.priorities[ind[i]])
        return
    
    def get_IS_w(self, inds, beta):
        cur = self.tree.get_probs()[inds]
        beta = -1 * beta
        n_p = cur / self.tree.get_size()
        weights = n_p ** beta
        for i in range(weights.size):
            if math.isinf(weights[i]):
                weights[i] = 0
        return weights / max(weights)
            
        
