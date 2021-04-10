import numpy as np

class SumTree:
    def __init__(self, num_leaves):
        self.num_leaves = num_leaves
        self.tree = np.zeros((2 * num_leaves) - 1)
        self.filled_leaves = 0
    
    # method to add a new object into the tree
    def add_leaf(self, leaf, p):
        self._update(self.filled_leaves + self.num_leaves - 1, p - self.tree[self.filled_leaves + self.num_leaves - 1])
        self.filled_leaves += 1
        # if the tree overflows, write over the original values
        if self.filled_leaves >= self.num_leaves:
            self.filled_leaves = 0
        return
    
    # method to get the randomly sampled leaf and its index
    def get_leaf(self, low, high):
        rand_val = np.random.uniform(low, high)
        ind = self._find(0, rand_val) - (self.num_leaves - 1)
        return ind
    
    # method to update the probability of a leaf (indexing starts at 0)
    def update_leaf(self, ind, p):
        dif = p - self.tree[ind + self.num_leaves - 1]
        self._update(ind + self.num_leaves - 1, dif)
        return
    
    def get_total(self):
        return self.tree[0]
    
    def get_probs(self):
        return self.tree[self.num_leaves - 1:]

    def get_size(self):
        return self.num_leaves
    
    # internal method to update the probability of a leaf
    def _update(self, ind, p):
        self.tree[ind] += p

        parent = (ind - 1) // 2
        if parent >= 0:
            self._update(parent, p)
        return
    
    # internal method to find recursively find the leaf to sample
    def _find(self, cur_ind, rand_val):
        left = 2 * cur_ind + 1
        right = left + 1
        # if at a leaf, return this leaf
        if left >= (2 * self.num_leaves) - 1:
            return cur_ind
        # otherwise, return either its right or left child
        if rand_val <= self.tree[left]:
            return self._find(left, rand_val)
        # if the right tree is needed, must remove everything below
        return self._find(right, rand_val - self.tree[left])
       
        
        
        