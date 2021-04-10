import torch
import numpy as np

class Dueling(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Dueling, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [ torch.nn.Tanh() for l in  self.layers ]
        self.V = torch.nn.Linear(self.hidden_sizes[-1], 1)
        self.V_act = torch.nn.Tanh()
        self.A = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
        self.A_act = torch.nn.Tanh()

    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        V = self.V(self.V_act(x))
        A = self.A(self.A_act(x))
        return V + (A - A.mean())

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()
