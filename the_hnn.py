import torch
from torchdiffeq import odeint

import hamiltonian

class SelfSupHNN(torch.nn.Module):
	def __init__(self, n, hidden_dim, nonlinearity):
		super().__init__()
		self.n = n
		self.linear1 = torch.nn.Linear(n*2, hidden_dim)
		self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = torch.nn.Linear(hidden_dim, 1, bias=False)
		
		torch.nn.init.orthogonal_(self.linear1.weight)
		torch.nn.init.orthogonal_(self.linear2.weight)
		torch.nn.init.orthogonal_(self.linear3.weight)
		
		self.nonlinearity = getattr(torch, nonlinearity)
		
		self.dx = hamiltonian.dx_fun(self)
	
	def forward(self, tx):
		return self.linear3(self.nonlinearity(self.linear2(self.nonlinearity(self.linear1(tx))))).squeeze()
	
	def loss(self, x1, x2, t):
		x1 = torch.tensor(x1, requires_grad=True)
		x2 = torch.tensor(x2)
		return torch.nn.MSELoss()(odeint(self.dx, x1, torch.tensor(t))[-1], x2)
