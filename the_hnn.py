import torch
from torchdiffeq import odeint

class SelfSupHNN(torch.nn.Module):
	def __init__(self, n, hidden_dim, nonlinearity):
		super().__init__()
		self.n = n
		self.linear1 = torch.nn.Linear(1+n*2, hidden_dim)
		self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = torch.nn.Linear(hidden_dim, 1, bias=False)
		
		torch.nn.init.orthogonal_(self.linear1.weight)
		torch.nn.init.orthogonal_(self.linear2.weight)
		torch.nn.init.orthogonal_(self.linear3.weight)
		
		self.nonlinearity = getattr(torch, nonlinearity)
	
	def forward(self, tx):
		return self.linear3(self.nonlinearity(self.linear2(self.nonlinearity(self.linear1(tx)))))
	
	def predict(self, t1, x1, t2):
		def fun(t, x):
			tx = torch.cat([torch.tensor([t]), x])
			tx.requires_grad_(True)
			self(tx).backward()
			_, dq, dp = torch.split(tx.grad, [1, self.n, self.n])
			return torch.cat((dp, -dq), 0)
		return odeint(fun, x1, torch.tensor([t1, t2]))[-1]
	
	def loss(self, data1, data2):
		loss = torch.nn.MSELoss()
		input = self.predict(data1['t'], torch.tensor(data1['x']), data2['t'])
		input.requires_grad_(True)
		target = torch.tensor(data2['x'])
		return loss(input, target)
