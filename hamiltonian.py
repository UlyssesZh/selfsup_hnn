import torch
from torch.autograd import grad

def dx_fun(h):
	def dx(t, x):
		tx = torch.cat([torch.tensor([t]), x])
		tx.requires_grad_(True)
		n = len(x) // 2
		_, partial_q, partial_p = torch.split(grad(h(tx), tx, create_graph=True)[0], [1, n, n])
		return torch.cat([partial_p, -partial_q])
	return dx
