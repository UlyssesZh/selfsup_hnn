import torch
from torch.autograd import grad

def dx_fun(h, batch=True):
	if batch:
		def dx(t, x):
			n = x.shape[1] // 2
			partial_q, partial_p = torch.split(grad(h(x).sum(), x, create_graph=True)[0], [n, n], dim=1)
			return torch.cat([partial_p, -partial_q], dim=1)
	else:
		def dx(t, x):
			n = len(x) // 2
			partial_q, partial_p = torch.split(grad(h(x), x, create_graph=True)[0], [n, n])
			return torch.cat([partial_p, -partial_q])
	return dx
