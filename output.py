import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torch

from the_hnn import SelfSupHNN
import hamiltonian

def get_args():
	parser = ArgumentParser(description=None)
	parser.add_argument('--load_path', type=str)
	parser.add_argument('--initial', type=str)
	parser.add_argument('--max_t', default=10, type=float)
	parser.add_argument('--steps', default=100, type=int)
	parser.add_argument('--n', type=int)
	parser.add_argument('--hidden_dim', default=200, type=int)
	parser.add_argument('--nonlinearity', default='tanh', type=str)
	parser.add_argument('--ground_truth', type=str)
	return parser.parse_args()

if __name__ == '__main__':
	args = get_args()
	initial = torch.tensor([float(s) for s in args.initial.split(',')], requires_grad=True)
	model = SelfSupHNN(args.n, args.hidden_dim, args.nonlinearity)
	model.load_state_dict(torch.load(args.load_path))
	model.cpu()
	fig, ax = plt.subplots()
	t = torch.linspace(0, args.max_t, args.steps)
	x = odeint(hamiltonian.dx_fun(model, batch=False), initial, t).detach()
	y = odeint(hamiltonian.dx_fun(eval(f'lambda x: {args.ground_truth}'), batch=False), initial, t).detach()
	ax.plot(t, x[:, 0])
	ax.plot(t, x[:, 1])
	ax.plot(t, y[:, 0])
	ax.plot(t, y[:, 1])
	plt.show()
