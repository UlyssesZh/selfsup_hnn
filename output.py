import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torch

from nn_models import MLP
from hnn import HNN
from hamiltonian import dx_fun
from utils import L2_loss

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
	parser.add_argument('--field_type', default='solenoidal', type=str)
	return parser.parse_args()

if __name__ == '__main__':
	args = get_args()
	initial_list = [float(s) for s in args.initial.split(',')]
	nn_model = MLP(args.n * 2, args.hidden_dim, 2, args.nonlinearity)
	model = HNN(args.n * 2, differentiable_model=nn_model, field_type=args.field_type)
	fig, ax = plt.subplots()
	t = torch.linspace(0, args.max_t, args.steps)
	initial = torch.tensor(initial_list, requires_grad=True)
	truth = odeint(dx_fun(eval(f'lambda x: {args.ground_truth}'), batch=False), initial, t).detach()
	initial = torch.tensor([initial_list], requires_grad=True)
	for path in args.load_path.split(','):
		model.load_state_dict(torch.load(path))
		model.cpu()
		x = odeint(lambda t, x: model.time_derivative(x), initial, t).detach().squeeze()
		loss = torch.zeros(args.steps)
		for i in range(args.steps):
			loss[i] = L2_loss(x[i], truth[i])
		ax.plot(t, loss)
	plt.show()
