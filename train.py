from argparse import ArgumentParser
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from dataset import load_dataset
from hnn import HNN
from utils import L2_loss
from nn_models import MLP
from torchdiffeq import odeint

def get_args():
	parser = ArgumentParser(description=None)
	parser.add_argument('--dataset_dir', type=str)
	parser.add_argument('--learn_rate', default=1e-3, type=float)
	parser.add_argument('--hidden_dim', default=200, type=int)
	parser.add_argument('--nonlinearity', default='tanh', type=str)
	parser.add_argument('--save_dir', default='path')
	parser.add_argument('--name', default='selfsup_hnn')
	parser.add_argument('--n', type=int)
	parser.add_argument('--log_dir', default=None, type=str)
	parser.add_argument('--field_type', default='solenoidal', type=str)
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--epoches', default=200, type=int)
	return parser.parse_args()

total_i = 0
def log(writer, loss):
	global total_i
	writer.add_scalar('Loss/train', loss, total_i)
	total_i += 1

def train(writer, model, data, optim):
	x1 = torch.tensor(data['x1'], requires_grad=True)
	x2 = torch.tensor(data['x2'])
	loss = L2_loss(odeint(lambda t, x: model.time_derivative(x), x1, torch.tensor(data['t']))[-1], x2)
	loss.backward(); optim.step(); optim.zero_grad()
	log(writer, loss)

if __name__ == '__main__':
	args = get_args()
	if not args.dataset_dir:
		print('Specify dataset dir with "--dataset_dir"')
		exit()
	if not args.n:
		print('Specify DOF with "--n"')
		exit()
	
	if torch.cuda.is_available():
		torch.device('cuda')
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
	torch.autograd.set_detect_anomaly(True)
	
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	writer = SummaryWriter(log_dir=args.log_dir)
	nn_model = MLP(args.n * 2, args.hidden_dim, 2, args.nonlinearity)
	model = HNN(args.n * 2, differentiable_model=nn_model, field_type=args.field_type)
	
	optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
	for data in os.listdir(args.dataset_dir):
		data = load_dataset(f'{args.dataset_dir}/{data}')
		for epoch in range(args.epoches):
			train(writer, model, data, optim)
	
	writer.close()
	
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	torch.save(model.state_dict(), f"{args.save_dir}/{args.name}.tar")
