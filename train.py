from argparse import ArgumentParser
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import load_dataset
from the_hnn import SelfSupHNN

def get_args():
	parser = ArgumentParser(description=None)
	parser.add_argument('--dataset_dir', type=str)
	parser.add_argument('--learn_rate', default=1e-3, type=float)
	parser.add_argument('--hidden_dim', default=520, type=int)
	parser.add_argument('--nonlinearity', default='tanh', type=str)
	parser.add_argument('--save_dir', default='path')
	parser.add_argument('--name', default='selfsup_hnn')
	parser.add_argument('--n', type=int)
	parser.add_argument('--log_dir', default='runs', type=str)
	parser.add_argument('--batch_size', default=10, type=int)
	return parser.parse_args()

total_i = 0
def log(writer, loss):
	global total_i
	writer.add_scalar('Loss/train', loss, total_i)
	total_i += 1

def train(writer, model, dataset, learn_rate, batch_size):
	dataset = load_dataset(dataset)
	optim = torch.optim.Adam(model.parameters(), learn_rate, weight_decay=1e-4)
	
	dataset_size = len(dataset)
	data_size = len(dataset[0])
	
	for i in range(0, data_size - 1, batch_size):
		last_loss = None
		for j in range(dataset_size):
			loss = torch.tensor(0.)
			k = 0
			for k in range(batch_size):
				if i + k == data_size - 1:
					break
				loss += model.loss(dataset[j][i + k], dataset[j][i + k + 1])
			loss /= k
			optim.zero_grad()
			loss.backward()
			optim.step()
			last_loss = loss.item()
			log(writer, loss)
		
		print(f"{i}: {last_loss}")

if __name__ == '__main__':
	args = get_args()
	if not args.dataset_dir:
		print('Specify dataset dir with "--dataset_dir"')
		exit()
	if not args.n:
		print('Specify DOF with "--n"')
		exit()
		
	writer = SummaryWriter(log_dir=args.log_dir)
	model = SelfSupHNN(args.n, args.hidden_dim, args.nonlinearity)
	
	'''if torch.cuda.is_available():
		torch.device('cuda')
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
		model.cuda()'''
	
	epoch = 0
	for dataset in os.listdir(args.dataset_dir):
		print(f"Epoch {epoch}:")
		train(writer, model, f"{args.dataset_dir}/{dataset}", args.learn_rate, args.batch_size)
		epoch += 1
	writer.close()
	
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	torch.save(model.state_dict(), f"{args.save_dir}/{args.name}.tar")
