from argparse import ArgumentParser
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import load_dataset
from the_hnn import SelfSupHNN

def get_args():
	parser = ArgumentParser(description=None)
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--learn_rate', default=1e-3, type=float)
	parser.add_argument('--hidden_dim', default=520, type=int)
	parser.add_argument('--nonlinearity', default='tanh', type=str)
	parser.add_argument('--save_dir', default='path')
	parser.add_argument('--name', default='selfsup_hnn')
	return parser.parse_args()

def train(writer, dataset, learn_rate, hidden_dim, nonlinearity):
	dataset = load_dataset(dataset)
	model = SelfSupHNN(len(dataset[0][0]['x']) // 2, hidden_dim, nonlinearity)
	optim = torch.optim.Adam(model.parameters(), learn_rate, weight_decay=1e-4)
	stats = []
	
	dataset_size = len(dataset)
	data_size = len(dataset[0])
	
	for i in range(data_size - 1):
		last_loss = None
		for j in range(dataset_size):
			loss = model.loss(dataset[j][i], dataset[j][i + 1])
			optim.zero_grad()
			loss.backward()
			optim.step()
			last_loss = loss.item()
			writer.add_scalar('Loss/train', loss, i*dataset_size+j)
			stats.append(last_loss)
			
		if i % 100 == 0:
			print(f"{i}: {last_loss}")
	
	return model, stats

if __name__ == '__main__':
	args = get_args()
	if not args.dataset:
		print('Specify dataset with "--dataset"')
		exit()
	writer = SummaryWriter()
	model, stats = train(writer, args.dataset, args.learn_rate,
	                     args.hidden_dim, args.nonlinearity)
	writer.close()
	
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	
	torch.save(model.state_dict(), f"{args.save_dir}/{args.name}.tar")
