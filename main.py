import pandas as pd 
import torch
import torch.nn as nn

from config import config 
from data_preprocessing import B3DBDataset, create_loader
from main_loops import main_loop
from model import B3DBModel

data = pd.read_csv('B3DB_classification.tsv', sep='\t')
print(data.head())

dataset = B3DBDataset(data)
train_loader, val_loader, test_loader = create_loader(dataset)

num_features = dataset[0].x.shape[1]
batch_size = config['batch_size']
device     = config['device']
hidden_dim = config['hidden_dim']
output_dim = config['output_dim']
lr         = config['lr']
num_epochs = config['num_epochs']

model = B3DBModel(num_features, hidden_dim, output_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCEWithLogitsLoss()

main_loop(model, optimizer, loss_fn, train_loader, val_loader, test_loader, device, num_epochs)