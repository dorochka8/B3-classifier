import torch

config = {
  'batch_size' : 32, 
  'device'     : 'cuda' if torch.cuda.is_available else 'cpu',
  'hidden_dim' : 32,
  'output_dim' : 1,
  'lr'         : 0.001,
  'num_epochs' : 50,
}