import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from train import GrogginessModel

def eval(test_loader, current_index, person, input_size, criterion = nn.MSELoss(), ):
  model = GrogginessModel(input_size, 5, 1)
  model.load_state_dict(torch.load(f'models/{person}-{current_index}.pt'))
  model.eval()
  test_loss = 0
  score = 0
  for inputs, labels in test_loader:
    assert len(inputs) == 1
    outputs = model(inputs)
    test_loss += criterion(outputs, labels).item() * inputs.size(0)
    score += torch.sum(torch.abs(outputs - labels)).item()
  
  print(f'Score: {score} @ Day {current_index}')
  
  return test_loss, score
