import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



def rename_columns(df):
  column_mappings = {
    'Name': 'name',
    'Date': 'date',
    'Grogginess': 'grogginess',
    'How many alarms did you set?': 'num_alarms',
    'What time did you set your first alarm for?': 'first_alarm',
    'Did you take sleep-aiding medicine (not weed, ie melatonin/antihystamine)?': 'sleep_medicine',
    "What was the temperature when you woke up? (Don't include °F, just the number)": 'waking_temp',
    'Were you intoxicated when you went to sleep?': 'intoxicated',
    'Were you sick when you went to sleep?': 'sick',
    'Did you eat within an hour of going to bed?': 'eat_before_bed',
    'Did you sleep alone?': 'sleep_alone',
    'Did you sleep in your own bed/room?': 'own_bed',
    'How stressed were you last night?': 'stress',
    'Did you use your phone before going to sleep?': 'phone_before_bed',
    "When was the latest you ingested caffeine before going to bed? (Don't answer if N/A)": 'caffeine_before_bed'
  }
  
  df_renamed = df.rename(columns=column_mappings)
  return df_renamed

  
def time_to_minutes(time_str):
  if pd.isna(time_str) or time_str == '':
    return 0
  time = pd.to_datetime(time_str, format='%I:%M:%S %p')
  return time.hour * 60 + time.minute

class GrogginessModel(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(GrogginessModel, self).__init__()
    self.layer1 = nn.Linear(input_size, hidden_size) 
    self.layer2 = nn.Linear(hidden_size, output_size)
    
  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = self.layer2(x)
    return x
  
def evaluate_model(model, test_loader, criterion, lambda1):
  model.eval()
  total_loss = 0.0
  with torch.no_grad():
    for inputs, targets in test_loader:
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      l1_norm = sum(p.abs().sum() for p in model.parameters())
      total_loss += loss + lambda1 * l1_norm
      
  avg_loss = total_loss / len(test_loader.dataset)
  return avg_loss

class EarlyStopping:
  def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf
    self.delta = delta
    self.path = path

  def __call__(self, val_loss, model):
    score = -val_loss

    if self.best_score is None:
      self.best_score = score
      self.save_checkpoint(val_loss, model)
    elif score < self.best_score + self.delta:
      self.counter += 1
      # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
      if self.counter >= self.patience:
          self.early_stop = True
    else:
      self.best_score = score
      self.save_checkpoint(val_loss, model)
      self.counter = 0

  def save_checkpoint(self, val_loss, model):
    if self.verbose:
      print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    torch.save(model.state_dict(), self.path)
    self.val_loss_min = val_loss

def main(person: str, 
         features: list[str] = ['sleepBPM', 'wakingBPM',
                                 'inBed_minutes', 'is_weekend', 'num_alarms',
                                   'phone_before_bed', 'caffeine_before_bed'],
         split_point: int = 15, 
         seed = 42, 
         lambda1 = 0.01, 
         criterion = nn.MSELoss(), 
         epochs=4000):
         
  data = pd.read_csv(f'{person}/autosleep-{person}.csv')
  data['date'] = pd.to_datetime(data['ISO8601'], errors='coerce')
  data['date'] = data['date'].apply(lambda x: x.tz_localize(None).normalize() if x is not pd.NaT else pd.NaT)
  data['year'] = data['date'].dt.year
  data['month'] = data['date'].dt.month
  data['day'] = data['date'].dt.day
  data['inBed_minutes'] = pd.to_timedelta(data['inBed']).dt.total_seconds() / 60
  data['day_of_week'] = data['date'].dt.dayofweek
  data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

  suppl = rename_columns(pd.read_csv('form.csv'))

  suppl['date'] = pd.to_datetime(suppl['date'], format='%m/%d/%Y').dt.normalize()



  suppl['first_alarm'] = suppl['first_alarm'].apply(time_to_minutes)
  suppl['caffeine_before_bed'] = suppl['caffeine_before_bed'].apply(time_to_minutes)
  lag_periods = 1
  suppl['grogginess_lag1'] = suppl['grogginess'].shift(lag_periods)
  suppl = suppl[suppl['name'] == person].fillna(0)

  merged_data = pd.merge(suppl, data, on='date', how='inner').fillna(0)

  to_normalize = ['sleepBPM', 'sleepBPMAvg7', 'dayBPM', 'dayBPMAvg7', 'wakingBPM', 'wakingBPMAvg7', 
            'hrv', 'hrvAvg7', 'sleepHRV', 'sleepHRVAvg7', 'SpO2Avg', 'SpO2Min', 'SpO2Max', 
            'respAvg', 'respMin', 'respMax', 'inBed_minutes', 'day_of_week', 'is_weekend', 
            'first_alarm', 'num_alarms', 'sleep_medicine', 'waking_temp', 'intoxicated', 
            'sick', 'eat_before_bed', 'sleep_alone', 'own_bed', 'stress', 'phone_before_bed', 
            'caffeine_before_bed']


  for cat in to_normalize:
    merged_data[cat] = merged_data[cat] / (merged_data[cat].max())


  merged_data.dropna(axis=1, inplace=True)

  target = ['grogginess']
  features = [f for f in features if f in merged_data.columns]

  X = merged_data[features]
  y = merged_data[target] 

  X_train, X_test = X[:split_point], X[split_point:split_point+1]
  y_train, y_test = y[:split_point], y[split_point:split_point+1]

  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # torch.cuda.set_device(device)

  X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
  y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
  X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
  y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
  batch_size = 32

  training_data = TensorDataset(X_train_tensor, y_train_tensor)
  test_data = TensorDataset(X_test_tensor, y_test_tensor)

  train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

  input_size = len(features) 
  hidden_size = 5  
  output_size = 1
  model = GrogginessModel(input_size, hidden_size, output_size)

  optimizer = optim.Adam(model.parameters(), lr=0.001) 

  early_stopping = EarlyStopping(patience=1000, verbose=False, path=f'models/{person}-{split_point}.pt')

  for epoch in range(epochs):
    model.train()
    val_loss = evaluate_model(model, test_loader, criterion, lambda1)
    early_stopping(val_loss, model)
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets) 
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        total_loss = loss + lambda1 * l1_norm
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item() * inputs.size(0) 

    epoch_loss = running_loss / len(train_loader.dataset)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

  return model, test_loader



# if __name__ == '__main__':
#   main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])