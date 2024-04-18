import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

def read_autosleep_data(person: str) -> pd.DataFrame:
  data = pd.read_csv(f'{person}/autosleep-{person}.csv')
  data['date'] = pd.to_datetime(data['ISO8601'], errors='coerce')
  data['date'] = data['date'].apply(lambda x: x.tz_localize(None).normalize() if x is not pd.NaT else pd.NaT)
  data['year'] = data['date'].dt.year
  data['month'] = data['date'].dt.month
  data['day'] = data['date'].dt.day
  data['inBed_minutes'] = pd.to_timedelta(data['inBed']).dt.total_seconds() / 60
  data['day_of_week'] = data['date'].dt.dayofweek
  data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
  data['date'][len(data) - 10:]
  return data

def read_form_data(person: str) -> pd.DataFrame:
  suppl = rename_columns(pd.read_csv('form.csv'))

  suppl['date'] = pd.to_datetime(suppl['date'], format='%m/%d/%Y').dt.normalize()


  def time_to_minutes(time_str):
    if pd.isna(time_str) or time_str == '':
        return 0
    time = pd.to_datetime(time_str, format='%I:%M:%S %p')
    return time.hour * 60 + time.minute

  suppl['first_alarm'] = suppl['first_alarm'].apply(time_to_minutes)
  suppl['caffeine_before_bed'] = suppl['caffeine_before_bed'].apply(time_to_minutes)
  suppl['grogginess_lag1'] = suppl['grogginess'].shift(1)
  suppl['grogginess_lag2'] = suppl['grogginess'].shift(2)
  suppl['grogginess_lag3'] = suppl['grogginess'].shift(3)
  suppl = suppl[suppl['name'] == person].fillna(0)
  return suppl

features = ['sleepBPM',
            'sleepBPMAvg7', 'dayBPM', 'dayBPMAvg7', 'wakingBPM', 'wakingBPMAvg7', 
            'hrv', 'hrvAvg7', 'sleepHRV', 'sleepHRVAvg7', 'SpO2Avg', 'SpO2Min', 'SpO2Max', 
            # 'respAvg', 'respMin', 'respMax', 
            #'year',
             'month', 
            #'day', 
            'inBed_minutes', 'day_of_week', 'is_weekend', ]
features += ['first_alarm', 'num_alarms', 'sleep_medicine', 'waking_temp', 'intoxicated', 'sick', 'eat_before_bed', 'sleep_alone', 'own_bed', 'stress', 'phone_before_bed', 'caffeine_before_bed', 'grogginess_lag1']

target = ['grogginess']


class GrogginessModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GrogginessModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size) 
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = x * 9 + 1
        return x
    

def evaluate_model(model, test_loader, criterion):
  model.eval()  # Set the model to evaluation mode
  total_loss = 0.0
  with torch.no_grad():  # No need to track gradients
    for inputs, targets in test_loader:
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      total_loss += loss.item() * inputs.size(0)
  avg_loss = total_loss / len(test_loader.dataset)
  return avg_loss

def make_prediction(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())  # Assuming you're using CPU for simplicity
    return predictions[-1]  # Return the last prediction if there are multiple

  
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
    # if self.verbose:
      # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    torch.save(model.state_dict(), self.path)
    self.val_loss_min = val_loss
 

def train_model(model, train_loader, val_loader, epochs=2000, early_stopping=EarlyStopping(patience=1000, verbose=True)):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    val_loss = evaluate_model(model, val_loader, criterion)

    for inputs, targets in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()
      running_loss += loss.item() * inputs.size(0)
    avg_loss = running_loss / len(train_loader.dataset)
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
      # print("Early stopping")
      break
    # if epoch % 100 == 0:
      # print(f'Epoch {epoch + 1}/{epochs} train loss: {avg_loss}, val loss: {val_loss}')
  return model