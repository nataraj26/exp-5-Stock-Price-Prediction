# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
#### Problem Statement

The aim of this experiment is to develop a Recurrent Neural Network (RNN) model to learn sequential patterns from time-series data and make accurate future predictions.

#### Dataset

A time-series dataset is used, where each sample represents sequential data points (such as stock prices, temperature readings, or sensor signals).
The data is preprocessed and divided into training and testing sets to train and evaluate the RNN model effectively.


## Design Steps

### Step 1: 
Import necessary libraries and load the dataset.
### Step 2: 
Preprocess the data and create training and testing sets.
### Step 3: 
Define the RNN model architecture.
### Step 4: 
Initialize loss function, optimizer, and training parameters.
### Step 5: 
Train the model using the training data.
### Step 6: 
Evaluate the model performance on test data.
### Step 7: 
Visualize and analyze the training loss.



## Program
#### Name: NATARAJ KUMARAN S
#### Register Number:212223230137
```Python 
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc= nn.Linear(hidden_size, output_size)
  def forward(self, x):
    out, _ = self.rnn(x) #outshape:batch_size, seq_length, hidden_size
    out = self.fc(out[:, -1, :]) #take the output of the last step
    return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train the Model
epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}')

```

## Output

### True Stock Price, Predicted Stock Price vs time

<img width="882" height="566" alt="image" src="https://github.com/user-attachments/assets/90423c6c-254f-46b6-9f02-515ed1826a03" />


### Predictions 

<img width="920" height="46" alt="image" src="https://github.com/user-attachments/assets/076c76c7-0bb5-4b63-bac1-85408be75c54" />


## Result
The Recurrent Neural Network (RNN) model was successfully trained using the given time-series dataset.
The training loss gradually decreased over the epochs, showing that the model effectively learned the sequential patterns in the data and produced accurate predictions for future values.


