import pandas as pd
import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt

df = pd.read_csv("AirQualityUCI_cleaned.csv", sep=";")

third_column = df.iloc[:, 2].values
third_column_modified = [float(value.replace(',', '.')) for value in third_column]

n = 10

x_train, y_train = [], []

for i in range(len(third_column_modified) - n):
    x_train.append(third_column_modified[i : i + n])
    y_train.append(third_column_modified[i + n])

x_train = torch.tensor(np.array(x_train), dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(-1)

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

class MotionDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


train_dataset = MotionDataset(x_train2, y_train2)
test_dataset = MotionDataset(x_test2, y_test2)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1])
        return predictions


input_size = 1
hidden_layer_size = 64
output_size = 1
learning_rate = 5e-4

model = LSTMModel(input_size, hidden_layer_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 300
loss_history = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}')

plt.plot(range(epochs), loss_history, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Function During Training")
plt.legend()
plt.grid()
plt.show()

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model.eval()
y_pred_list = []
y_true_list = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        y_pred = model(batch_x)
        y_pred_list.append(y_pred.item())
        y_true_list.append(batch_y.item())

y_pred_array = np.array(y_pred_list)
y_true_array = np.array(y_true_list)

r2 = r2_score(y_true_array, y_pred_array)
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_true_array, y_pred_array, color='blue', label="Predicted vs. True")
plt.plot(y_true_array, y_true_array, color='red', linestyle="--", label="Ideal Fit (y=x)")
plt.xlabel("Ground Truth (y_test)")
plt.ylabel("Predicted Values (y_pred)")
plt.title(f"Predicted vs. Ground Truth on Test Set\nR² Score: {r2:.4f}")
plt.legend()
plt.grid()
plt.show()
