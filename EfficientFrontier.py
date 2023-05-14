import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from DataGeneration import generate_data

data_x, data_y = generate_data(100000)
data_x = torch.tensor(data_x, dtype=torch.float32)
data_y = torch.tensor(data_y, dtype=torch.float32)

# calculate sharpe ratio from tensors
def sharpe_ratio(y, mu, sigma):
    port_mu = torch.matmul(y.unsqueeze(1), mu.unsqueeze(-1)).view(-1)
    port_sigma = torch.matmul(torch.matmul(y.unsqueeze(1), sigma), y.unsqueeze(-1)).view(-1)
    return port_mu / port_sigma

# loss function is MAE between log sharpe ratios
def sharpe_loss(x, y, y_hat):
    mu = x[:,:5] # first 5 elements
    sigma = x[:,-25:] # last 25 elements
    sigma = sigma.view(-1,5,5) # reshape to 5x5
    return torch.mean(torch.abs(torch.log(sharpe_ratio(y, mu, sigma)) - torch.log(sharpe_ratio(y_hat, mu, sigma))))

class SharpeData(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y
    
SharpeDataset = SharpeData(data_x, data_y)
train_dataloader = DataLoader(SharpeDataset, batch_size=128, shuffle=True)

# init the model
model = nn.Sequential(nn.Linear(30, 512), nn.ReLU(), nn.Linear(512,256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64,5), nn.Softmax(dim=1))
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)

# train the model
for e in range(100):

    epoch_loss = 0

    for i, data in enumerate(train_dataloader):
        X, Y = data
        optimizer.zero_grad()
        Y_hat = model(X)
        loss = sharpe_loss(X, Y, Y_hat)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print("epoch", e, "loss:", np.mean(epoch_loss))