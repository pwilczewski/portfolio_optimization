import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from DataGeneration import generate_data, SharpeData
import matplotlib.pyplot as plt

regen_data = False

if regen_data == True:
    data_x, data_y = generate_data(4000000)
else:
    data_x = pd.read_parquet("data_x1.parquet").iloc[:,1:]
    data_y = pd.read_parquet("data_y1.parquet").iloc[:,1:]

data_x = torch.tensor(np.array(data_x), dtype=torch.float32)
data_y = torch.tensor(np.array(data_y), dtype=torch.float32)

def portfolio_mu(y, mu):
    return torch.matmul(y.unsqueeze(1), mu.unsqueeze(-1)).view(-1)

def portfolio_sigma(y, sigma):
    return torch.matmul(torch.matmul(y.unsqueeze(1), sigma), y.unsqueeze(-1)).view(-1)

def sharpe_ratio(y, mu, sigma):
    return portfolio_mu(y, mu) / torch.sqrt(portfolio_sigma(y, sigma))

def simple_port_loss(x, y, y_hat):
    mu = x[:,:5]
    sigma = x[:,-25:].view(-1,5,5)
    mu_loss = torch.mean(torch.abs(portfolio_mu(y, mu) - portfolio_mu(y_hat, mu)))
    sigma_loss = torch.mean(torch.abs(portfolio_sigma(y, sigma) - portfolio_sigma(y_hat, sigma)))
    return mu_loss + sigma_loss
    
SharpeDataset = SharpeData(data_x, data_y)
train_dataloader = DataLoader(SharpeDataset, batch_size=2**10, shuffle=True)

device = torch.device('cuda:0')
torch.cuda.set_device(device)

model = nn.Sequential(nn.Linear(30, 1024), nn.ReLU(),
                      nn.Linear(1024, 256), nn.ReLU(),
                      nn.Linear(256, 128), nn.ReLU(),
                      nn.Linear(128, 64), nn.ReLU(),
                      nn.Linear(64, 5), nn.Softmax(dim=1))
optimizer = torch.optim.AdamW(model.parameters(),1e-3)
model = model.to(device)
print("# of parameters:", sum(param.numel() for param in model.parameters()))

for e in range(75):

    epoch_loss = 0

    for i, data in enumerate(train_dataloader):
        X, Y = data
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        Y_hat = model(X)
        loss = simple_port_loss(X, Y, Y_hat)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print("epoch", e, "loss:", np.mean(epoch_loss))

torch.save(model.state_dict(), "model_4m.pth")
