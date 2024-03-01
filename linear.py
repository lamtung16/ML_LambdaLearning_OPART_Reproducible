import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from opart_functions import tune_lldas, SquaredHingeLoss
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(123)
np.random.seed(123)

# Define the linear model
class LinearModel(nn.Module):

    def __init__(self, input_size=1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    def forward(self, x):
        return self.linear(x)



# learn lldas
def linear(feature, targets, n_ites=300):

    # prepare training dataset
    dataset    = TensorDataset(feature, targets)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Instantiate model, loss function and opimizer
    model = LinearModel()
    criterion = SquaredHingeLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for _ in range(n_ites):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    with torch.no_grad():
        lldas = model(feature).numpy().reshape(-1)

    return tune_lldas(lldas)