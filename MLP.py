import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from opart_functions import tune_lldas, SquaredHingeLoss
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(123)
np.random.seed(123)

# Define the MLP model
class MLPModel(nn.Module):

    def __init__(self, input_size, hidden_layers, hidden_size):
        super(MLPModel, self).__init__()
        self.input_size    = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size   = hidden_size

        if(self.hidden_layers == 0):
            self.linear_model = nn.Linear(input_size, 1)                                                        # Define linear model
        else:
            self.input_layer = nn.Linear(input_size, hidden_size)                                               # Define input layer
            self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers-1)])  # Define hidden layers
            self.output_layer = nn.Linear(hidden_size, 1)                                                       # Define output layer
        
    def forward(self, x):
        if(self.hidden_layers == 0):
            return self.linear_model(x)
        else:
            x = torch.relu(self.input_layer(x))
            for layer in self.hidden:
                x = torch.relu(layer(x))
            x = self.output_layer(x)
            return x



# cross validation to learn the number of iterations
# NEED TO REVIEW THIS PART
def cv_learn(n_splits, X, y, n_hiddens, layer_size, batch_size, n_ite):
    
    # Define the K-fold cross-validation
    kf = KFold(n_splits, shuffle=True, random_state=123)

    # loss function
    loss_func = SquaredHingeLoss()

    # learn best ite
    total_losses = {'train': np.zeros(n_ite), 'val': np.zeros(n_ite)}
    data_splits = {'X_train': [], 'X_val': [], 'y_train': [], 'y_val': []}
    
    for subtrain_idx, val_idx in kf.split(X):

        # Split the data into training and validation sets
        data_splits['X_train'].append(X[subtrain_idx])
        data_splits['X_val'].append(X[val_idx])
        data_splits['y_train'].append(y[subtrain_idx])
        data_splits['y_val'].append(y[val_idx])

        # Create DataLoader
        dataset    = TensorDataset(data_splits['X_train'], data_splits['y_train'])
        dataloader = DataLoader(dataset, batch_size, shuffle=True)

        # Define your model
        model = MLPModel(X.shape[1], n_hiddens, layer_size)

        # define optimizer
        optimizer = optim.Adam(model.parameters())

        # Training loop for the specified number of iterations
        for i in range(n_ite):
            # training
            train_loss = 0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                loss = loss_func(model(inputs), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # validating
            model.eval()
            with torch.no_grad():
                val_loss = loss_func(model(data_splits['X_val']), data_splits['y_val'])

            # add train_loss and val_loss into arrays
            total_losses['train'][i] += train_loss/len(dataloader)
            total_losses['val'][i] += val_loss.item()

    best_no_ite = np.argmin(total_losses['val'])
    return best_no_ite + 1



# learn lldas
def mlp(features, targets, hidden_layers, hidden_size, batch_size, n_ite):

    # prepare training dataset
    dataset    = TensorDataset(features, targets)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    # Instantiate model, loss function and opimizer
    model = MLPModel(features.shape[1], hidden_layers, hidden_size)
    criterion = SquaredHingeLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for _ in range(n_ite + 1):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    with torch.no_grad():
        lldas = model(features).numpy().reshape(-1)

    return tune_lldas(lldas)
