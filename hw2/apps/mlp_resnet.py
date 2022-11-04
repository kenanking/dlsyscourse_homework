import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTIONS
    fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    
    return nn.Sequential(
        nn.Residual(fn),
        nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_err = 0.0
    avg_loss = 0.0
    data_size = len(dataloader.dataset)
    
    if opt is None:
        model.eval()
    else:
        model.train()
    
    loss_func = nn.SoftmaxLoss()
    for i, (x, y) in enumerate(dataloader):
        if opt is not None:
            opt.reset_grad()

        B = x.shape[0]
        x = x.reshape((B, -1))
        y_hat = model(x)
        loss = loss_func(y_hat, y)

        if opt is not None:        
            loss.backward()
            opt.step()
            
        avg_err += (y_hat.numpy().argmax(axis=1) != y.numpy()).sum()
        avg_loss += loss.numpy() * B
        
        # print("Step %d: loss=%.4f" % (i, loss.numpy()))
        
    avg_err /= data_size
    avg_loss /= data_size
    
    return avg_err, avg_loss
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"), 
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), 
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = MLPResNet(28*28, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch_idx in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
        test_err, test_loss = epoch(test_dataloader, model)
        # print("Epoch %d: train_err=%.4f, train_loss=%.4f, test_err=%.4f, test_loss=%.4f" % (
        #     epoch_idx, train_err, train_loss, test_err, test_loss))
    
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION

if __name__ == "__main__":
    train_mnist(data_dir="../data")
