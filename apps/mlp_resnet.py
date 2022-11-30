import enum
import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION    

    fn = nn.Sequential(
        nn.Linear(dim,hidden_dim), 
        norm(hidden_dim),
        nn.ReLU(), 
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim))

    return nn.Sequential(nn.Residual(fn), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    module_list = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        module_list.append(ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob))
    module_list.append(nn.Linear(hidden_dim, num_classes))
    
    return nn.Sequential(*module_list)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    running_loss = 0.
    num_sample = 0.
    error = 0.

    loss_func = nn.SoftmaxLoss()

    print()

    if opt:
        model.train()
        for step, batch in enumerate(dataloader):
            img, label = batch[0].reshape((batch[0].shape[0], -1)), batch[1]
            out = model(img)
            loss = loss_func(out, label)

            opt.reset_grad()
            loss.backward()
            opt.step()

            running_loss += loss.numpy().item()
            error += (out.numpy().argmax(axis=1) != label.numpy()).sum()
            num_sample += label.shape[0]

            rate = (step + 1) / len(dataloader)
            a = "*" * int(rate * 50)
            b = "." * int((1-rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}  error:{}".format(int(rate*100), a, b, loss.numpy().item(), error), end="")
        print()
    else:
        model.eval()
        for step, batch in enumerate(dataloader):
            img, label = batch[0].reshape((batch[0].shape[0], -1)), batch[1]
            out = model(img)
            loss = loss_func(out, label)

            running_loss += loss.numpy().item()
            error += (out.numpy().argmax(axis=1) != label.numpy()).sum()
            num_sample += label.shape[0]

            rate = (step + 1) / len(dataloader)
            a = "*" * int(rate * 50)
            b = "." * int((1-rate) * 50)
            print("\reval loss: {:^3.0f}%[{}->{}]{:.3f}  error:{}".format(int(rate*100), a, b, loss.numpy().item(), error), end="")
        print()
        
    accuracy = error / num_sample
    running_loss /= (step+1)


    return accuracy, running_loss

    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    mnist_train_dataset = ndl.data.MNISTDataset(data_dir+"/train-images-idx3-ubyte.gz",
                                                data_dir+"/train-labels-idx1-ubyte.gz")
    mnist_train_dataloader = ndl.data.DataLoader(dataset=mnist_train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)
    mnist_test_dataset = ndl.data.MNISTDataset(data_dir+"/t10k-images-idx3-ubyte.gz",
                                                data_dir+"/t10k-labels-idx1-ubyte.gz")
    mnist_test_dataloader = ndl.data.DataLoader(dataset=mnist_test_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_error = train_loss = test_error = test_loss = 0.

    for _ in range(epochs):
        train_error, train_loss = epoch(mnist_train_dataloader, model, opt=opt)

    test_error, test_loss = epoch(mnist_test_dataloader, model)

    return (train_error, train_loss, test_error, test_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
