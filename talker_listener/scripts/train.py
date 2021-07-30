#!/usr/bin/env python
##based on example implementation given in task 3
from __future__ import print_function

from os import mkdir

import torch as t

from models.Net import Net

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cpu" #this ros is setup without gpu support

def get_train_loader(batch_size, kwargs):
    return DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)


def get_test_loader(batch_size, kwargs):
    return DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)


def train(train_loader, model, optimizer, criterion, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


def test(test_loader, model, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with t.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main():
    # training parameters
    batch_size = 200
    epochs = 1
    learning_rate = 0.01
    log_interval = 10
    save_model = True

    # create neural network models
    model = Net(input_shape=[1, 28, 28]).to(device)

    # Model 1: optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)
    # Model 2: optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    # Model 3: optimizer = t.optim.Adadelta(model.parameters(), lr=learning_rate)
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    # create a loss function
    criterion = t.nn.NLLLoss()

    # get training and test data sets
    train_loader = get_train_loader(batch_size=batch_size, kwargs={})
    test_loader = get_test_loader(batch_size=batch_size, kwargs={})

    # train and test for x epochs
    for epoch in range(1, epochs + 1):
        train(train_loader, model, optimizer, criterion, epoch, log_interval)
        test(test_loader, model, criterion)

    if save_model:
        try:
            mkdir("trained_models")
        except FileExistsError:
            pass
        t.save( model.state_dict(), 'trained_models/mnist_simple_fc.pth')


if __name__ == '__main__':
    main()
