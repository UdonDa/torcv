import torch
import torch.nn as nn


class Ciffar10_Solver(object):
    """Cifar10 solver model
    Args:
        config (argparse): Some settings. please conclude gpu_number('0'...), optimizer('sgd'), lr, momentum, total_epochs.
        train_loader (DataLoader): Train dataloader.
        test_loader (DataLoader): Test dataloader.
        model (nn.Module): Model.
    """
    def __init__(self, config, train_loader=None, test_loader=None, val_loader=None, model=None):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        
        self.device = torch.device('cuda:{}'.format(self.config.gpu_number) if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()

        self.build_model(model)

    def build_model(self, model):
        self.model = model.to(self.device)

        if self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=self.config.momentum)

    def train(self):
        for epoch in range(self.config.total_epochs):
            for i, (x, label) in enumerate(self.train_loader):
                x, label = x.to(self.device), label.to(self.device)

                self.optimizer.zero_grad()

                y = self.model(x)
                loss = self.criterion(y, label)
                loss.backward()
                self.optimizer.step()

                if (i+1) % 10 == 0:
                    print('Epoch: [{}], Iter: [{}/{}] loss: {:.4f}'.format(epoch+1, i+1, len(self.train_loader), loss.item()))
            
            with torch.no_grad():
                correct = 0
                total = 0
                for x, label in self.test_loader:
                    x, label = x.to(self.device), label.to(self.device)
                    y = self.model(x)
                    _, predicted = torch.max(y.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
            print("\nAccuracy: {} %\n".format(100 * correct / total))