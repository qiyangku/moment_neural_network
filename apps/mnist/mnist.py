# -*- coding: utf-8 -*-
from Mnn_Core.mnn_modules import *
import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/mnist_task")


class Mnn_MLP_without_Corr(torch.nn.Module):
    def __init__(self):
        super(Mnn_MLP_without_Corr, self).__init__()
        self.layer1 = Mnn_Layer_without_Rho(784, 800)
        self.layer2 = Mnn_Linear_without_Corr(800, 10, bias=True)

    def forward(self, ubar, sbar):
        ubar, sbar = self.layer1(ubar, sbar)
        ubar, sbar = self.layer2(ubar, sbar)
        return ubar, sbar


class Model_Training:
    def __init__(self):
        self.file_path = "D:\Data_repos\MNIST"
        self.EPOCHS = 5
        self.batch_size_train = 600
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.random_seed = 1024
        self.log_interval = 10
        self.eps = 1e-8
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.fetch_dataset()

    def fetch_dataset(self):
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(self.file_path, train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.RandomRotation(10),
                                           torchvision.transforms.RandomCrop(28, padding=3),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.batch_size_train, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(self.file_path, train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.batch_size_test, shuffle=True)

    def training_mlp_model(self, fix_seed=True, save_op=True, save_name="mnist_mlp.pt"):
        if fix_seed:
            torch.manual_seed(self.random_seed)
        net = Mnn_MLP_without_Corr()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        net.train()
        criterion = torch.nn.CrossEntropyLoss()
        count = 0
        print("------ MNIST MLP TRAINING START ------")
        for epoch in range(self.EPOCHS):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                data = data.view(self.batch_size_train, -1)
                data = data.type(torch.float64)
                out1, out2 = net(data, torch.sqrt(torch.abs(data)))
                loss = criterion(out1, target)
                loss.backward()
                optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100 * batch_idx / len(self.train_loader), loss.item()))
                    writer.add_scalar("loss", loss.item(), count)
                    count += 1
        self.model = net
        if save_op:
            torch.save(net, save_name)

    def test_mlp_model(self, model_name="mnist_mlp.pt"):
        if self.model is None:
            net = torch.load(model_name)
        else:
            net = self.model
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.view(self.batch_size_test, -1)
                data = data.type(torch.float64)
                out1, out2 = net(data, torch.sqrt(torch.abs(data)))
                test_loss += F.cross_entropy(out1, target, reduction="sum").item()
                pred = out1.data.max(1, keepdim=True)[1]
                correct += torch.sum(pred.eq(target.data.view_as(pred)))
                test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))


if __name__ == "__main__":
    utils = Model_Training()
    utils.EPOCHS = 20
    utils.training_mlp_model()
    utils.test_mlp_model()
