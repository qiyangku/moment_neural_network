# -*- coding: utf-8 -*-
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from Mnn_Core.mnn_pytorch import *
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/cifar10_experiment_1')


class Mnn_Classic(torch.nn.Module):
    """Some Information about Net"""

    def __init__(self):
        super(Mnn_Classic, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),  # 3*32*32 -> 16*32*32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 16*32*32 -> 16*16*16
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, padding=1),  # 16*16*16 -> 32*16*16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 32*16*16 -> 32*8*8
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, padding=1),  # 32*8*8 -> 64*8*8
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 64*8*8 -> 64*4*4
        )
        self.layer1 = Mnn_Layer_without_Rho(64 * 4 * 4, 64*4)
        self.layer2 = Mnn_Layer_without_Rho(64*4, 10)
        self.layer3 = Mnn_Linear_without_Corr(10, 10, bias=True)
        self.add_noise = 0.0
        self.mul_noise = 1.0

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64 * 4 * 4)
        y = torch.sqrt(torch.abs(x.clone()))*self.mul_noise + self.add_noise
        x, y = self.layer1(x, y)
        x, y = self.layer2(x, y)
        x, y = self.layer3(x, y)
        return x, y


class Mnn_FC(torch.nn.Module):
    def __init__(self):
        super(Mnn_FC, self).__init__()
        self.layer1 = Mnn_Layer_without_Rho(3*32*32, 32*32)
        self.layer2 = Mnn_Layer_without_Rho(32*32, 500)
        self.layer3 = Mnn_Layer_without_Rho(500, 500)
        self.layer4 = Mnn_Layer_without_Rho(500, 100)
        self.output = Mnn_Linear_without_Corr(100, 10, bias=True)

    def forward(self, ubar, sbar):
        ubar, sbar = self.layer1(ubar, sbar)
        ubar, sbar = self.layer2(ubar, sbar)
        ubar, sbar = self.layer3(ubar, sbar)
        ubar, sbar = self.layer4(ubar, sbar)
        ubar, sbar = self.output(ubar, sbar)
        return ubar, sbar


class Tradition_Net(torch.nn.Module):
    def __init__(self):
        super(Tradition_Net, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1,padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512*4*4,1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024,1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = x.view(-1, 512*4*4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x


class Mnn_Auto_Encoder(nn.Module):
    def __init__(self):
        super(Mnn_Auto_Encoder, self).__init__()
        self.encoder_layer1 = Mnn_Layer_without_Rho(3*32*32, 32*32)
        self.encoder_layer2 = Mnn_Layer_without_Rho(32*32, 500)
        self.encoder_layer3 = Mnn_Layer_without_Rho(500, 100)

        self.decoder_layer1 = Mnn_Layer_without_Rho(100, 500)
        self.decoder_layer2 = Mnn_Layer_without_Rho(500, 32*32)
        self.decoder_layer3 = nn.Sequential(nn.Linear(32*32, 3*32*32),
                                            nn.Sigmoid())

    def encoder(self, ubar, sbar):
        ubar, sbar = self.encoder_layer1(ubar, sbar)
        ubar, sbar = self.encoder_layer2(ubar, sbar)
        ubar, sbar = self.encoder_layer3(ubar, sbar)
        return ubar, sbar

    def decoder(self, ubar, sbar):
        ubar, sbar = self.decoder_layer1(ubar, sbar)
        ubar, sbar = self.decoder_layer2(ubar, sbar)
        out = self.decoder_layer3(ubar)
        return out

    def forward(self, ubar, sbar):
        batch = ubar.size()[0]
        ubar, sbar = self.encoder(ubar, sbar)
        out = self.decoder(ubar, sbar)
        out = out.view(batch, 3, 32, 32)
        return out


class Train_Cifar10_Model:
    def __init__(self):
        self.BATCH = 512
        self.data_path = 'D:\Data_repos\Cifar10'
        self.EPOCHS = 50
        self.LR = 0.1
        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def train_mnn_classic(self, model, criterion, optimizer, trainloader, model_name, epochs=10, log_interval=50):
        print('----- Train Start -----')
        count = 0
        for epoch in range(epochs):
            running_loss = 0.0
            for step, (batch_x, batch_y) in enumerate(trainloader):
                batch_x = batch_x.type(torch.float64)

                output = model(batch_x)[0]

                optimizer.zero_grad()
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if step % log_interval == (log_interval - 1):
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, step + 1, running_loss / log_interval))
                    writer.add_scalar("loss/"+model_name, running_loss / log_interval, count)
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    _, predict = torch.max(output.data, 1)
                    total += batch_y.size(0)
                    correct += (predict == batch_y).sum().item()
                    writer.add_scalar("accuracy/"+model_name, 100.0 * correct / total, count)
                    count += 1
                    print('Accuracy of the network on the %d tran images: %.3f %%' % (total, 100.0 * correct / total))
        print('----- Train Finished -----')

    def train_mnn_fc(self, model, criterion, optimizer, trainloader, model_name, epochs=10, log_interval=50):
        print('----- Train Start -----')
        count = 0
        for epoch in range(epochs):
            running_loss = 0.0
            for step, (batch_x, batch_y) in enumerate(trainloader):
                batch_x = batch_x.type(torch.float64)
                batch_x = batch_x.view(-1, 3*32*32)
                output = model(batch_x, torch.sqrt(torch.abs(batch_x.clone())))[0]
                optimizer.zero_grad()
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if step % log_interval == (log_interval - 1):
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, step + 1, running_loss / log_interval))
                    writer.add_scalar("loss/"+model_name, running_loss / log_interval, count)
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    _, predict = torch.max(output.data, 1)
                    total += batch_y.size(0)
                    correct += (predict == batch_y).sum().item()
                    writer.add_scalar("accuracy/"+model_name, 100.0 * correct / total, count)
                    count += 1
                    print('Accuracy of the network on the %d tran images: %.3f %%' % (total, 100.0 * correct / total))
        print('----- Train Finished -----')

    def train_mnn_ae(self, model, criterion, optimizer, trainloader, model_name, epochs=10, log_interval=50):
        print('----- Train Start -----')
        count = 0
        for epoch in range(epochs):
            running_loss = 0.0
            for step, (batch_x, batch_y) in enumerate(trainloader):
                batch_x = batch_x.type(torch.float64)
                flatten_x = batch_x.view(-1, 3*32*32)
                output = model(flatten_x, torch.sqrt(torch.abs(flatten_x.clone())))
                optimizer.zero_grad()
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if step % log_interval == (log_interval - 1):
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, step + 1, running_loss / log_interval))
                    writer.add_scalar("loss/" + model_name, running_loss / log_interval, count)
                    count += 1
                    running_loss = 0.0
        print('----- Train Finished -----')

    def select_n(self, data, labels, start=10):
        '''
        Selects n random datapoints and their corresponding labels from a dataset
        '''
        assert len(data) == len(labels)
        return data[start: start + 10], labels[start: start+10]

    def test_classic(self, model, testloader, model_name="Mnn_classic"):
        print('------ Test {} Start -----'.format(model_name))

        correct = 0
        total = 0

        with torch.no_grad():
            for test_x, test_y in testloader:
                images, labels = test_x, test_y
                images = images.type(torch.float64)
                output = model(images)[0]
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print("total samples: {:}, num of correct: {:}".format(total, correct))
        print('Accuracy of the network is: %.4f %%' % accuracy)
        return accuracy

    def test_fc(self, model, testloader, model_name="Mnn_Fc"):
        print('------ Test {:} Start -----'.format(model_name))
        correct = 0
        total = 0
        with torch.no_grad():
            for test_x, test_y in testloader:
                images, labels = test_x, test_y
                images = images.type(torch.float64)
                images = images.view(-1, 3*32*32)
                sbar = torch.sqrt(torch.abs(images.clone()))
                output = model(images, sbar)[0]
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print("total samples: {:}, num of correct: {:}".format(total, correct))
        print('Accuracy of the network is: %.4f %%' % accuracy)
        return accuracy

    def train_model(self):
        EPOCHS = self.EPOCHS
        BATCH_SIZE = self.BATCH
        data_path = self.data_path
        LR = self.LR

        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            download=False,
            transform=transform_train
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=BATCH_SIZE, shuffle=True)

        net = Mnn_Classic()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        self.train_mnn_classic(net, criterion, optimizer, trainloader, epochs=EPOCHS, model_name="Mnn_Classic")
        torch.save(net, "mnn_classic.pt")

        net = Mnn_FC()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        self.train_mnn_fc(net, criterion, optimizer, trainloader, epochs=EPOCHS, model_name="Mnn_FC")
        torch.save(net, "mnn_fc.pt")

        net = Mnn_Auto_Encoder()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        self.train_mnn_ae(net, criterion, optimizer, trainloader, epochs=EPOCHS, model_name="Mnn_AE")
        torch.save(net, "mnn_ae.pt")

    def test_model(self):
        BATCH_SIZE = self.BATCH
        data_path = self.data_path
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        testset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            download=False,
            transform=transform
        )

        trainset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            download=False,
            transform=transform
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=BATCH_SIZE, shuffle=True)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=BATCH_SIZE, shuffle=True)

        # 类别标签

        net = torch.load("mnn_classic.pt")
        net.eval()
        self.test_classic(net, trainloader)
        self.test_classic(net, testloader)

        net = torch.load("mnn_fc.pt")
        net.eval()
        self.test_fc(net, trainloader)
        self.test_fc(net, testloader)


if __name__ == "__main__":
    test = Train_Cifar10_Model()
    test.test_model()

