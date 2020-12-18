# -*- coding: utf-8 -*-
from apps.cifar10.cifar10_train import *


def test_classic(model, testloader):
    print('------ Test Start -----')

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
    print('Accuracy of the network is: %.4f %%' % accuracy)
    return accuracy


def test_fc(model, testloader):
    print('------ Test Start -----')

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
    print('Accuracy of the network is: %.4f %%' % accuracy)
    return accuracy


EPOCHS = 20
BATCH_SIZE = 512

net = torch.load("mnn_fc.pt")
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(
    root='D:\Data_repos\Cifar10',
    train=False,
    download=False,
    transform=transform
)


testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=True)

# 类别标签
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

test_fc(net, testloader)