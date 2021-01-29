# -*- coding: utf-8 -*-
import os
import pickle
from Mnn_Core.mnn_modules import *


class Event():
    '''
    This class provides a way to store, read, write and visualize spike event.

    Members:
        * ``x`` (numpy ``int`` array): `x` index of spike event.
        * ``y`` (numpy ``int`` array): `y` index of spike event (not used if the spatial dimension is 1).
        * ``p`` (numpy ``int`` array): `polarity` or `channel` index of spike event.
        * ``t`` (numpy ``double`` array): `timestamp` of spike event. Time is assumend to be in ms.

    Usage:

    >>> TD = Event(xEvent, yEvent, pEvent, tEvent)
    '''

    def __init__(self, xEvent, yEvent, pEvent, tEvent):
        if yEvent is None:
            self.dim = 1
        else:
            self.dim = 2

        self.x = xEvent if type(xEvent) is np.ndarray else np.asarray(xEvent)  # x spatial dimension
        self.y = yEvent if type(yEvent) is np.ndarray else np.asarray(yEvent)  # y spatial dimension
        self.p = pEvent if type(pEvent) is np.ndarray else np.asarray(pEvent)  # spike polarity
        self.t = tEvent if type(tEvent) is np.ndarray else np.asarray(tEvent)  # time stamp in ms

        if not issubclass(self.x.dtype.type, np.integer): self.x = self.x.astype('int')
        if not issubclass(self.p.dtype.type, np.integer): self.p = self.p.astype('int')

        if self.dim == 2:
            if not issubclass(self.y.dtype.type, np.integer): self.y = self.y.astype('int')

        self.p -= self.p.min()


def read2Dspikes(file_path):
    '''
    Reads two dimensional binary spike file and returns a TD event.
    It is the same format used in neuromorphic datasets NMNIST & NCALTECH101.

    The binary file is encoded as follows:
        * Each spike event is represented by a 40 bit number.
        * First 8 bits (bits 39-32) represent the xID of the neuron.
        * Next 8 bits (bits 31-24) represent the yID of the neuron.
        * Bit 23 represents the sign of spike event: 0=>OFF event, 1=>ON event.
        * The last 23 bits (bits 22-0) represent the spike event timestamp in microseconds.

    Arguments:
        * ``filename`` (``string``): path to the binary file.

    Usage:

    >>> TD = read2Dspikes(file_path)
    '''
    with open(file_path, 'rb') as inputFile:
        inputByteArray = inputFile.read()
    inputAsInt = np.asarray([x for x in inputByteArray])
    xEvent = inputAsInt[0::5]
    yEvent = inputAsInt[1::5]
    pEvent = inputAsInt[2::5] >> 7
    tEvent = ((inputAsInt[2::5] << 16) | (inputAsInt[3::5] << 8) | (inputAsInt[4::5])) & 0x7FFFFF
    return Event(xEvent, yEvent, pEvent, tEvent / 1000)  # convert spike times to ms


def raw2Tensor(TD: Event, frameRate=60):
    if TD.dim != 2:
        raise Exception('Expected Td dimension to be 2. It was: {}'.format(TD.dim))
    interval = 1e3 / frameRate  # in ms
    xDim = TD.x.max() + 1
    yDim = TD.y.max() + 1
    if xDim != 34 or yDim != 34 or xDim != yDim:
        print(xDim, yDim)
        raise ValueError

    minFrame = int(np.floor(TD.t.min() / interval))
    maxFrame = int(np.ceil(TD.t.max() / interval))
    samples = maxFrame - minFrame
    raw_data = torch.zeros(samples, yDim, xDim)
    for i in range(samples):
        tStart = (i + minFrame) * interval
        tEnd = (i + minFrame + 1) * interval
        timeMask = (TD.t >= tStart) & (TD.t < tEnd)
        positive = (timeMask & (TD.p == 1))
        negative = (timeMask & (TD.p == 0))
        raw_data[i, TD.y[positive], TD.x[positive]] = 1
        raw_data[i, TD.y[negative], TD.x[negative]] = -1

    x = raw_data.view(samples, 1, -1)
    y = torch.transpose(x.clone().detach(), dim0=-2, dim1=-1)

    x_mean = torch.mean(x, dim=0)
    x_std = torch.std(x, dim=0)

    y_mean = torch.mean(y, dim=0)

    vx = x - x_mean
    vy = y - y_mean

    correlation = torch.sum(vx * vy, dim=0) / torch.sqrt(torch.sum(vx ** 2, dim=0) * torch.sum(vy ** 2, dim=0))
    temp = torch.zeros_like(correlation)
    correlation = torch.where(torch.isnan(correlation), temp, correlation)
    correlation = correlation.fill_diagonal_(1.0)

    return x_mean.view(-1), x_std.view(-1), correlation


class nmistDataset(torch.utils.data.Dataset):
    def __init__(self, datasetPath="D:/Data_repos/N-MNIST/data/", mode: str = "train", frames=120):
        super(nmistDataset, self).__init__()
        self.path = datasetPath
        self.mode = mode
        self.file_path = None
        self.labels = None
        self.frames = frames
        self._fetch_files_path()

    def _fetch_files_path(self):
        data_dir = self.path + self.mode + "/"
        files_name = []
        labels = []
        for i in os.listdir(data_dir):
            next_dir = data_dir + i
            for j in os.listdir(next_dir):
                labels.append(i)
                files_name.append(data_dir + i + "/" + j)
        self.file_path = files_name
        self.labels = labels

    def remove_outlier(self):
        totol_sample = len(self.file_path)
        remove = 0
        correct_file = []
        correct_label = []
        for item in range(len(self.file_path)):
            file = self.file_path[item]
            td = read2Dspikes(file)
            xDim = td.x.max() + 1
            yDim = td.y.max() + 1
            if xDim != 34 or yDim != 34 or xDim != yDim:
                print(file)
                remove += 1
                if os.path.exists(file):
                    os.remove(file)
                else:
                    print("The file {:} does not exist".format(file))
            else:
                correct_file.append(file)
                correct_label.append(self.labels[item])
        self.file_path = correct_file
        self.labels = correct_label
        print("For {:} dataset, before check:{:}, remove: {:}".format(self.mode, totol_sample, remove))

    def __getitem__(self, item):
        input_sample = self.file_path[item]
        class_label = eval(self.labels[item])

        u, s, r = raw2Tensor(read2Dspikes(input_sample), frameRate=self.frames)

        return u, s, r, class_label

    def __len__(self):
        return len(self.file_path)


class Mnn_MLP_with_Corr(torch.nn.Module):
    def __init__(self):
        super(Mnn_MLP_with_Corr, self).__init__()
        self.layer1 = Mnn_Linear_Module_with_Rho(34 * 34, 34 * 34)
        self.layer2 = Mnn_Linear_Corr(34 * 34, 10, bias=True)

    def forward(self, ubar, sbar, rho):
        ubar, sbar, rho = self.layer1(ubar, sbar, rho)
        ubar, sbar, rho = self.layer2(ubar, sbar, rho)
        return ubar, sbar, rho


class N_Mnist_Model_Training:
    def __init__(self):
        self.file_path = "D:/Data_repos/N-MNIST/data/"
        self.EPOCHS = 5
        self.BATCH = 256
        self.learning_rate = 0.01
        self.random_seed = 1024
        self.log_interval = 10
        self.eps = 1e-8
        self.fps = 240
        self.train_loader = None
        self.test_loader = None
        self.model = None

    def fetch_dataset(self):
        self.train_loader = torch.utils.data.DataLoader(dataset=nmistDataset(
            datasetPath=self.file_path, mode="train", frames=self.fps),
            batch_size=self.BATCH, shuffle=True, num_workers=4)

        self.test_loader = torch.utils.data.DataLoader(dataset=nmistDataset(
            datasetPath=self.file_path, mode="test", frames=self.fps),
            batch_size=self.BATCH, shuffle=True, num_workers=4)

    def training_mlp_with_corr(self, fix_seed=True, save_op=True, save_name="n_mnist_mlp_corr.pt"):
        if fix_seed:
            torch.manual_seed(self.random_seed)
        if self.train_loader is None:
            self.fetch_dataset()
        net = Mnn_MLP_with_Corr()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        net.train()
        criterion = torch.nn.CrossEntropyLoss()
        count = 0
        print("------ MNIST MLP_CORR TRAINING START ------")
        for epoch in range(self.EPOCHS):
            for batch_idx, (mean, std, rho, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                out1, out2, out3 = net(mean, std, rho)
                loss = criterion(out1, target)
                loss.backward()
                optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(mean), len(self.train_loader.dataset),
                               100 * batch_idx / len(self.train_loader), loss.item()))
                    count += 1
        self.model = net
        if save_op:
            torch.save(net, save_name)

    def pred_func_fano(self, out1, out2):
        return torch.abs(out1) / (out2 + self.eps)

    def test_mlp_corr_model(self, model_name="n_mnist_mlp_corr.pt", mode="Test"):
        net = torch.load(model_name)

        if self.test_loader is None or self.train_loader is None:
            self.fetch_dataset()
        net.eval()
        test_loss = 0
        correct = 0
        if mode == "Test":
            data_loader = self.test_loader
        else:
            data_loader = self.train_loader
        with torch.no_grad():
            for mean, std, rho, target in data_loader:
                out1, out2, out3 = net(mean, std, rho)
                test_loss += F.cross_entropy(out1, target, reduction="sum").item()
                pred = out1.data.max(1, keepdim=True)[1]
                correct += torch.sum(pred.eq(target.data.view_as(pred)))
        test_loss /= len(data_loader.dataset)

        print('\nModel: {:}, {:} set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            str(type(net)), mode, test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))

    def continue_training(self, model_name, save_op=True):
        if self.model is None:
            net = torch.load(model_name)
        else:
            net = self.model
        if self.train_loader is None:
            self.fetch_dataset()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        net.train()
        criterion = torch.nn.CrossEntropyLoss()

        print("------ MNIST MLP_CORR TRAINING START ------")
        for epoch in range(self.EPOCHS):
            for batch_idx, (mean, std, rho, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                out1, out2, out3 = net(mean, std, rho)
                loss = criterion(out1, target)
                loss.backward()
                optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(mean), len(self.train_loader.dataset),
                               100 * batch_idx / len(self.train_loader), loss.item()))

        self.model = net
        if save_op:
            torch.save(net, model_name)

    def find_wrong_sample(self, model_name: str):
        if self.model is None:
            net = torch.load(model_name)
        else:
            net = self.model
        net.eval()
        wrong_indx = list()
        wrong_pred = list()
        data_set = nmistDataset(datasetPath=self.file_path, mode="test")
        correct = 0
        with torch.no_grad():
            for i in range(len(data_set)):
                u, s, r, t = data_set[i]
                u = u.view(1, -1)
                s = s.view(1, -1)
                r = r.view(1, 34*34, -1)
                out1, out2, out3 = net(u, s, r)
                pred = (out1 / (out2 + self.eps)).data.max(1, keepdim=True)[1]
                if pred == t:
                    correct += 1
                else:
                    wrong_indx.append(i)
                    wrong_pred.append(pred)

        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(data_set), 100. * correct / len(data_set)))
        with open("wrong_indx.bin", "wb") as f:
            pickle.dump(wrong_indx, f)
        with open("wrong_pred.bin", "wb") as f:
            pickle.dump(wrong_pred, f)


if __name__ == "__main__":
    name = "mnn_mlp_nmnist.pt"
    tool = N_Mnist_Model_Training()
    tool.file_path = "./data/n_mnist/"
    tool.fps = 500
    tool.EPOCHS = 5
    tool.fetch_dataset()
    tool.continue_training(name)
    tool.test_mlp_corr_model(model_name=name)

