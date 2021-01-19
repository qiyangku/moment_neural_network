# -*- coding: utf-8 -*-
from Mnn_Core.mnn_modules import *
import os


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
    def __init__(self, datasetPath="D:/Data_repos/N-MNIST/data/", mode: str="train"):
        super(nmistDataset, self).__init__()
        self.path = datasetPath
        self.mode = mode
        self.file_path = None
        self.labels = None
        self._fetch_files_path()

    def _fetch_files_path(self):
        data_dir = self.path + self.mode
        files_name = []
        labels = []
        for i in os.listdir(data_dir):
            next_dir = data_dir + "/" + i
            for j in os.listdir(next_dir):
                labels.append(i)
                files_name.append(data_dir + "/" + i + "/" + j)
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

        u, s, r = raw2Tensor(read2Dspikes(input_sample))

        return u, s, r, class_label

    def __len__(self):
        return len(self.file_path)


class Mnn_MLP_with_Corr(torch.nn.Module):
    def __init__(self):
        super(Mnn_MLP_with_Corr, self).__init__()
        self.layer1 = Mnn_Layer_with_Rho(34*34, 34*34)
        self.layer2 = Mnn_Linear_Corr(34*34, 10, bias=True)

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
        self.train_loader = None
        self.test_loader = None
        self.model = None

    def fetch_dataset(self):
        self.train_loader = torch.utils.data.DataLoader(dataset=nmistDataset(datasetPath=self.file_path, mode="train"),
                                                        batch_size=self.BATCH, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=nmistDataset(datasetPath=self.file_path, mode="test"),
            batch_size=self.BATCH, shuffle=True)

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

    def test_mlp_corr_model(self, model_name="n_mnist_mlp_corr.pt"):
        if self.model is None:
            net = torch.load(model_name)
        else:
            net = self.model

        if self.test_loader is None:
            self.fetch_dataset()
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for mean, std, rho, target in self.test_loader:
                out1, out2, out3 = net(mean, std, rho)
                test_loss += F.cross_entropy(out1, target, reduction="sum").item()
                pred = out1.data.max(1, keepdim=True)[1]
                correct += torch.sum(pred.eq(target.data.view_as(pred)))
        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))


def plot_event_static_info(pack, cmap="coolwarm", save_op=False, save_name=None, show_im=True):
    u, s, r, t = pack
    nrows = 1
    ncols = 3
    fig = plt.figure(figsize=(ncols*5, nrows*5))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(nrows, ncols, 1)
    im = ax.imshow(u.view(34, 34), cmap=cmap)
    ax.set_title("mean")
    plt.colorbar(im)
    
    ax = fig.add_subplot(nrows, ncols, 2)
    im = ax.imshow(s.view(34, 34), cmap=cmap)
    ax.set_title("std")
    plt.colorbar(im)
    
    ax = fig.add_subplot(nrows, ncols, 3)
    im = ax.imshow(r, cmap=cmap)
    ax.set_title("rho")
    plt.colorbar(im)
    
    plt.suptitle("label: {:}".format(t))
    if save_op:
        if save_name is None:
            fig.savefig("preprocessed_label_"+str(t)+".png")
        else:
            fig.savefig(save_name)
    if show_im:
        plt.show()


def plt_output_case(cmap="coolwarm"):
    data = nmistDataset()
    samples_idx = []
    check = []
    for i in range(len(data)):
        if len(samples_idx) == 10:
            break
        if data.labels[i] not in check:
            samples_idx.append(i)
            check.append(data.labels[i])
        else:
            continue
    # sample indx: [0, 5766, 12359, 18149, 24109, 29811, 35060, 40808, 46929, 52611]
    net = torch.load("n_mnist_mlp_corr.pt")
    net.eval()
    with torch.no_grad():
        for i in samples_idx:
            u, s, r, t = data[i]
            u = u.view(1, -1)
            s = s.view(1, -1)
            r = r.view(1, 34 * 34, -1)
            out1, out2, out3 = net(u, s, r)
            pred = out1.data.max(1, keepdim=True)[1]
            fig = plt.figure(figsize=(15, 5))
            fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, hspace=0.1, wspace=0.1)
            ax = fig.add_subplot(1, 3, 1)
            ax.plot(out1.detach().numpy().reshape(-1))
            ax.set_title("output mean")

            ax = fig.add_subplot(1, 3, 2)
            ax.plot(out2.detach().numpy().reshape(-1))
            ax.set_title("output std")

            ax = fig.add_subplot(1, 3, 3)
            im = ax.imshow(out3.detach().numpy().reshape(10, 10), cmap=cmap)
            ax.set_title("output rho")
            plt.colorbar(im)

            plt.suptitle("predict: {:}, ground true: {:}".format(pred.item(), t))
            fig.savefig("output_label_{:}.png".format(t))


if __name__ == '__main__':
    data = nmistDataset()
    u, s, r, t = data[10000]
    r = r.to_sparse()
    ans = r*r
    ans = ans.to_dense()
    plt.imshow(ans, cmap="rainbow")
    plt.show()









