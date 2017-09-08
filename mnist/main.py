from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    metavar='N',
    help='input batch size for training (default: 64)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    metavar='N',
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    metavar='N',
    help='number of epochs to train (default: 10)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.5,
    metavar='M',
    help='SGD momentum (default: 0.5)')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

## nlim_max should equal sample split count defined by batch size, if drop_last set
## TODO train size 60000 set as variable in datasets.MNIST?
N = 60000
N_test = 10000
nlim_max = N // args.batch_size
nlim = nlim_max
#nlim = 10
nlim_test = N_test // args.test_batch_size
args.nlim = nlim
args.nlim_test = nlim_test

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
    batch_size=args.batch_size,
    shuffle=(nlim == nlim_max),
    drop_last=True,
    **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
    batch_size=args.test_batch_size,
    shuffle=True,
    **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def get_n(epoch, batch_size, nlim, niter):
    return batch_size * ((epoch - 1) * nlim + (niter + 1))


def train(epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        #
        n = get_n(epoch, args.batch_size, nlim, batch_idx)
        writer.add_scalar('train/loss', loss, n)
        #
        if (batch_idx % args.log_interval == 0) | (batch_idx == nlim):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(
                    train_loader), loss.data[0]))
        if batch_idx > nlim:
            break


def test(epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        _test_loss = F.nll_loss(
            output, target, size_average=False).data[0]  # sum up batch loss
        n = get_n(epoch, args.test_batch_size, nlim_test, batch_idx)
        writer.add_scalar('test/loss', _test_loss / args.test_batch_size, n)
        test_loss += _test_loss
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        _correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        writer.add_scalar('test/acc', _correct / args.test_batch_size, n)
        correct += _correct

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
          format(test_loss, correct,
                 len(test_loader.dataset), 100. * correct / len(
                     test_loader.dataset)))


if __name__ == '__main__':

    try:
        writer = SummaryWriter()
        writer.add_text('config', str(args))

        for epoch in range(1, args.epochs + 1):
            train(epoch, writer)
            test(epoch, writer)

    except:
        writer.close()
