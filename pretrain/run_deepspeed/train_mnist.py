'''
https://open-models-platform.github.io/CloudDeepSpeed/tutorials/cifar-10/

'''

import torch
import torchvision
import torchvision.transforms as transforms

import argparse
import deepspeed


def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR')

    # data
    # cuda
    parser.add_argument('--with_cuda', default=True,
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


# def initialize(args,
#                model,
#                optimizer=None,
#                model_params=None,
#                training_data=None,
#                lr_scheduler=None,
#                mpu=None,
#                dist_init_required=True,
#                collate_fn=None):


def main():
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    # batch_size = 4

    # trainset = torchvision.datasets.CIFAR10(root='/data/expGPT/pretrain/run_deepspeed/cifar-10-batches-py',
    #                                         train=True,
    #                                         # download=True,
    #                                         transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                           shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='/data/expGPT/pretrain/run_deepspeed/cifar-10-batches-py',
    #                                        train=False,
    #                                        # download=True,
    #                                        transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                          shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    from mnist_dataset import MNIST
    trainset = MNIST('/data/expGPT/mnist_data/data',
                     train=True, download=False,
                     transform=transform)

    import torch.nn as nn
    import torch.nn.functional as F

    # class Net(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.conv1 = nn.Conv2d(3, 6, 5)
    #         self.pool = nn.MaxPool2d(2, 2)
    #         self.conv2 = nn.Conv2d(6, 16, 5)
    #         self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #         self.fc2 = nn.Linear(120, 84)
    #         self.fc3 = nn.Linear(84, 10)
    #
    #     def forward(self, x):
    #         x = self.pool(F.relu(self.conv1(x)))
    #         x = self.pool(F.relu(self.conv2(x)))
    #         x = torch.flatten(x, 1)  # flatten all dimensions except batch
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            # output = F.log_softmax(x, dim=1)
            # return output
            return x

    net = Net()
    # net.to(torch.bfloat16)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    args = add_argument()

    # Initialize DeepSpeed to use the following features
    # 1) Distributed model
    # 2) Distributed data loader
    # 3) DeepSpeed optimizer
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=args, model=net,
                                                                   model_parameters=parameters,
                                                                   training_data=trainset)

    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data[0].to(model_engine.device)#.to(torch.bfloat16)
        labels = data[1].to(model_engine.device)#.to(torch.int32)

        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()
        if i % 1000 == 0:
            print(f'iter: {i}, loss: {loss.item()}')


if __name__ == '__main__':
    main()
