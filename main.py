from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# following tutorial at https://pytorch.org/tutorials/beginner/blitz
# very good lectures about ML and pytorch: https://www.youtube.com/watch?v=SKq-pmkekTk&list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # network for first few examples
        ## 1 input image channel, 6 output channels (predictions?), 3x3 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 6, 3)
        ## is this 6 inputs, 16 outputs, 3x3 square convolution kernel?
        #self.conv2 = nn.Conv2d(6, 16, 3)
        ## an affine (linear, scaled, parallel lines stay parallel) operation: y = Wx + b
        #self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

        # network for classifier example

        # 3 input image channel, 6 output channels (predictions?), 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 36, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # is this 6 inputs, 16 outputs, 5x5 square convolution kernel?
        self.conv2 = nn.Conv2d(36, 16, 5)
        # an affine (linear, scaled, parallel lines stay parallel) operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from ???
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # select CUDA if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using", self.device)

    def forward(self, x):
        # network for first few examples

        ## Max pooling over a (2, 2) window
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        ## If the size is a square you can only specify a single number
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = x.view(-1, self.num_flat_features(x))
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        #return x

        # network for classifier example

        # Max pooling over a (2, 2) window
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def basics_demo():

    x = torch.empty(5, 3)
    print(x)

    print("CUDA STATUS: " + str(torch.cuda.is_available()))

    x = torch.randn(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        y = torch.ones_like(x, device=device)
        x = x.to(device)
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))

    x = torch.ones(2, 2, requires_grad=True)
    print(x)

    y = x + 2
    print(y)
    print(y.grad_fn)

    z = y * y * 3
    out = z.mean()

    print(z, out)

    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    print(a.requires_grad)
    a.requires_grad_(True)
    print(a.requires_grad)
    b = (a * a).sum()
    print(b.grad_fn)

    out.backward()
    print(x.grad)

    x = torch.randn(3, requires_grad=True)

    y = x * 2
    # Euclidean norm
    while y.data.norm() < 1000:
        y = y * 2

    print(y)

    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)

    y.backward(v)
    print(x.grad)

    print(x.requires_grad)
    # ** is exponentiation operator
    print((x ** 2).requires_grad)

    with torch.no_grad():
        print((x ** 2).requires_grad)

    print(x.requires_grad)
    y = x.detach()
    print(y.requires_grad)
    print(x.eq(y).all())


def net_demo():
    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight

    input_data = torch.randn(1, 1, 32, 32)
    out = net(input_data)
    print(out)

    net.zero_grad()
    out.backward(torch.randn(1, 10))

    output = net(input_data)
    target = torch.randn(10)  # dummy target
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)

    print(loss.grad_fn)
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    net.zero_grad()  # zeroes the gradient buffers of all parameters

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    # create optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    # in the training loop
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(input_data)

    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # does the update


def classifier_train_demo():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # show images
    imgshow(torchvision.utils.make_grid(images))

    net = Net()
    net.to(net.device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print('Training...')

    for epoch in range(4):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(net.device), data[1].to(net.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print stats
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i+ 1, running_loss/2000))
                running_loss = 0.0

    print('Finished Training')
    print('Saving trained model')
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print the images
    print('Ground Truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    imgshow(torchvision.utils.make_grid(images))

    # test the trained model
    net = Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)

    _, predicated = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicated[j]] for j in range(4)))

    print('Checking whole dataset')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data  # data[0].to(net.device), data[1].to(net.device)
            outputs = net(images)
            _, predicated = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicated == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    # performance check per class
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data  # data[0].to(net.device), data[1].to(net.device)
            outputs = net(images)
            # max is like saying "pick the largest prediction value and ignore the rest"
            _, predicated = torch.max(outputs, 1)
            c = (predicated == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


def imgshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    print("Hello PyTorch world!")
    classifier_train_demo()


if __name__ == "__main__":
    main()




