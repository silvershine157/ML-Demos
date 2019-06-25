import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg') # no disp
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## Load & Normalize CIFAR10

batch_size = 64
num_epochs = 10

# [0, 1] -> [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
    'horse', 'ship', 'truck')

def imsave(img, name):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('./data/visual/'+name+'.png')

'''
# get random images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# save images
imsave(torchvision.utils.make_grid(images), 'CIFAR10_grid')
# print albels
print(''.join('%5s' % classes[labels[j]] for j in range(batch_size)))
'''

## Define CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()




## Define loss, optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


## Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(device)
net.to(device)

## Train network

for epoch in range(num_epochs):
    running_loss = 0.0
    running_count = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        running_count += inputs.size(0)

        # cuda tensors on GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print stat
        running_loss += loss.item()
    print('[Epoch %d] loss: %.3f' %(epoch+1, running_loss/running_count))
    running_loss = 0.0
    running_count = 0

print('Finished Training')


## Test

'''
dataiter = iter(testloader)
images, labels = dataiter.next()

imsave(torchvision.utils.make.make_grid(images), 'CNN test grid')
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

images = images.to(device)
labels = labels.to(device)
outputs = net(images)
outputs = outputs.to(cpu)

_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))
'''

'''
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        outputs = outputs.to(cpu)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %d %%'%(total, 100*correct/total))
'''

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        outputs = outputs.to(cpu)
        labels = labels.to(cpu)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            if(i >= labels.size(0)): # final batch size is different
                break
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i]/class_total[i]))

