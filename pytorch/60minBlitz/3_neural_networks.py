import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## Define neural network with nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # input ch: 1, output ch: 6, kernel: 3 x 3
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) # 2 x 2 max pool
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # 2 x 2 max pool
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # except batch dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)


## Parameters
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1 weight


## Gradients
input_data = torch.randn(1, 1, 32, 32)
out = net(input_data)
print(out)
# zero the gradient buffers of all parameters
net.zero_grad()
# backprop with random gradients
out.backward(torch.rand(1, 10))


## Loss Function
output = net(input_data)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# trace back
print(loss.grad_fn) # MSE
print(loss.grad_fn.next_functions[0][0]) # linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU


## Backprop
net.zero_grad() # otherwise accumulated
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


## Update Weights

# vanila SGD
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# general optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# loop this
optimizer.zero_grad()
output = net(input_data)
loss = criterion(output, target)
loss.backward()
optimizer.step()










