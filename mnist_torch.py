# Blog: [MNIST手寫辨識(Pytorch version)](https://hackmd.io/@Maxlight/SkuYB0w6_)
import torch
from torch.utils import data as data_
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

import torchvision

# check if GPU CUDA is available
if torch.cuda.is_available():
  print(True)
else:
  print(False)

# Hyperparameter
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

# Display MNIST pictures
train_data = torchvision.datasets.MNIST(root = './mnist',train = True,transform = torchvision.transforms.ToTensor(),download = DOWNLOAD_MNIST)

print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.ion()
for i in range(11):
  plt.imshow(train_data.train_data[i].numpy(), cmap = 'gray')
  plt.title('%i' % train_data.train_labels[i])
  plt.pause(0.5)
plt.show()

# Data Processing
train_loader = data_.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True,num_workers = 2)

test_data = torchvision.datasets.MNIST(root = './mnist/', train = False)
test_x = torch.unsqueeze(test_data.test_data, dim = 1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]

# Build CNN
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 2,),# stride = 1, padding = (kernel_size-1)/2 = (5-1)/2
        nn.ReLU(),# (16, 28, 28)
        nn.MaxPool2d(kernel_size = 2),# (16, 14, 14)
    )
    self.conv2 = nn.Sequential(# (16, 14, 14)
        nn.Conv2d(16, 32, 5, 1, 2),# (32, 14, 14)
        nn.ReLU(),# (32,14,14)
        nn.MaxPool2d(2)# (32, 7, 7)
    )
    self.out = nn.Linear(32*7*7, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    output = self.out(x)
    return output, x

cnn = CNN()
print(cnn)

# Optimization & Loss function
optimization = torch.optim.Adam(cnn.parameters(), lr = LR)
loss_func = nn.CrossEntropyLoss()

# Training
for epoch in range(EPOCH):
  for step, (batch_x, batch_y) in enumerate(train_loader):
    bx = Variable(batch_x)
    by = Variable(batch_y)
    output = cnn(bx)[0]
    loss = loss_func(output, by)
    optimization.zero_grad()
    loss.backward()
    optimization.step()

    if step % 50 == 0:
        test_output, last_layer = cnn(test_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# Testing
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
