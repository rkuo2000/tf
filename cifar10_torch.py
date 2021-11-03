# CIFAR-10 CNN in PyTorch
import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

## Prepare Dataset
train_data = datasets.CIFAR10(root='data', train=True,  transform = ToTensor(), download=True)
test_data  = datasets.CIFAR10(root='data', train=False, transform = ToTensor())

print(train_data)
print(test_data)

batch_size = 100
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
testloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

## Visualization of Dataset
# plot one train_data
plt.imshow(train_data.data[0])
plt.title('%s' % classes[train_data.targets[0]])
plt.show()

# plot multiple train_data
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    classid = train_data[sample_idx][1]
    figure.add_subplot(rows, cols, i)
    plt.title(classes[classid])
    plt.axis("off")
    plt.imshow(train_data.data[sample_idx])
plt.show()

## Build Model
class CNN(nn.Module):    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( 3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(64 *5 *5, 512) # kernel=5 : 1600, kernel=3 : 2304
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.fl (x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

net = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

## Model Summary
# !pip3 install torch-summary
from torchsummary import summary
summary(net, (3,32,32))

## Train Model
num_epochs = 30

net.train() # set to training model
for epoch in range(num_epochs):
    for images, labels in tqdm.tqdm(trainloader):
        images = images.to(device) # for GPU
        labels = labels.to(device) # for GPU
        outputs = net(images)
        loss = criterion(outputs, labels)        
        optimizer.zero_grad()  # clear gradients for this training step        
        loss.backward()        # backpropagation, compute gradients
        optimizer.step()       # apply graidents
        # accuracy
        pred_y = torch.max(outputs, 1)[1].data.squeeze()
        accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    print("Epoch {}/{}, Loss: {:.4f}, Accuracy: {:.3f}".format(epoch+1, num_epochs, loss.item(), accuracy))

## Save Model
torch.save(net.state_dict(), 'models/cifar10_cnn.pth')

## Test Model
net.eval() # set to evaluate model
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device) # for GPU
        labels = labels.to(device) # for GPU
        outputs = net(images)
        pred_y = torch.max(outputs, 1)[1].data.squeeze()
        accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    print('Model accuracy of the %5d test images = %.3f' % (len(test_data), accuracy))

# test first 10 samples
sample = next(iter(testloader))
imgs, lbls = sample

actual_y = lbls[:10].numpy()
print(actual_y)

images = imgs[:10]
images = images.to(device) # for GPU
outputs = net(images)
pred_y = torch.max(outputs, 1)[1].data.cpu().numpy().squeeze()
print(f'Predict: {pred_y}')
print(f'Actual : {actual_y}')
