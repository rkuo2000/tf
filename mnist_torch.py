# MNIST CNN in PyTorch
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

## Dataset
train_data = datasets.MNIST(root='data', train=True,  transform = ToTensor(), download=True)
test_data  = datasets.MNIST(root='data', train=False, transform = ToTensor())

batch_size = 100
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
testloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

## Visualization of MNIST dataset
# plot one train_data
plt.imshow(train_data.data[0], cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()

# plot multiple train_data
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

## Build Model
class CNN(nn.Module):    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        #self.conv2_drop = nn.Dropout2d()        
        self.fc = nn.Linear(32 * 7 * 7, 10) # fully connected layer, output 10 classes
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        x = self.fc(x)
        return x    # return x for visualization

net = CNN()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

## Train Model
num_epochs = 10

net.train() # set to training model
for epoch in range(num_epochs):
    for images, labels in tqdm.tqdm(trainloader):
        outputs = net(images)
        loss = loss_func(outputs, labels)        
        optimizer.zero_grad()  # clear gradients for this training step        
        loss.backward()        # backpropagation, compute gradients
        optimizer.step()       # apply graidents
        # accuracy
        pred_y = torch.max(outputs, 1)[1].data.squeeze()
        accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    print("Epoch {}/{}, Loss: {:.4f}, Accuracy: {:.3f}".format(epoch+1, num_epochs, loss.item(), accuracy))

## Save Model 
torch.save(net.state_dict(), 'models/mnist_cnn.pth')

## Test Model
net.eval() # set to evaluate model
with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        pred_y = torch.max(outputs, 1)[1].data.squeeze()
        accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    print('Test Accuracy of the model on the 10000 test images: %.3f' % accuracy)

# test first 10 samples
sample = next(iter(testloader))
imgs, lbls = sample

actual_y = lbls[:10].numpy()
print(actual_y)

test_output = net(imgs[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(f'Predict: {pred_y}')
print(f'Actual : {actual_y}')

