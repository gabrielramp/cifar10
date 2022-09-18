#%matplotlib inline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda
import torchvision
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def loader():
    # Number of images to obtain from database
    global batchsizenum
    batchsizenum = 4

    # Transforming the images allow us to avoid overfitting during training. We are normalizing the images here 
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    global trainloader, testloader
    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsizenum, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=False,download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsizenum, shuffle=False, num_workers=2)

    # This function loads up the images for viewing
    def imshow(img):
        img = img / 2 + 0.5 # Unnormalize image
        npimg = img.numpy() # Obtain images
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # Obtain random images for training
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    # Load the images for viewing
    imshow(torchvision.utils.make_grid(images))
    
    # Print their labels
    print("Here's what our images consist of:")
    print(' '.join('%5s' % classes[labels[j]] for j in range(batchsizenum)))   
    #plt.show()
    
def model():
    # Here is our network, that will classify the images
    class Net(nn.Module):
        # Convolutional Layers
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 4, padding=2) # Padding of 2 outputs 32x32
            self.conv2 = nn.Conv2d(32, 32, 4)   
            self.conv3 = nn.Conv2d(32, 16, 4)
            self.maxpool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 5 * 5, 120) # Parameter is output channels times the resolution of these channels. 32 > 23 > 14 > 10 > 5
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        # Activational Layers
        def forward(self, x):
            x = self.maxpool(F.relu(self.conv1(x))) # Conv w/ padding outputs 32x32, ReLU retains 32x32, maxpool
            x = self.maxpool(F.relu(self.conv2(x)))
            x = (F.relu(self.conv3(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    net = Net()

    crossentropyloss = nn.CrossEntropyLoss()
    optimize = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Learning over the dataset:
    for epoch in range(1):
        progressloss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Obtain inputs; data is a list of [inputs, labels]
            net.to(device)
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradient
            optimize.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = crossentropyloss(outputs, labels)
            loss.backward()
            optimize.step()

            # Print statistics
            progressloss += loss.item()
            if i % 1000 == 999:
                print(f'Epoch {epoch + 1}, Sample{i + 1:5d} Loss: {progressloss / 2000:.3f}')
                progressloss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), './trainingdata')

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # This function loads up the images for viewing
    def imshow(img):
        img = img / 2 + 0.5 # Unnormalize image
        npimg = img.numpy() # Obtain images
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
    # Show images
    imshow(torchvision.utils.make_grid(images))
    print('Real: ', ''.join(f'{classes[labels[j]]:5s}' for j in range(batchsizenum)))
    plt.show()

    net = Net()
    net.load_state_dict(torch.load('./trainingdata'))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'for j in range(batchsizenum)))

    numcorrect = 0
    numtotal = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            numtotal += labels.size(0)
            numcorrect += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * numcorrect // numtotal} %')

    # prepare to count predictions for each class
    predictedCorrectly = {classname: 0 for classname in classes}
    totalPredicted = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    predictedCorrectly[classes[label]] += 1
                totalPredicted[classes[label]] += 1

    # Print accuracy for each class
    for classname, correctcount in predictedCorrectly.items():
        accuracy = 100 * float(correctcount) / totalPredicted[classname]
        print(f'{classname:5s} Accuracy: {accuracy:.2f} %')

# Main class
if __name__ == '__main__':
    # Create global device and classes to be called in main()
    # Device specifies cuda vs CPU
    # Classes lists the possible classes in dataset
    global device, classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(device)
    loader()
    model()