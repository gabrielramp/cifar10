#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

global trainloader

# Number of images to obtain from database
batchsizenum = 8

# Transforming the images allow us to avoid overfitting during training. We are normalizing the images here 
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsizenum, shuffle=True, num_workers=2).to(device)

testset = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=False,download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsizenum, shuffle=False, num_workers=2).to(device)

if __name__ == '__main__':    
    


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
    plt.show()
