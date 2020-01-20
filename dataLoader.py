import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
classes = [0,1]

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

if __name__ == '__main__':
    
    # how many samples per batch to load
    batch_size = 32
    # percentage of training set to use as validation
    test_size = 0.3
    valid_size = 0.1

    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    data = datasets.ImageFolder('face',transform=transform)

    #For test
    num_data = len(data)
    indices_data = list(range(num_data))
    np.random.shuffle(indices_data)
    split_tt = int(np.floor(test_size * num_data))
    train_idx, test_idx = indices_data[split_tt:], indices_data[:split_tt]

    #For Valid
    num_train = len(train_idx)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size * num_train))
    train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_new_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
        sampler=train_sampler, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=1)
    test_loader = torch.utils.data.DataLoader(data, sampler = test_sampler, batch_size=batch_size, 
        num_workers=1)
    classes = [0,1]
    

        
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy() # convert images to numpy for display

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(10, 4))
    # display 20 images
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])
    plt.show()