import numpy as np
import torch
import matplotlib.pyplot as plt
import os, shutil
import torchvision
import time
import copy

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from torchsummary import summary

from PIL import Image
from pathlib import Path

import glob
from scipy.signal import convolve2d
from skimage import img_as_ubyte, img_as_float
from tqdm.notebook import tqdm


def count_files(directory):
    """
    Count the number of files in a directory.

    Args:
        directory (str): The path to the directory.

    Returns:
        int: The total number of files in the directory.
    """
    file_count = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            file_count += 1
    return file_count
    
    
def create_dataset(src, dst, range_, class_):
    """
    Create a dataset by copying files from a source directory
    to a destination directory.

    Args:
        src (str): The path to the source directory
                   containing the original files.
        dst (str): The path to the destination directory
                   where the copied files will be saved.
        range_ (tuple): A tuple representing the range of
                        file indices to be included in the dataset.
        class_ (str): The class label or name of the files.

    Returns:
        None
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    fnames = [f'{class_}.{i}.jpeg' for i in range(*range_)]
    for fname in fnames:
        src_file = os.path.join(src, fname)
        dst_file = os.path.join(dst, fname)
        shutil.copyfile(src_file, dst_file)


def train_model_with_patience(model, criterion, optimizer, dataloaders,
                dev, dataset_sizes, num_epochs=25, patience=2, delta=0.001):
    """
    Trains a model with early stopping based on validation loss.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        criterion: Loss function to optimize.
        optimizer: Optimizer for updating model parameters.
        num_epochs (int): Number of epochs to train the model (default: 25).
        patience (int): Number of epochs to wait for improvement in
                        validation loss before early stopping (default: 2).
        delta (float): Minimum change in validation loss to be
                       considered as improvement (default: 0.001).

    Returns:
        torch.nn.Module: The trained model.
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.Inf
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(dev)
                labels = labels.to(dev)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'validation':
                if epoch_loss < best_loss - delta:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f'Early stopping triggered.'
                  f'No improvement in validation loss for {patience} epochs.')
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m'
          f' {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

def evaluate(model, test_loader, dev):
    """
    Evaluates the model on the test dataset and prints the test accuracy.

    Parameters:
        model (torch.nn.Module): The model to be evaluated.
        test_loader: Data loader for the test dataset.
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(dev)
            labels = labels.to(dev)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())

    print("Test Accuracy: {:.4f}".format(correct / total))
    
    
def imageshow(img):
    """
    Display an image.

    Parameters:
        img: Image tensor to be displayed.
    """
    img = img / 2 + 0.5
    npimg = img.numpy()
    npimg = np.clip(npimg, 0, 1)
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def visualize_preds(model, dataloaders, dev, class_names): 
    """
    Visualize predictions made by the model on a sample batch of images.

    Parameters:
        model (torch.nn.Module): The model used for prediction.

    """
    images, labels = next(iter(dataloaders['test']))
    images = images.to(dev)
    labels = labels.to(dev)

    imageshow(torchvision.utils.make_grid(images.cpu()))

    print('Real labels: ',
          ' '.join('%5s' % i for i in
                   [class_names[label] for label in labels]))

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
       
    print('Predicted: ',
          ' '.join('%5s' % i for i in
                   [class_names[label] for label in predicted]))
    
    
    
def white_patch(image, percentile=100):
    image_wp = img_as_ubyte(
        (image * 1.0 / np.percentile(image, percentile, axis=(0,1))).clip(0,1)
    )
    return image_wp
