import argparse
import json
import time
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='This is Deep Laerning Training program'
)
parser.add_argument('data_directory', action="store")
parser.add_argument('--save_dir', action="store")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float, default=0.0001)
parser.add_argument('--hidden_units', action="store", type=int, default=4096)
parser.add_argument('--epochs', action="store", type=int, default=6)
parser.add_argument('--gpu', action="store_true", default=False)

arguments = parser.parse_args()

print(arguments)

# Set the data directory
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# Set the Transforms
# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([ transforms.Resize(225),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir,transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir,transform=valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

# read the flower names mapping file
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# set the device 
device = torch.device("cuda" if (torch.cuda.is_available() and arguments.gpu) else "cpu")
print("device is:", device)

# set the CNN model
arch = arguments.arch
if arch == "vgg11":
    model = models.vgg11(pretrained=True)
elif arch == "vgg13":
    model = models.vgg13(pretrained=True)
elif arch == "vgg19":
    model = models.vgg19(pretrained=True)
else:
    model = models.vgg16(pretrained=True)
    arch = "vgg16"
print("model: ", model)

for param in model.parameters():
    param.requires_grad = False

alpha = arguments.learning_rate
hiddent_units = arguments.hidden_units
epochs = arguments.epochs

fc = nn.Sequential(nn.Linear(25088,hiddent_units),
                   nn.ReLU(),
                   nn.Dropout(p=0.5),
                   nn.Linear(hiddent_units,102),
                   nn.LogSoftmax(dim=1)
                )

model.classifier = fc

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(),lr=alpha)

model.to(device)

#Training the network
steps = 0
running_loss = 0
accuracy = 0

print_every = 8
start = time.time()

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
duration = time.time() - start

print('Training complete in {:.0f}m {:.0f}s'.format(
        duration // 60, duration % 60))


# Save the checkpoint 
model.class_to_idx = train_dataset.class_to_idx


checkpoint = {"CNN":arch,
              "epochs":epochs,
              "classifier":model.classifier,
              "class_to_idx":model.class_to_idx,
              "state_dict":model.state_dict()}
if(arguments.save_dir):
    path = arguments.save_dir + "/checkpoint.pth"
else:
    path = "checkpoint.pth"  
print(path)
model.cpu()
torch.save(checkpoint,path)

