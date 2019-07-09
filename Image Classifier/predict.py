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

parser = argparse.ArgumentParser(
    description='This is Deep Laerning Prediction program'
)
parser.add_argument('image_path', action="store")
parser.add_argument('checkpoint', action="store")
parser.add_argument('--top_k', action="store", type=int, default=5)
parser.add_argument('--category_names', action="store", default='cat_to_name.json')
parser.add_argument('--gpu', action="store_true", default=False)


print(parser.parse_args())
arguments = parser.parse_args()

# Load the checkpoint and create the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint["CNN"]
    if arch == "vgg11":
        model = models.vgg11(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    epochs = checkpoint['epochs']
    return model

# process the Image 
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Resize the image preserving aspect ratio
    width, height= image.size
    
    if width > height:
        ratio = width/height
        size = int(256 * ratio) , 256
    else:
        ratio = height/width
        size = 256, int(256*ratio)
    
    image.thumbnail(size,Image.ANTIALIAS)
    
    #Crop the Image
    width, height = image.size
    h_margin = (width - 244)/2 
    v_margin = (height - 244)/2
    
    left_margin = int(h_margin)
    upper_margin = int(v_margin)
    right_margin = left_margin + 244
    lower_margin = upper_margin + 244
    
    
    box = (left_margin, upper_margin, right_margin, lower_margin)
    image = image.crop(box)
    
    #Normalize the image
    image = np.array(image)
    image = image.astype(np.float32)
    image/=255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std
    
    #color channel first dimension
    image = image.transpose((2, 0, 1))
    
    return image

#Predict the classes of the image
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)

    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    model_input = image_tensor.unsqueeze(0)
    model_input = model_input.to(device)
    logps = model.forward(model_input)
    ps = torch.exp(logps)
    probs, classes = ps.topk(topk, dim=1)
    #for cls in classes:
    indx_to_class = {value: key for key, value in model.class_to_idx.items()}

    top_classes = classes.cpu().detach().numpy().flatten()
    top_probs = probs.cpu().detach().numpy().flatten()
    
    top_classes = [indx_to_class[cls] for cls in top_classes]
        
    return top_probs,top_classes

device = torch.device("cuda" if (torch.cuda.is_available() and arguments.gpu) else "cpu")
checkpoint_path = arguments.checkpoint
model = load_checkpoint(checkpoint_path)
model.to(device)
image_path = arguments.image_path
image = Image.open(image_path)
image = process_image(image)
top_k = arguments.top_k
probs, classes = predict(image_path, model,top_k)



with open(arguments.category_names, 'r') as f:
    cat_to_name = json.load(f)

flower_num = image_path.split('/')[2]
title_ = cat_to_name[flower_num]
print('flower_class:', title_)
labels = [cat_to_name[cls] for cls in classes ]

print('top', top_k, 'probabilities:')

for prob, label in zip(probs,labels):
    print(label, prob)