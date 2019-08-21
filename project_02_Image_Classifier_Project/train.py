# -------------------
# RENAN VIEIRA DIAS
# UDACITY
# TRAIN
# -------------------

## python train.py flowers --epochs 1 --gpu
## python train.py flowers --epochs 1 --arch alexnet --gpu


# Imports
import argparse
import numpy as np
import json
import torch
import torch.nn.functional as F
import re
from torch import nn, optim
from torchvision import transforms,datasets,models
from tqdm import tqdm 
from PIL import Image

print('Import Complete')

# Creating the argument receiver
parser = argparse.ArgumentParser(description='Train Arguments')
parser.add_argument('data_dir'        , default = 'flowers'                    , help = 'Dataset Directory')
parser.add_argument('--save_dir'      , default = 'renan_checkpoint'           , help = 'Set directory to save checkpoints: python train.py data_dir --save_dir save_directory')
parser.add_argument('--arch'          , default = 'vgg16'                      , help = 'Choose architecture: python train.py data_dir --arch vgg16,alexnet,vgg11,vgg13,vgg19')
parser.add_argument('--learning_rate' , default = 0.0001           ,type=float , help = 'hyperparameters training learning_rate 0.0001')
parser.add_argument('--hidden_units'  , default = 8000             ,type=int   , help = 'hyperparameters hidden units 8000')
parser.add_argument('--epochs'        , default = 3                ,type=int   , help = 'hyperparameters training epochs 3')
parser.add_argument('--gpu'           , default = 'cpu'                        , help = 'To use GPU to train or the default CPU', action='store_const', const='cuda')
# p.add_argument('-f', '--foo', action='store_true')
args = parser.parse_args()

print('Arguments Complete')

# Constants
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir  = args.data_dir + '/test'

# Reading Category Label
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Selecting Device
device = torch.device(args.gpu)

# Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #Comment this line to debug and see the pictures
    ])
    ,'test': transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #Comment this line to debug and see the pictures
    ])
}
# Defining Validation to be the same as test
data_transforms['valid'] = data_transforms['test']
data_transforms['train'] = data_transforms['test']

# Load the datasets with ImageFolder
image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(train_dir, transform = data_transforms['train'])
image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform = data_transforms['valid'])
image_datasets['test']  = datasets.ImageFolder(test_dir , transform = data_transforms['test'])

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader( image_datasets['train'], shuffle=True)
dataloaders['valid'] = torch.utils.data.DataLoader( image_datasets['valid'], shuffle=True)
dataloaders['test']  = torch.utils.data.DataLoader( image_datasets['test'] , shuffle=True)

print('Image Transformation Complete')

# Importing model
input_layer = 0 #Make the same as the output from the pre-trained model (features model)

if args.arch == 'alexnet':
    model = models.alexnet(pretrained=True)
    input_layer = 9216

elif args.arch == 'vgg11':
    model = models.vgg11(pretrained=True)
    input_layer = 25088

elif args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    input_layer = 25088

elif args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_layer = 25088

elif args.arch == 'vgg19':
    model = models.vgg19(pretrained=True)
    input_layer = 25088

else:
    model = models.vgg16(pretrained=True)
    input_layer = 25088

#DEBUG    
# model = models.vgg16(pretrained=True)
# input_layer = 25088

print('Pre-Model Complete')

# Using the convolution layer as it is. Keeping it from changing during training.
for param in model.parameters():
    param.requires_grad = False

# Setup of the network
dropout_rate = 0.2
flower_categories = 102 #The output of our model

# Creating the Classifier part of the model
model.classifier = nn.Sequential(
    nn.Linear(input_layer, args.hidden_units),
    nn.ReLU(),
    nn.Dropout( p = dropout_rate ),
    nn.Linear(args.hidden_units, flower_categories),
    nn.LogSoftmax(dim=1)
)

# Selecting a loss criteria 
criterion = nn.NLLLoss()

# Selecting a Optimizer
optimizer = optim.Adam( model.classifier.parameters(), lr = args.learning_rate )

# Moving to device
model.to(device)

print('Model Created')

# Train the classifier layers using backpropagation using the pre-trained network to get the features
losses = {}
acc = {}
losses['train'], losses['valid'] = [],[]
acc['train'], acc['valid'] = [],[]

for e in range(args.epochs):
        
        print("Training epoch: {}/{}   please wait...".format(e+1, args.epochs))
        train_loss = 0
        train_accuracy = 0
        train_len = len(dataloaders['train'])
        for images, labels in dataloaders['train']:
            
            # Training step 
            images_d, labels_d = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model( images_d )
            loss = criterion(output, labels_d)
            loss.backward()
            
            optimizer.step()      
            train_loss += loss.item()

            # Measuring Accuracy   
            output_prob = torch.exp(output)
            top_p, top_class = output_prob.topk(1, dim=1)
            equals = top_class == labels_d.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor))

            #Debug
            # break


        # ------ Model Results -----
        print("Evaluating epoch: {}/{}   please wait...".format(e+1, args.epochs))
        model.eval()
        valid_loss = 0
        valid_accuracy = 0
        valid_len = len(dataloaders['valid'])
        for images, labels in dataloaders['valid']:

            # Measuring Validation Accuracy 
            images_d, labels_d = images.to(device), labels.to(device)
            output = model( images_d )
            loss = criterion(output, labels_d)
            valid_loss += loss.item()
            output_prob = torch.exp(output)
            top_p, top_class = output_prob.topk(1, dim=1)
            equals = top_class == labels_d.view(*top_class.shape)
            valid_accuracy += torch.mean(equals.type(torch.FloatTensor))

            #Debug
            # break

        # ----- Saving Losses and Accuracy -----     
        losses['train'].append(train_loss/len(dataloaders['train']))
        losses['valid'].append(valid_loss/valid_len)
        acc['train'].append(train_accuracy/len(dataloaders['train']))
        acc['valid'].append(valid_accuracy/valid_len)

        print("Epoch: {}/{} |".format(e+1, args.epochs),
              "Train Loss: {:.3f}  | ".format(train_loss    /train_len),
              "Train Accu: {:.3f}  | ".format(train_accuracy/train_len),
              "Valid Loss: {:.3f}  | ".format(valid_loss    /valid_len),
              "Valid Accu: {:.3f}    ".format(valid_accuracy/valid_len))

        model.train()

print("Model Trained")  


# Do validation on the test set
print('Testing the model')
model.eval()
for data_type in ['train','valid','test']:
    accuracy = 0
    
    
    data_size = len(dataloaders[data_type])
    for images, labels in dataloaders[data_type]:
        images, labels = images.to(device), labels.to(device)

        output = model( images )
       
        output_prob = torch.exp(output)
        top_p, top_class = output_prob.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

        # DEBUG
        # break
    
    print("accuracy: {:.3f}  Type: {}   DataSize: {}".format(accuracy/data_size, data_type,data_size) )
model.train()


# TODO: Save the checkpoint 
file_name = args.save_dir + '/renandias_model_checkpoint.pth'

checkpoint = {
    'image_transform_train':data_transforms['train'],
    'image_transform_test_validation':data_transforms['test'],
    'class_to_index':image_datasets['train'].class_to_idx,
    'input_size': input_layer,
    'hidden_size':args.hidden_units,
    'output_size': flower_categories,
    'dropout_rate':dropout_rate,
    'optimizer_state_dict': optimizer.state_dict,
    'epochs':args.epochs,
    'model_state_dict': model.state_dict(),
    'model_arch': args.arch
}


torch.save( checkpoint, file_name )
print('Checkpoitn Saved')