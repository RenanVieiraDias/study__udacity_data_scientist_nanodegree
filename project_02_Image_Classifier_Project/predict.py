# -------------------
# RENAN VIEIRA DIAS
# UDACITY
# PREDICT
# -------------------

## python predict.py flowers/train/17/image_03838.jpg renan_checkpoint/renandias_model_checkpoint.pth --gpu
## python predict.py flowers/train/17/image_03838.jpg renan_checkpoint/renandias_model_checkpoint.pth --gpu


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

print('importe complete')


# Creating the argument receiver
parser = argparse.ArgumentParser(description='Predict Arguments')
parser.add_argument('image_dir'        , default = 'flowers/train/17/image_03838.jpg'                  , help = 'Path to predict the image')
parser.add_argument('checkpoint'       , default = 'renan_checkpoint/renandias_model_checkpoint.pth'   , help = 'The model checkpoint')
parser.add_argument('--top_k'          , default = '1'                  ,type=int                      , help = 'Return top KK most likely classes')
parser.add_argument('--category_names' , default = 'cat_to_name.json'                                  , help = 'Use a mapping of categories to real names')
parser.add_argument('--gpu'            , default = 'cpu'                                               , help = 'To use GPU to train or the default CPU', action='store_const', const='cuda')
# p.add_argument('-f', '--foo', action='store_true')
args = parser.parse_args()

print('Argument complete')
#print(args) #DEBUG


# ------------ Functions --------------

# Loading a checkpoint and rebuilds the model
def restoremodel(state_dict):
       
    # restored = models.vgg16(pretrained=True) #DEBUG
	# Restoring type of model
	if state_dict['model_arch'] == 'alexnet':
	    restored = models.alexnet(pretrained=True)
	elif state_dict['model_arch'] == 'vgg11':
	    restored = models.vgg11(pretrained=True)
	elif state_dict['model_arch'] == 'vgg13':
	    restored = models.vgg13(pretrained=True)
	elif state_dict['model_arch'] == 'vgg16':
	    restored = models.vgg16(pretrained=True)
	elif state_dict['model_arch'] == 'vgg19':
	    restored = models.vgg19(pretrained=True)
	else:
	    restored = models.vgg16(pretrained=True)

	restored.classifier = nn.Sequential(
            nn.Linear(state_dict['input_size'], state_dict['hidden_size']),
            nn.ReLU(),
            nn.Dropout( p = state_dict['dropout_rate'] ),
            nn.Linear(state_dict['hidden_size'], state_dict['output_size']),
            nn.LogSoftmax(dim=1)
        )

	
	restored.load_state_dict( state_dict['model_state_dict'] )
    
	return(restored)


# Function to process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''  
    means = [0.485, 0.456, 0.406]
    stddev = [0.229, 0.224, 0.225]
    resize = 256
    cutsize = 224
    
    #First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio.
    im_resized = image.resize((resize,resize))
    
    #Then you'll need to crop out the center 224x224 portion of the image.
    margin = (resize-cutsize)/2
    im_crop = im_resized.crop((margin, margin, resize-margin, resize-margin))
    
    #Convert from 0-255 to 0-1 
    np_image = np.array(im_crop)/255
    
    #Normalize Mean and Standard Deviation 
    np_image_mean_n = ( np_image - means ) / stddev
        
    #Re-order array.
    processed_image = np_image_mean_n.transpose((2,0,1))
    
    return(processed_image)


# Implement the code to predict the class from an image file
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model_d = model.double()
    
    im = Image.open(image_path)
    im_p = process_image(im)
    im_t = torch.from_numpy(im_p).reshape([1, 3, 224, 224])
    im_out = model_d(im_t.to(device))
    im_prb = torch.exp(im_out)
    im_top_p, im_top_class = im_prb.topk(topk, dim=1)

    return( (im_top_p.tolist()[0], im_top_class.tolist()[0]) )
  

# ------ main --------

# Selecting Device
device = torch.device(args.gpu)
print('Device Selected')


# Loading model
loaded_file = torch.load( args.checkpoint )
checkpoint_model = restoremodel(loaded_file)
checkpoint_model.to(device)
print('Model Restored')


# TODO: Display an image along with the top 5 classes
top_p, top_c = predict( args.image_dir, checkpoint_model , args.top_k)
print('Prediction Done')


# Folder number to folder name
folder_number_to_name = {0:'1',1:'10',2:'100',3:'101',4:'102',5:'11',6:'12',7:'13',8:'14',9:'15',10:'16',11:'17',12:'18',13:'19',14:'2',15:'20',16:'21',17:'22',18:'23',19:'24',20:'25',21:'26',22:'27',23:'28',24:'29',25:'3',26:'30',27:'31',28:'32',29:'33',30:'34',31:'35',32:'36',33:'37',34:'38',35:'39',36:'4',37:'40',38:'41',39:'42',40:'43',41:'44',42:'45',43:'46',44:'47',45:'48',46:'49',47:'5',48:'50',49:'51',50:'52',51:'53',52:'54',53:'55',54:'56',55:'57',56:'58',57:'59',58:'6',59:'60',60:'61',61:'62',62:'63',63:'64',64:'65',65:'66',66:'67',67:'68',68:'69',69:'7',70:'70',71:'71',72:'72',73:'73',74:'74',75:'75',76:'76',77:'77',78:'78',79:'79',80:'8',81:'80',82:'81',83:'82',84:'83',85:'84',86:'85',87:'86',88:'87',89:'88',90:'89',91:'9',92:'90',93:'91',94:'92',95:'93',96:'94',97:'95',98:'96',99:'97',100:'98',101:'99'}

# Reading Category Label
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Rescuing The name from image path
m = re.search('flowers/[A-z]*/(.+?)/', args.image_dir)
if m:
    im_category_number = m.group(1)
im_category_name = cat_to_name[im_category_number]
print('Category Number and Name Created')

# Transforming from class number to name
top_names = []
for c in top_c:
    top_names.append( cat_to_name[folder_number_to_name[c]] )
print('top names stored')



# Final Results
print(' ----- The species from the Flower ----- ')

print("Category Number: {} |".format(im_category_number),
      "Category Name: {}  | ".format(im_category_name))

print(' ----- The Prediction ----- ')

for j in range(0,args.top_k):
	print("Prob: {:.3f} | ".format(top_p[j]),
		  "Category Number: {} | ".format(top_c[j]),
          "Category Name: {} ".format(top_names[j]))