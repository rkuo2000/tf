# Import necessary libraries 
import sys
import torch 
from PIL import Image 
import torchvision.transforms as transforms 
 
filename = sys.argv[1] 
image = Image.open(filename+".jpg") 
  
# Define a transform to convert PIL  
# image to a Torch tensor 
transform = transforms.Compose([ 
    transforms.PILToTensor() 
]) 
  
# transform = transforms.PILToTensor() 
# Convert the PIL image to Torch tensor 
img_tensor = transform(image) 
  
torch.save(img_tensor, filename+".pt")
