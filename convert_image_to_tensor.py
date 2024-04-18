# Import necessary libraries 
import sys
import torch 
from PIL import Image 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms 

filename = sys.argv[1] 
img = Image.open(filename) 

## resize
#img = img.resize((32,32))
#plt.imshow(img)
#plt.show()
#print("Resize to (32,32)...")

## convert to gray
#gray_img = img.convert("L")
#image = gray_img
image = img

# Convert the PIL image to Torch tensor 
transform = transforms.Compose([ transforms.PILToTensor() ]) 
img_tensor = transform(image) 

# Save tensor to .pt file
filename = filename.rsplit('.', maxsplit=1)[0]
torch.save(img_tensor, filename+".tensor")
print(f"Save tensor to {filename}.tensor file...")
