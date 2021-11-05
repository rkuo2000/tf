# $ cd ~/tf
# $ git clone https://github.com/rkuo2000/facenet-pytorch facenet_pytorch
# $ pip3 install opencv-python=4.5.3.56
# $ pip3 install pandas requests
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib
import matplotlib.pyplot as plt
workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# setup MTCNN and FaceNet model
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

net = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Prepare Faces Dataset
def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('facenet_pytorch/data/test_images')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

print(dataset.idx_to_class)

aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

print(len(aligned))
print(names)

# calculate image embeddings
aligned = torch.stack(aligned).to(device)

embeddings = net(aligned).detach().cpu()

print(len(embeddings))
print(embeddings[0].shape)

# print distance matrix for classes
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=names, index=names))

# Test : single face picture
mtcnn = MTCNN(keep_all=False)
img = plt.imread("facenet_pytorch/data/test_images/angelina_jolie/1.jpg")
plt.axis('off')
plt.imshow(img)
plt.show()

x_test, prob = mtcnn(img, return_prob=True)
print(prob)
print(x_test.shape)

x_aligned=[]
x_aligned.append(x_test)
test_aligned = torch.stack(x_aligned).to(device)
test_embeddings = net(test_aligned).detach().cpu()

print(len(embeddings), len(test_embeddings))

e1 = test_embeddings
dists = [(e1 - e2).norm().item() for e2 in embeddings]
print(dists)

print(names[np.argmin(dists)])

# Test : multiple faces picture
img = plt.imread("facenet_pytorch/data/angelina-leonardo-bradley.jpg")
plt.axis('off')
plt.imshow(img)
plt.show()

mtcnn = MTCNN(keep_all=True)

x_test, prob = mtcnn(img, return_prob=True)
print(prob)
print(x_test.shape)

test_aligned = x_test.to(device)
test_embeddings = net(test_aligned).detach().cpu()

dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in test_embeddings]
test_no=[]
[test_no.append(i) for i in range(len(test_embeddings))]
print(pd.DataFrame(dists, columns=names, index=test_no))

