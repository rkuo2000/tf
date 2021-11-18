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
import cv2
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
    x_aligned, bbox, prob = mtcnn(x, return_prob=True)
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

# Test : multiple faces in camera captured
mtcnn = MTCNN(keep_all=True)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces, bboxes, probs = mtcnn(img, return_prob=True)

    test_aligned = faces.to(device)
    test_embeddings = net(test_aligned).detach().cpu()

    detected_names = []
    for e1 in test_embeddings:
        dists = [(e1-e2).norm().item() for e2 in embeddings]
        detected_names.append(names[np.argmin(dists)])

    for i, bbox in enumerate(bboxes):
        print(i, detected_names[i], bbox)
        x1,y1,x2,y2 = bbox
        cv2.rectangle(frame,(int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
        cv2.putText(frame, detected_names[i], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)    

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
