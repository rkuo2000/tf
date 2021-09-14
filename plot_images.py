# pip install matplotlib
# git clone https://github.com/rkuo2000/tf
# cd tf
import os
import glob
import matplotlib.pyplot as plt

img_dir = './image'
all_img_files = glob.glob(img_dir+'/*.jpg')

img_files = all_img_files[:10]
print(img_files)

plt.figure(figsize=(10, 10))

for idx, file_path in enumerate(img_files):
    plt.subplot(5, 5, idx+1)

    img = plt.imread(file_path)
    plt.tight_layout()
    plt.imshow(img, cmap='gray')
plt.show()
