import os, shutil
import numpy as np

root = '/mnt/shared'
all_images = os.listdir(os.path.join(root, 'camvid_old', 'images'))[:-1]
all_labels = os.listdir(os.path.join(root, 'camvid_old', 'labels'))

valid_images = np.loadtxt(os.path.join(root, 'camvid_old', 'valid.txt'), str)
valid_labels = [f'{img.split(".")[0]}_P.{img.split(".")[1]}' for img in valid_images]

train_images, train_labels = [], []
for img, lab in zip(all_images, all_labels):
    if img in valid_images and lab in valid_labels:
        pass
    else:
        train_images.append(img)
        train_labels.append(lab)

def move_imgs(img_list, source, dest):
    for img in img_list:
        shutil.copy2(os.path.join(source, img), os.path.join(dest, img))

move_imgs(train_images, os.path.join(root, 'camvid_old', 'images'), os.path.join(root, 'camvid', 'images', 'train'))
move_imgs(valid_images, os.path.join(root, 'camvid_old', 'images'), os.path.join(root, 'camvid', 'images', 'test'))
move_imgs(train_labels, os.path.join(root, 'camvid_old', 'labels'), os.path.join(root, 'camvid', 'labels', 'train'))
move_imgs(valid_labels, os.path.join(root, 'camvid_old', 'labels'), os.path.join(root, 'camvid', 'labels', 'test'))

