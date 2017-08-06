
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from progress_bar import InitBar

NUM_CLASSES = 21
cwd = os.getcwd()
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
H, W = 321, 321

def load_dataset(part, SZ):
    # sz: the output size of the model, to which we want to shrink our target sizes
    # load training data
    images = []
    labels = []
    data_list_path = os.path.join(cwd, 'datasetVOC')
    dataset_path = '/home/monaj/bin/VOCdevkit/VOC2012'
    print('Loading {} data:'.format(part))
    with open(os.path.join(data_list_path, '{}.txt'.format(part))) as f:
        lines = f.readlines()
    filenames = [l.strip() for l in lines]
    N = len(filenames)
    pbar = InitBar()
    for i in range(N):
        pbar(100.0*float(i)/float(N))
        image_path, label_path = filenames[i].split()
        full_image_path = dataset_path + image_path
        full_label_path = dataset_path + label_path
        image = cv2.resize(cv2.imread(full_image_path), (H,W), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(cv2.imread(full_label_path, 0), SZ, interpolation=cv2.INTER_NEAREST)
        images.append(image)
        labels.append(label)
    mean_image = np.stack([IMG_MEAN[0]*np.ones((H,W)), IMG_MEAN[1]*np.ones((H,W)), IMG_MEAN[2]*np.ones((H,W))], axis=-1)
    mean_batch = np.expand_dims(mean_image, axis=0)
    images = np.stack(images, axis=0)
    images = images.astype('float32')
    images -= mean_batch
    labels = np.stack(labels, axis=0)
    print('\n\n{} data was loaded with a shape of {}'.format(part, images.shape))
    print('{} label was loaded with a shape of {}'.format(part, labels.shape))

    return (images, labels)
