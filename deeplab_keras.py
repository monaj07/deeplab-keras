
import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Activation
from tensorflow.contrib.keras.python.keras.layers.merge import Add
from tensorflow.contrib.keras.python.keras.layers.core import Dropout, Dense, Flatten, Reshape
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.utils import to_categorical
from tensorflow.contrib.keras.python.keras.backend import learning_phase
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pdb
from progress_bar import InitBar


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    #pdb.set_trace()
    images = images.astype('float32')
    images -= mean_batch
    labels = np.stack(labels, axis=0)
    print('\n\n{} data was loaded with a shape of {}'.format(part, images.shape))
    print('{} label was loaded with a shape of {}'.format(part, labels.shape))

    return (images, labels)


def main():
    ################################################
    ########## Preparing the dataset ###############
    ################################################
    training_images, training_labels = load_dataset('train_limited', (40,40)) # The output of this resnet model is (40,40)
    val_images, val_labels = load_dataset('val', (40,40))
    training_labels = training_labels.astype('float32')
    val_labels = val_labels.astype('float32')
    training_labels[training_labels==255] = -1
    val_labels[val_labels==255] = -1

    N = training_labels.shape[0]
    Nv = val_labels.shape[0]
    perm_train = np.random.permutation(N)

    training_images = training_images[perm_train, :, :, :]
    training_labels = training_labels[perm_train, :, :]

    training_labels = np.reshape(training_labels, (N, 40*40, 1))
    val_labels = np.reshape(val_labels, (Nv, 40*40, 1))

    ################################################
    ######## Building the model ####################
    ################################################
    input_layer = Input(shape=(H,W,3), name='input_layer')
    conv1_1  = Conv2D(filters=64, kernel_size=7, strides=(2,2), use_bias=False, padding='same', name='conv1')(input_layer)
    bn1_1    = BatchNormalization(name='bn_conv1')(conv1_1)
    relu1_1  = Activation('relu')(bn1_1)
    mxp1_1   = MaxPooling2D(pool_size=3, strides=(2,2))(relu1_1)
    conv1_2  = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2a_branch1')(mxp1_1)
    bn1_2    = BatchNormalization(name='bn2a_branch1')(conv1_2)

    conv2_1  = Conv2D(filters=64, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2a_branch2a')(mxp1_1)
    bn2_1    = BatchNormalization(name='bn2a_branch2a')(conv2_1)
    relu2_1  = Activation('relu')(bn2_1)
    conv2_2  = Conv2D(filters=64, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res2a_branch2b')(relu2_1)
    bn2_2    = BatchNormalization(name='bn2a_branch2b')(conv2_2)
    relu2_2  = Activation('relu')(bn2_2)
    conv2_3  = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2a_branch2c')(relu2_2)
    bn2_3    = BatchNormalization(name='bn2a_branch2c')(conv2_3)

    merge3   = Add()([bn1_2, bn2_3])
    relu3_1  = Activation('relu')(merge3)
    conv3_1  = Conv2D(filters=64, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2b_branch2a')(relu3_1)
    bn3_1    = BatchNormalization(name='bn2b_branch2a')(conv3_1)
    relu3_2  = Activation('relu')(bn3_1)
    conv3_2  = Conv2D(filters=64, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res2b_branch2b')(relu3_2)
    bn3_2    = BatchNormalization(name='bn2b_branch2b')(conv3_2)
    relu3_3  = Activation('relu')(bn3_2)
    conv3_3  = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2b_branch2c')(relu3_3)
    bn3_3    = BatchNormalization(name='bn2b_branch2c')(conv3_3)

    merge4   = Add()([relu3_1, bn3_3])
    relu4_1  = Activation('relu')(merge4)
    conv4_1  = Conv2D(filters=64, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2c_branch2a')(relu4_1)
    bn4_1    = BatchNormalization(name='bn2c_branch2a')(conv4_1)
    relu4_2  = Activation('relu')(bn4_1)
    conv4_2  = Conv2D(filters=64, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res2c_branch2b')(relu4_2)
    bn4_2    = BatchNormalization(name='bn2c_branch2b')(conv4_2)
    relu4_3  = Activation('relu')(bn4_2)
    conv4_3  = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2c_branch2c')(relu4_3)
    bn4_3    = BatchNormalization(name='bn2c_branch2c')(conv4_3)

    merge5   = Add()([relu4_1, bn4_3])
    relu5_1  = Activation('relu')(merge5)
    conv5_1  = Conv2D(filters=512, kernel_size=1, strides=(2,2), use_bias=False, padding='same', name='res3a_branch1')(relu5_1)
    bn5_1    = BatchNormalization(name='bn3a_branch1')(conv5_1)

    conv6_1  = Conv2D(filters=128, kernel_size=1, strides=(2,2), use_bias=False, padding='same', name='res3a_branch2a')(relu5_1)
    bn6_1    = BatchNormalization(name='bn3a_branch2a')(conv6_1)
    relu6_1  = Activation('relu')(bn6_1)
    conv6_2  = Conv2D(filters=128, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res3a_branch2b')(relu6_1)
    bn6_2    = BatchNormalization(name='bn3a_branch2b')(conv6_2)
    relu6_2  = Activation('relu')(bn6_2)
    conv6_3  = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3a_branch2c')(relu6_2)
    bn6_3    = BatchNormalization(name='bn3a_branch2c')(conv6_3)

    merge7   = Add()([bn5_1, bn6_3])
    relu7_1  = Activation('relu', name='res3a_relu')(merge7)
    conv7_1  = Conv2D(filters=128, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3b1_branch2a')(relu7_1)
    bn7_1    = BatchNormalization(name='bn3b1_branch2a')(conv7_1)
    relu7_2  = Activation('relu')(bn7_1)
    conv7_2  = Conv2D(filters=128, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res3b1_branch2b')(relu7_2)
    bn7_2    = BatchNormalization(name='bn3b1_branch2b')(conv7_2)
    relu7_3  = Activation('relu')(bn7_2)
    conv7_3  = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3b1_branch2c')(relu7_3)
    bn7_3    = BatchNormalization(name='bn3b1_branch2c')(conv7_3)

    merge8   = Add()([relu7_1, bn7_3])
    relu8_1  = Activation('relu', name='res3b1_relu')(merge8)
    conv8_1  = Conv2D(filters=128, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3b2_branch2a')(relu8_1)
    bn8_1    = BatchNormalization(name='bn3b2_branch2a')(conv8_1)
    relu8_2  = Activation('relu')(bn8_1)
    conv8_2  = Conv2D(filters=128, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res3b2_branch2b')(relu8_2)
    bn8_2    = BatchNormalization(name='bn3b2_branch2b')(conv8_2)
    relu8_3  = Activation('relu')(bn8_2)
    conv8_3  = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3b2_branch2c')(relu8_3)
    bn8_3    = BatchNormalization(name='bn3b2_branch2c')(conv8_3)

    merge9   = Add()([relu8_1, bn8_3])
    relu9_1  = Activation('relu', name='res3b2_relu')(merge9)
    conv9_1  = Conv2D(filters=128, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3b3_branch2a')(relu9_1)
    bn9_1    = BatchNormalization(name='bn3b3_branch2a')(conv9_1)
    relu9_2  = Activation('relu')(bn9_1)
    conv9_2  = Conv2D(filters=128, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res3b3_branch2b')(relu9_2)
    bn9_2    = BatchNormalization(name='bn3b3_branch2b')(conv9_2)
    relu9_3  = Activation('relu')(bn9_2)
    conv9_3  = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3b3_branch2c')(relu9_3)
    bn9_3    = BatchNormalization(name='bn3b3_branch2c')(conv9_3)

    merge10  = Add()([relu9_1, bn9_3])
    relu10_1 = Activation('relu', name='res3b3_relu')(merge10)
    conv10_1 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4a_branch1')(relu10_1)
    bn10_1   = BatchNormalization(name='bn4a_branch1')(conv10_1)

    conv11_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4a_branch2a')(relu10_1)
    bn11_1   = BatchNormalization(name='bn4a_branch2a')(conv11_1)
    relu11_1 = Activation('relu')(bn11_1)
    at_conv11_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4a_branch2b')(relu11_1)
    bn11_2   = BatchNormalization(name='bn4a_branch2b')(at_conv11_2)
    relu11_2 = Activation('relu')(bn11_2)
    conv11_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4a_branch2c')(relu11_2)
    bn11_3   = BatchNormalization(name='bn4a_branch2c')(conv11_3)

    merge12  = Add()([bn10_1, bn11_3])
    relu12_1 = Activation('relu', name='res4a_relu')(merge12)
    conv12_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b1_branch2a')(relu12_1)
    bn12_1   = BatchNormalization(name='bn4b1_branch2a')(conv12_1)
    relu12_2 = Activation('relu')(bn12_1)
    conv12_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b1_branch2b')(relu12_2)
    bn12_2   = BatchNormalization(name='bn4b1_branch2b')(conv12_2)
    relu12_3 = Activation('relu')(bn12_2)
    conv12_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b1_branch2c')(relu12_3)
    bn12_3   = BatchNormalization(name='bn4b1_branch2c')(conv12_3)

    merge13  = Add()([relu12_1, bn12_3])
    relu13_1 = Activation('relu', name='res4b1_relu')(merge13)
    conv13_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b2_branch2a')(relu13_1)
    bn13_1   = BatchNormalization(name='bn4b2_branch2a')(conv13_1)
    relu13_2 = Activation('relu')(bn13_1)
    conv13_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b2_branch2b')(relu13_2)
    bn13_2   = BatchNormalization(name='bn4b2_branch2b')(conv13_2)
    relu13_3 = Activation('relu')(bn13_2)
    conv13_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b2_branch2c')(relu13_3)
    bn13_3   = BatchNormalization(name='bn4b2_branch2c')(conv13_3)

    merge14  = Add()([relu13_1, bn13_3])
    relu14_1 = Activation('relu', name='res4b2_relu')(merge14)
    conv14_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b3_branch2a')(relu14_1)
    bn14_1   = BatchNormalization(name='bn4b3_branch2a')(conv14_1)
    relu14_2 = Activation('relu')(bn14_1)
    conv14_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b3_branch2b')(relu14_2)
    bn14_2   = BatchNormalization(name='bn4b3_branch2b')(conv14_2)
    relu14_3 = Activation('relu')(bn14_2)
    conv14_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b3_branch2c')(relu14_3)
    bn14_3   = BatchNormalization(name='bn4b3_branch2c')(conv14_3)

    merge15  = Add()([relu14_1, bn14_3])
    relu15_1 = Activation('relu', name='res4b3_relu')(merge15)
    conv15_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b4_branch2a')(relu15_1)
    bn15_1   = BatchNormalization(name='bn4b4_branch2a')(conv15_1)
    relu15_2 = Activation('relu')(bn15_1)
    conv15_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b4_branch2b')(relu15_2)
    bn15_2   = BatchNormalization(name='bn4b4_branch2b')(conv15_2)
    relu15_3 = Activation('relu')(bn15_2)
    conv15_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b4_branch2c')(relu15_3)
    bn15_3   = BatchNormalization(name='bn4b4_branch2c')(conv15_3)

    merge16  = Add()([relu15_1, bn15_3])
    relu16_1 = Activation('relu', name='res4b4_relu')(merge16)
    conv16_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b5_branch2a')(relu16_1)
    bn16_1   = BatchNormalization(name='bn4b5_branch2a')(conv16_1)
    relu16_2 = Activation('relu')(bn16_1)
    conv16_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b5_branch2b')(relu16_2)
    bn16_2   = BatchNormalization(name='bn4b5_branch2b')(conv16_2)
    relu16_3 = Activation('relu')(bn16_2)
    conv16_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b5_branch2c')(relu16_3)
    bn16_3   = BatchNormalization(name='bn4b5_branch2c')(conv16_3)

    merge17  = Add()([relu16_1, bn16_3])
    relu17_1 = Activation('relu', name='res4b5_relu')(merge17)
    conv17_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b6_branch2a')(relu17_1)
    bn17_1   = BatchNormalization(name='bn4b6_branch2a')(conv17_1)
    relu17_2 = Activation('relu')(bn17_1)
    conv17_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b6_branch2b')(relu17_2)
    bn17_2   = BatchNormalization(name='bn4b6_branch2b')(conv17_2)
    relu17_3 = Activation('relu')(bn17_2)
    conv17_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b6_branch2c')(relu17_3)
    bn17_3   = BatchNormalization(name='bn4b6_branch2c')(conv17_3)

    merge18  = Add()([relu17_1, bn17_3])
    relu18_1 = Activation('relu', name='res4b6_relu')(merge18)
    conv18_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b7_branch2a')(relu18_1)
    bn18_1   = BatchNormalization(name='bn4b7_branch2a')(conv18_1)
    relu18_2 = Activation('relu')(bn18_1)
    conv18_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b7_branch2b')(relu18_2)
    bn18_2   = BatchNormalization(name='bn4b7_branch2b')(conv18_2)
    relu18_3 = Activation('relu')(bn18_2)
    conv18_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b7_branch2c')(relu18_3)
    bn18_3   = BatchNormalization(name='bn4b7_branch2c')(conv18_3)

    merge19  = Add()([relu18_1, bn18_3])
    relu19_1 = Activation('relu', name='res4b7_relu')(merge19)
    conv19_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b8_branch2a')(relu19_1)
    bn19_1   = BatchNormalization(name='bn4b8_branch2a')(conv19_1)
    relu19_2 = Activation('relu')(bn19_1)
    conv19_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b8_branch2b')(relu19_2)
    bn19_2   = BatchNormalization(name='bn4b8_branch2b')(conv19_2)
    relu19_3 = Activation('relu')(bn19_2)
    conv19_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b8_branch2c')(relu19_3)
    bn19_3   = BatchNormalization(name='bn4b8_branch2c')(conv19_3)

    merge20  = Add()([relu19_1, bn19_3])
    relu20_1 = Activation('relu', name='res4b8_relu')(merge20)
    conv20_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b9_branch2a')(relu20_1)
    bn20_1   = BatchNormalization(name='bn4b9_branch2a')(conv20_1)
    relu20_2 = Activation('relu')(bn20_1)
    conv20_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b9_branch2b')(relu20_2)
    bn20_2   = BatchNormalization(name='bn4b9_branch2b')(conv20_2)
    relu20_3 = Activation('relu')(bn20_2)
    conv20_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b9_branch2c')(relu20_3)
    bn20_3   = BatchNormalization(name='bn4b9_branch2c')(conv20_3)

    merge21  = Add()([relu20_1, bn20_3])
    relu21_1 = Activation('relu', name='res4b9_relu')(merge21)
    conv21_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b10_branch2a')(relu21_1)
    bn21_1   = BatchNormalization(name='bn4b10_branch2a')(conv21_1)
    relu21_2 = Activation('relu')(bn21_1)
    conv21_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b10_branch2b')(relu21_2)
    bn21_2   = BatchNormalization(name='bn4b10_branch2b')(conv21_2)
    relu21_3 = Activation('relu')(bn21_2)
    conv21_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b10_branch2c')(relu21_3)
    bn21_3   = BatchNormalization(name='bn4b10_branch2c')(conv21_3)

    merge22  = Add()([relu21_1, bn21_3])
    relu22_1 = Activation('relu', name='res4b10_relu')(merge22)
    conv22_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b11_branch2a')(relu22_1)
    bn22_1   = BatchNormalization(name='bn4b11_branch2a')(conv22_1)
    relu22_2 = Activation('relu')(bn22_1)
    conv22_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b11_branch2b')(relu22_2)
    bn22_2   = BatchNormalization(name='bn4b11_branch2b')(conv22_2)
    relu22_3 = Activation('relu')(bn22_2)
    conv22_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b11_branch2c')(relu22_3)
    bn22_3   = BatchNormalization(name='bn4b11_branch2c')(conv22_3)

    merge23  = Add()([relu22_1, bn22_3])
    relu23_1 = Activation('relu', name='res4b11_relu')(merge23)
    conv23_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b12_branch2a')(relu23_1)
    bn23_1   = BatchNormalization(name='bn4b12_branch2a')(conv23_1)
    relu23_2 = Activation('relu')(bn23_1)
    conv23_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b12_branch2b')(relu23_2)
    bn23_2   = BatchNormalization(name='bn4b12_branch2b')(conv23_2)
    relu23_3 = Activation('relu')(bn23_2)
    conv23_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b12_branch2c')(relu23_3)
    bn23_3   = BatchNormalization(name='bn4b12_branch2c')(conv23_3)

    merge24  = Add()([relu23_1, bn23_3])
    relu24_1 = Activation('relu', name='res4b12_relu')(merge24)
    conv24_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b13_branch2a')(relu24_1)
    bn24_1   = BatchNormalization(name='bn4b13_branch2a')(conv24_1)
    relu24_2 = Activation('relu')(bn24_1)
    conv24_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b13_branch2b')(relu24_2)
    bn24_2   = BatchNormalization(name='bn4b13_branch2b')(conv24_2)
    relu24_3 = Activation('relu')(bn24_2)
    conv24_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b13_branch2c')(relu24_3)
    bn24_3   = BatchNormalization(name='bn4b13_branch2c')(conv24_3)

    merge25  = Add()([relu24_1, bn24_3])
    relu25_1 = Activation('relu', name='res4b13_relu')(merge25)
    conv25_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b14_branch2a')(relu25_1)
    bn25_1   = BatchNormalization(name='bn4b14_branch2a')(conv25_1)
    relu25_2 = Activation('relu')(bn25_1)
    conv25_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b14_branch2b')(relu25_2)
    bn25_2   = BatchNormalization(name='bn4b14_branch2b')(conv25_2)
    relu25_3 = Activation('relu')(bn25_2)
    conv25_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b14_branch2c')(relu25_3)
    bn25_3   = BatchNormalization(name='bn4b14_branch2c')(conv25_3)

    merge26  = Add()([relu25_1, bn25_3])
    relu26_1 = Activation('relu', name='res4b14_relu')(merge26)
    conv26_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b15_branch2a')(relu26_1)
    bn26_1   = BatchNormalization(name='bn4b15_branch2a')(conv26_1)
    relu26_2 = Activation('relu')(bn26_1)
    conv26_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b15_branch2b')(relu26_2)
    bn26_2   = BatchNormalization(name='bn4b15_branch2b')(conv26_2)
    relu26_3 = Activation('relu')(bn26_2)
    conv26_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b15_branch2c')(relu26_3)
    bn26_3   = BatchNormalization(name='bn4b15_branch2c')(conv26_3)

    merge27  = Add()([relu26_1, bn26_3])
    relu27_1 = Activation('relu', name='res4b15_relu')(merge27)
    conv27_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b16_branch2a')(relu27_1)
    bn27_1   = BatchNormalization(name='bn4b16_branch2a')(conv27_1)
    relu27_2 = Activation('relu')(bn27_1)
    conv27_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b16_branch2b')(relu27_2)
    bn27_2   = BatchNormalization(name='bn4b16_branch2b')(conv27_2)
    relu27_3 = Activation('relu')(bn27_2)
    conv27_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b16_branch2c')(relu27_3)
    bn27_3   = BatchNormalization(name='bn4b16_branch2c')(conv27_3)

    merge28  = Add()([relu27_1, bn27_3])
    relu28_1 = Activation('relu', name='res4b16_relu')(merge28)
    conv28_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b17_branch2a')(relu28_1)
    bn28_1   = BatchNormalization(name='bn4b17_branch2a')(conv28_1)
    relu28_2 = Activation('relu')(bn28_1)
    conv28_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b17_branch2b')(relu28_2)
    bn28_2   = BatchNormalization(name='bn4b17_branch2b')(conv28_2)
    relu28_3 = Activation('relu')(bn28_2)
    conv28_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b17_branch2c')(relu28_3)
    bn28_3   = BatchNormalization(name='bn4b17_branch2c')(conv28_3)

    merge29  = Add()([relu28_1, bn28_3])
    relu29_1 = Activation('relu', name='res4b17_relu')(merge29)
    conv29_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b18_branch2a')(relu29_1)
    bn29_1   = BatchNormalization(name='bn4b18_branch2a')(conv29_1)
    relu29_2 = Activation('relu')(bn29_1)
    conv29_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b18_branch2b')(relu29_2)
    bn29_2   = BatchNormalization(name='bn4b18_branch2b')(conv29_2)
    relu29_3 = Activation('relu')(bn29_2)
    conv29_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b18_branch2c')(relu29_3)
    bn29_3   = BatchNormalization(name='bn4b18_branch2c')(conv29_3)

    merge30  = Add()([relu29_1, bn29_3])
    relu30_1 = Activation('relu', name='res4b18_relu')(merge30)
    conv30_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b19_branch2a')(relu30_1)
    bn30_1   = BatchNormalization(name='bn4b19_branch2a')(conv30_1)
    relu30_2 = Activation('relu')(bn30_1)
    conv30_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b19_branch2b')(relu30_2)
    bn30_2   = BatchNormalization(name='bn4b19_branch2b')(conv30_2)
    relu30_3 = Activation('relu')(bn30_2)
    conv30_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b19_branch2c')(relu30_3)
    bn30_3   = BatchNormalization(name='bn4b19_branch2c')(conv30_3)

    merge31  = Add()([relu30_1, bn30_3])
    relu31_1 = Activation('relu', name='res4b19_relu')(merge31)
    conv31_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b20_branch2a')(relu31_1)
    bn31_1   = BatchNormalization(name='bn4b20_branch2a')(conv31_1)
    relu31_2 = Activation('relu')(bn31_1)
    conv31_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b20_branch2b')(relu31_2)
    bn31_2   = BatchNormalization(name='bn4b20_branch2b')(conv31_2)
    relu31_3 = Activation('relu')(bn31_2)
    conv31_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b20_branch2c')(relu31_3)
    bn31_3   = BatchNormalization(name='bn4b20_branch2c')(conv31_3)

    merge32  = Add()([relu31_1, bn31_3])
    relu32_1 = Activation('relu', name='res4b20_relu')(merge32)
    conv32_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b21_branch2a')(relu32_1)
    bn32_1   = BatchNormalization(name='bn4b21_branch2a')(conv32_1)
    relu32_2 = Activation('relu')(bn32_1)
    conv32_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b21_branch2b')(relu32_2)
    bn32_2   = BatchNormalization(name='bn4b21_branch2b')(conv32_2)
    relu32_3 = Activation('relu')(bn32_2)
    conv32_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b21_branch2c')(relu32_3)
    bn32_3   = BatchNormalization(name='bn4b21_branch2c')(conv32_3)

    merge33  = Add()([relu32_1, bn32_3])
    relu33_1 = Activation('relu', name='res4b21_relu')(merge33)
    conv33_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b22_branch2a')(relu33_1)
    bn33_1   = BatchNormalization(name='bn4b22_branch2a')(conv33_1)
    relu33_2 = Activation('relu')(bn33_1)
    conv33_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b22_branch2b')(relu33_2)
    bn33_2   = BatchNormalization(name='bn4b22_branch2b')(conv33_2)
    relu33_3 = Activation('relu')(bn33_2)
    conv33_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b22_branch2c')(relu33_3)
    bn33_3   = BatchNormalization(name='bn4b22_branch2c')(conv33_3)

    merge34  = Add()([relu33_1, bn33_3])
    relu34_1 = Activation('relu', name='res4b22_relu')(merge34)
    conv34_1 = Conv2D(filters=2048, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5a_branch1')(relu34_1)
    bn34_1   = BatchNormalization(name='bn5a_branch1')(conv34_1)

    conv35_1 = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5a_branch2a')(relu34_1)
    bn35_1   = BatchNormalization(name='bn5a_branch2a')(conv35_1)
    relu35_2 = Activation('relu')(bn35_1)
    conv35_2 = Conv2D(filters=512, kernel_size=3, dilation_rate=(4,4), use_bias=False, padding='same', name='res5a_branch2b')(relu35_2)
    bn35_2   = BatchNormalization(name='bn5a_branch2b')(conv35_2)
    relu35_3 = Activation('relu')(bn35_2)
    conv35_3 = Conv2D(filters=2048, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5a_branch2c')(relu35_3)
    bn35_3   = BatchNormalization(name='bn5a_branch2c')(conv35_3)

    merge36  = Add()([bn34_1, bn35_3])
    relu36_1 = Activation('relu', name='res5a_relu')(merge36)
    conv36_1 = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5b_branch2a')(relu36_1)
    bn36_1   = BatchNormalization(name='bn5b_branch2a')(conv36_1)
    relu36_2 = Activation('relu')(bn36_1)
    conv36_2 = Conv2D(filters=512, kernel_size=3, dilation_rate=(4,4), use_bias=False, padding='same', name='res5b_branch2b')(relu36_2)
    bn36_2   = BatchNormalization(name='bn5b_branch2b')(conv36_2)
    relu36_3 = Activation('relu')(bn36_2)
    conv36_3 = Conv2D(filters=2048, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5b_branch2c')(relu36_3)
    bn36_3   = BatchNormalization(name='bn5b_branch2c')(conv36_3)

    merge37  = Add()([relu36_1, bn36_3])
    relu37_1 = Activation('relu', name='res5b_relu')(merge37)
    conv37_1 = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5c_branch2a')(relu37_1)
    bn37_1   = BatchNormalization(name='bn5c_branch2a')(conv37_1)
    relu37_2 = Activation('relu')(bn37_1)
    conv37_2 = Conv2D(filters=512, kernel_size=3, dilation_rate=(4,4), use_bias=False, padding='same', name='res5c_branch2b')(relu37_2)
    bn37_2   = BatchNormalization(name='bn5c_branch2b')(conv37_2)
    relu37_3 = Activation('relu')(bn37_2)
    conv37_3 = Conv2D(filters=2048, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5c_branch2c')(relu37_3)
    bn37_3   = BatchNormalization(name='bn5c_branch2c')(conv37_3)

    merge38  = Add()([relu37_1, bn37_3])
    relu38_1 = Activation('relu', name='res5c_relu')(merge38)
    conv38_1 = Conv2D(filters=NUM_CLASSES, kernel_size=3, dilation_rate=(6,6), padding='same', name='fc1_voc12_c0')(relu38_1)
    conv38_2 = Conv2D(filters=NUM_CLASSES, kernel_size=3, dilation_rate=(12,12), padding='same', name='fc1_voc12_c1')(relu38_1)
    conv38_3 = Conv2D(filters=NUM_CLASSES, kernel_size=3, dilation_rate=(18,18), padding='same', name='fc1_voc12_c2')(relu38_1)
    conv38_4 = Conv2D(filters=NUM_CLASSES, kernel_size=3, dilation_rate=(24,24), padding='same', name='fc1_voc12_c3')(relu38_1)

    output   = Add(name='fc1_voc12')([conv38_1, conv38_2, conv38_3, conv38_4])
    output   = Reshape((output.shape[1].value*output.shape[1].value,NUM_CLASSES))(output)
    #output   = Activation('softmax')(output)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    """
    print('\n\n')
    print('-------------------------------------------------------')
    print('--------------- Trainable parameters ------------------')
    print('-------------------------------------------------------')
    total_params = 0
    for v in tf.trainable_variables():
        #pdb.set_trace()
        shape = v.get_shape()
        params = 1
        for dim in shape:
            params *= dim.value
        print('{:<30s}: {:<20s}\t{:<10s}'.format(v.name, str(shape), str(params)))
        total_params += params
    print('total_pamars = {}'.format(total_params))
    print('-------------------------------------------------------\n\n')
    """


    """
    #################################################################
    ### Exporting the trained weights into a dictionary
    ### This piece of code is run a Tensorflow program where we have trained paramaters.
    #################################################################
    voc_trained_weights = {}
    for v in tf.trainable_variables():
        wname = v.name.encode('ascii','ignore')
        wname = wname.replace('weights','kernel')
        wname = wname.replace('biases','bias')
        voc_trained_weights[wname] = sess.run(v)

    for v in tf.model_variables():
        wname = v.name.encode('ascii','ignore')
        voc_trained_weights[wname] = sess.run(v)

    np.save('voc_traineded_weights.npy', voc_trained_weights)
    #################################################################
    """

    #################################################################
    ### Writing the already trained and saved parameters into the keras weights
    #################################################################
    voc_trained_weights = np.load('voc_trained_weights.npy')[()]
    for l in model.layers:
        print('{:<20s}: {:<25s} -> {}'.format(l.name, l.input_shape,l.output_shape))
        trainable_weights = l.trainable_weights
        if not trainable_weights:
            continue
        len_w = len(trainable_weights)
        old_weights = l.get_weights()
        weights = []
        for i in range(len_w):
            wname = trainable_weights[i].name.encode('ascii','ignore')
            weights.append(voc_trained_weights[wname])
        if len(old_weights)>2:
            wnames = wname.split('/')
            wname = wnames[0] + '/' + 'moving_mean:0'
            weights.append(voc_trained_weights[wname])
            wname = wnames[0] + '/' + 'moving_variance:0'
            weights.append(voc_trained_weights[wname])
        l.set_weights(weights)
    print('Trained VOC12 weights were loaded into the Keras model.')

    ################################################################
    ### Test Results
    ################################################################
    BATCH_SIZE = 2
    accuracy = model.evaluate(x=val_images, y=val_labels, batch_size=BATCH_SIZE)
    print('\n')
    print('Keras test score = {}'.format(accuracy))
    print('\n')

    """
    #################################################################
    ### Sample visualization
    #################################################################
    # The following can not be run due to out-of-memory error, since we did not use minibatches
    predictions = model.predict(val_images)
    pdb.set_trace()
    predictions = np.argmax(predictions, axis=-1)
    for i in range(Nv):
        pred = np.reshape(predictions[i,:], (40,40))
        target = np.reshape(val_labels[i,:,:], (40,40))
        pdb.set_trace()
    """

if __name__ == '__main__':
    main()
