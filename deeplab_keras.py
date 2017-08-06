
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
from resnet50_seg_model import resnet50_deeplab
from load_dataset import load_dataset
from scipy.misc import imresize
from sklearn.metrics import confusion_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NUM_CLASSES = 21
cwd = os.getcwd()
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
H, W = 321, 321

def main():
    ################################################
    ########## Preparing the dataset ###############
    ################################################
    training_images, training_labels = load_dataset('train_limited', (40,40)) # The output of this resnet model is (40,40)
    val_images, val_labels = load_dataset('val', (H,W))
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
    #val_labels = np.reshape(val_labels, (Nv, H*W, 1))

    ################################################
    ######## Building the model ####################
    ################################################
    model = resnet50_deeplab()

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
    print('\nValidation step:\n')
    BATCH_SIZE = 2
    #accuracy = model.evaluate(x=val_images, y=val_labels, batch_size=BATCH_SIZE)
    #print('\n')
    #print('Keras test score = {}'.format(accuracy))
    #print('\n')

    #################################################################
    ### Sample visualization
    #################################################################
    MULTI_SCALE = 0
    conf_mat_total = np.zeros((NUM_CLASSES,NUM_CLASSES))
    pbar = InitBar()
    for i in range(Nv):
        pbar(100.0*float(i)/float(Nv))
        img = val_images[i]
        """ Strangely single-scale works a little better.
            I have even tried multi-scale with merging in (40,40) resolution
            and then resizing the result, but even though it boosted
            the performance 0.2%, it was still lower than the Single-scale architecture. """
        if not MULTI_SCALE:
            prediction100 = model.predict(np.expand_dims(img, axis=0))
            prediction    = np.reshape(prediction100, (H*W, NUM_CLASSES))
        else:
            img075 = cv2.resize(img, (int(0.75*H),int(0.75*W)), cv2.INTER_CUBIC)
            img050 = cv2.resize(img, (int(0.50*H),int(0.50*W)), cv2.INTER_CUBIC)
            prediction100 = model.predict(np.expand_dims(img, axis=0))
            prediction075 = model.predict(np.expand_dims(img075, axis=0))
            prediction050 = model.predict(np.expand_dims(img050, axis=0))

            prediction100 = np.reshape(prediction100, (H*W, NUM_CLASSES))
            prediction075 = np.reshape(prediction075, (H*W, NUM_CLASSES))
            prediction050 = np.reshape(prediction050, (H*W, NUM_CLASSES))
            prediction = np.maximum(prediction100, prediction075, prediction050)

        prediction = np.argmax(prediction, axis=-1)
        pred_img = np.reshape(prediction, (H,W))
        gt_img = val_labels[i]
        gt = gt_img[gt_img>=0]
        pred = pred_img[gt_img>=0]
        conf_mat = confusion_matrix(gt, pred, labels=list(range(NUM_CLASSES))) 
        conf_mat_total += conf_mat

    ious = np.zeros((NUM_CLASSES,1))
    for l in range(NUM_CLASSES):
        ious[l] = conf_mat_total[l,l] / (np.sum(conf_mat_total[l,:]) +
                                         np.sum(conf_mat_total[:,l]) -
                                         conf_mat_total[l,l])
    
    print(ious)
    print('Mean IOU = {}\n'.format(np.mean(ious)))

if __name__ == '__main__':
    main()
