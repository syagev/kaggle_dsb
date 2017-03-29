import keras
import h5py

import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv3D, MaxPooling3D, Lambda, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import matplotlib.pyplot as plt
import csv
import random
import os
import skimage.io

PATH_LABELS_CSV = "C:/DATA/projects/kaggle/data/stage1_labels.csv"
PATH_DETECTIONS_CSV = "/path/to.csv"
PATH_PROCESSED_CT = "/razberry/datasets/kaggle-dsb2017/stage1_processed"
PATH_PROCESSED_H5 = "/razberry/datasets/kaggle-dsb2017/stage1_rois.h5"
PATH_UNET = "/home/omer/unet.hdf5"
RAND_SEED = 13
INPUT_SHAPE = (48, 48,)

import kaggle.process_luna as kgluna    # for luna_generator, get_candidates

#--------------- MODEL
def get_unet(file_weights=PATH_UNET):
    """Initializes a 3D U-Net.

    Based on https://github.com/booz-allen-hamilton/DSB3Tutorial/blob/master/tutorial_code/LUNA_train_unet.py
    Loads the weights for initializing a degenerate 3D convolutional network.

    Args:
        fili_weights: /path/to the weights file

    Returns:
        The U-Net model with degenerate 3-D convolutions. The input tensor will
        be of dimension (B, H, W, N_SLICES x N_DETECTIONS, 1), where:
            N_SLICES: represents the number of slices for 2.5D (probably 9,
                      see `slice_cube()`
            N_DETECTIONS: number of detections per patient (probably in [1,20])
    """

    with h5py.File(PATH_UNET, "r") as fh5:

        # set up the architecture
        inputs = Input(INPUT_SHAPE + (None, 1))
  
        # in: (B, 48, 48, N_SLICES x N_DETECTIONS, 1)
        conv1 = Conv3D(32, (3, 3, 1), activation="relu", padding="same")(inputs)
        conv1 = Conv3D(32, (3, 3, 1), activation="relu", padding="same")(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

        # in: (B, 24, 24, N_SLICES x N_DETECTIONS, 32)
        conv2 = Conv3D(64, (3, 3, 1), activation="relu", padding="same")(pool1)
        conv2 = Conv3D(64, (3, 3, 1), activation="relu", padding="same")(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 3))(conv2)

        # in: (B, 12, 12, N_SLICES/3 x N_DETECTIONS, 64)
        conv3 = Conv3D(128, (3, 3, 1), activation="relu", padding="same")(pool2)
        conv3 = Conv3D(128, (3, 3, 1), activation="relu", padding="same")(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 3))(conv3)

        # in: (B, 6, 6, N_DETECTIONS, 128)
        conv4 = Conv3D(256, (3, 3, 1), activation="relu", padding="same")(pool3)
        conv4 = Conv3D(256, (3, 3, 1), activation="relu", padding="same")(conv4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 1))(conv4)

        # in: (B, 3, 3, N_DETECTIONS, 256)
        conv5 = Conv3D(512, (3, 3, 1), activation="relu", padding="same")(pool4)
        conv5 = Conv3D(512, (3, 3, 1), activation="relu", padding="same")(conv5)
        pool5 = MaxPooling3D(pool_size=(3, 3, 1))(conv5)

        # in: (B, 1, 1, N_DETECTIONS, 512)
        maxpool_det = Lambda((lambda x: K.max(x, axis=3)))(pool5)
        maxpool_det = Flatten()(maxpool_det)

        # in: (B, 512)
        dropout = Dropout(0.5)(maxpool_det)
        fc = Dense(1, activation="sigmoid")(dropout)

        model = Model(inputs=inputs, outputs=fc)

        # load the weights
        weights = fh5["model_weights"]
        inds_conv_layers = [i for i, layer in enumerate(model.layers)
                            if layer.get_config()["name"][0:4] == "conv"]
        for i in range(1, 11):
            # load weights
            w_i = weights["convolution2d_{}".format(i)]
            W = w_i["convolution2d_{}_W:0".format(i)].value
            W = np.expand_dims(W.transpose((2, 3, 1, 0)), 2) 
            b = w_i["convolution2d_{}_b:0".format(i)].value

            # set the weights
            model.layers[inds_conv_layers[i-1]].set_weights([W, b])

        model.compile(optimizer=Adam(lr=1e-5), loss="binary_crossentropy",
                  metrics=['accuracy'])

    return model

#--------------- DATA


# From https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25    
def normalize_hu(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return (image - PIXEL_MEAN)

def gen_simulated_data(list_data):

    if os.path.isfile(PATH_PROCESSED_H5):
        os.remove(PATH_PROCESSED_H5)

    with h5py.File(PATH_PROCESSED_H5, "w") as fh5:
        for item in list_data:
            fh5.create_dataset(item[0], dtype=np.int16,
                               shape=INPUT_SHAPE + (18,),
                               data=np.random.randint(-2000, 600,
                                                      INPUT_SHAPE + (18,),
                                                      dtype=np.int16))

def kaggle_generator(samples, batch_size=8):
    """Generates batches for training and validation.

    Args:
        samples: list of (patient_id, label) for sampling
        batch_size: required batch size (default 8)

    Returns:
        (data, labels): batch with data and labels
    """

    np.random.seed(RAND_SEED)
    inds_shuffled = np.random.permutation(len(samples))
    
    with h5py.File(PATH_PROCESSED_H5, "r") as fh5:
        while True:

            if len(inds_shuffled) < batch_size:
                inds_shuffled = np.random.permutation(len(samples))

            data = [normalize_hu(fh5.get(samples[i][0]).value)
                    for i in inds_shuffled[0:batch_size]]
            labels = [samples[i][1] for i in inds_shuffled[0:batch_size]]
            inds_shuffled = np.delete(inds_shuffled, range(0, batch_size))

            yield np.expand_dims(np.stack(data, axis=0), axis=5), labels

def luna_generator(d0, d1, batch_size=8, ids=None):
    """Generates batches for training and validation from LUNA2016 set.

    Args:
        d0, d1: dictionaries for non-class/class resp. key is patient id
        batch_size: required batch size (default 8)
        ids: use these scan IDs (for train/val split)

    Returns:
        (data, labels): batch with data and labels
    """

    np.random.seed(RAND_SEED)

    # construct the samples tuples (patient_id, label, idx)
    def load_samples(d, label):
        with h5py.File(os.path.join(kgluna.PATH_OUTPUT, \
                                    "{}.h5".format(label))) as f:
            return [(entry, label, i)
                    for entry in d if entry in ids
                    for i in range(0, f[entry].shape[-1])]
    samples = load_samples(d0, 0) + load_samples(d1, 1)

    inds_shuffled = np.random.permutation(len(samples))
    
    with h5py.File(os.path.join(kgluna.PATH_OUTPUT, "0.h5"), "r") as f0:
        with h5py.File(os.path.join(kgluna.PATH_OUTPUT, "1.h5"), "r") as f1:
            while True:

                if len(inds_shuffled) < batch_size:
                    inds_shuffled = np.random.permutation(len(samples))
            
                data = np.zeros((batch_size, ) + INPUT_SHAPE + (9,),
                                dtype = np.float32)
                for i in range(0, batch_size):
                    sample = samples[inds_shuffled[i]]
                    f = f0 if sample[1] == 0 else f1
                    cube = f.get(sample[0]).value[:, :, :, sample[2]]
                    crop_inds = np.random.randint(0, kgluna.SZ_CUBE -
                                                  INPUT_SHAPE[0], 3)
                    cube = cube[crop_inds[0]:crop_inds[0] + INPUT_SHAPE[0],
                                crop_inds[1]:crop_inds[1] + INPUT_SHAPE[0],
                                crop_inds[2]:crop_inds[2] + INPUT_SHAPE[0]]
                    data[i, :, :, :] = normalize_hu(slice_cube(cube))
                labels = [samples[i][1] for i in inds_shuffled[0:batch_size]]
                inds_shuffled = np.delete(inds_shuffled, range(0, batch_size))

                yield np.expand_dims(data, axis=5), labels

def slice_cube(cube):

    slices = np.zeros((cube.shape[0], cube.shape[0], 9), dtype=np.float32)

    # axis-aligned
    slices[:,:,0] = cube[np.floor(cube.shape[0] / 2).astype(int), :, :]
    slices[:,:,1] = cube[:, np.floor(cube.shape[0] / 2).astype(int), :]
    slices[:,:,2] = cube[:, :, np.floor(cube.shape[0] / 2).astype(int)]

    # diagonals
    slices[:,:,3] = cube.diagonal(axis1=0, axis2=1)
    slices[:,:,4] = cube.diagonal(axis1=0, axis2=2)
    slices[:,:,5] = cube.diagonal(axis1=1, axis2=2)
    slices[:,:,6] = np.flip(cube, 0).diagonal(axis1=0, axis2=1)
    slices[:,:,7] = np.flip(cube, 0).diagonal(axis1=0, axis2=2)
    slices[:,:,8] = np.flip(cube, 1).diagonal(axis1=1, axis2=2)

    return slices

def split_train_val(ratio_train=0.7):
    """Splits the training data to train and validations sets.
    
    Optionals:
        ratio_train: the training size in percentage

    Returns:
        train, val - lists of training and validation samples
    """

    random.seed(RAND_SEED)

    with open(PATH_LABELS_CSV, "r") as f:
        reader = csv.reader(f)
        samples = list(reader)
        
    # remove the header
    del samples[0]
    list_0 = [(sample[0], 0) for sample in samples if sample[1] == "0"]
    list_1 = [(sample[0], 1) for sample in samples if sample[1] == "1"]

    train = (random.sample(list_0, int(np.floor(ratio_train * len(list_0)))) +
        random.sample(list_1, int(np.floor(ratio_train * len(list_1)))))
    val = list((set(list_0) | set(list_1)) - set(train))

    return train, val

def process_detections():

    with open(PATH_DETECTIONS_CSV, "r") as f:
        reader = csv.reader(f)
        detections = list(reader)
    
    with h5py.File(PATH_PROCESSED_H5, 'w') as fh5:
        # iterate detections and create the ROI database
        for patient in detections:
            pass
