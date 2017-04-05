"""Train a classifier on the Kaggle DSB2017 dataset"""
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import keras.callbacks

INPUT_SZ = 48
# TODO: check with our initialized models!
PATH_UNET = "/home/omer/unet.hdf5"


def _get_model(optimizer, do_batch_norm):
    """Initializes a 3D U-Net.

    Based on https://github.com/booz-allen-hamilton/DSB3Tutorial/blob/master/tutorial_code/LUNA_train_unet.py
    Loads the weights for initializing a degenerate 3D convolutional network.

    Args:
        optimizer:
        do_batch_norm:

    Returns:
        A U-Net model with degenerate 3-D convolutions. The input tensor has
        dimensions (B, H, W, N_SLICES x N_DETECTIONS, 1), where:
            N_SLICES: number of slices for 2.5D (see `slice_cube()`)
            N_DETECTIONS: number of detections per patient (probably in [1,20])
    """

    # TODO: batch normalization when required

    # set up the architecture
    inputs = Input(INPUT_SZ, INPUT_SZ, None, 1)
  
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
    dropout = Dropout(0.9)(maxpool_det)
    fc = Dense(1, activation="sigmoid")(dropout)

    # TODO: sum + max pooling + 2x1 FC layer

    model = Model(inputs=inputs, outputs=fc)

    # load the weights
    with h5py.File(PATH_UNET, "r") as fh5:

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

    model.compile(optimizer=optimizer, loss="binary_crossentropy",
                metrics=['accuracy'])

    return model

def _slice_cube(cube):
    """Creates 2D slices from a 3D volume.

    Args:
        cube: a [N x N x N] numpy array

    Returns:
        slices: a [N x N x 9] array of 2D slices
    """

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

def _normalize_hu(image):
    # From https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    PIXEL_MEAN = 0.25    
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return (image - PIXEL_MEAN)

def _sample_generator(samples, batch_sz=8):
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

            # reshuffle when necessary
            if len(inds_shuffled) < batch_sz:
                inds_shuffled = np.random.permutation(len(samples))

            # figure out batch's maximal 3-rd dim size
            max_sz = 0
            for i in range(0, batch_sz):
                max_sz = max(fh5[samples[inds_shuffled[i]][0]].shape[-1],
                             max_size)

            # load the data from file (with augmentation and permutation)
            data = np.zeros((batch_sz, INPUT_SZ, INPUT_SZ, max_sz), np.float32)
            for i in range(0, batch_sz):
                cube_4d = fh5.get(samples[inds_shuffled[i]][0]).value
                for i_data, i_cube in np.random.permutation(cube_4d.shape[-1]):

                    crop_inds = np.random.randint(0, cube_4d.shape[0] -
                                                  INPUT_SZ, 3)
                    cube = cube_4d[crop_inds[0]:crop_inds[0] + INPUT_SZ,
                                   crop_inds[1]:crop_inds[1] + INPUT_SZ,
                                   crop_inds[2]:crop_inds[2] + INPUT_SZ,
                                   i_cube]
                    data[i, :, :, i_data] = _normalize_hu(_slice_cube(cube))
            
                
            labels = [samples[i][1] for i in inds_shuffled[0:batch_sz]]
            inds_shuffled = np.delete(inds_shuffled, range(0, batch_sz))

            yield np.expand_dims(data, axis=5), labels

def _train(train, val, do_batch_norm, optimizer, lr_scheduler, path_session):
    """Execute a single training task.


    Returns:
        model: /path/to/best_model as measured by validation's loss
        loss: the loss computed on the validation set
        acc: the accuracy computed on the validation set
    """

    session_id = os.path.basename(path_session)
    model_cp = keras.callbacks.ModelCheckpoint(
        os.path.join(path_session, "{}.hdf5".format(session_id)),
        monitor="val_loss",
        save_best_only=True)
    

    # train
    model = _get_model(optimizer, do_batch_norm)
    history = model.fit_generator(
        _sample_generator(train, batch_sz),
        steps_per_epoch=int(len(train) / batch_sz),
        epochs=epochs,
        validation_data=_sample_generator(val, 1),
        validation_steps=len(val),
        callbacks=[model_cp, lr_scheduler],
        verbose=1)

    # plot training curves
    def plot_history(metric):
        #plt.ioff()
        str_metric = "accuracy" if metric == "acc" else "loss"
        plt.plot(history.history[metric])
        plt.plot(history.history["val_{}".format(metric)])
        plt.title("model {}".format(str_metric))
        plt.ylabel(str_metric)
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig(os.path.join(output_dir_ver, "{}.png".format(str_metric)))
    
    # summarize history for loss
    plot_history("loss")
    plot_history("acc")
    
    # output model and performance measures
    ind_min_loss = np.argmin(history.history["val_loss"])
    return (os.path.join(path_session, "{}.hdf5".format(session_id)),
            history.history["val_loss"][ind_min_loss],
            history.history["val_acc"][ind_min_loss])



def split_train_val(path_train_labels, seed):
    """Splits the training data to training and validation sets.

    Args:
        path_train_labels: /path/to/train_labels.csv

    Returns:
        train, val: training and validation sets, list of (id, label)
    """

    random.seed(seed)

    with open(path_train_labels, "r") as f:
        reader = csv.reader(f)
        samples = list(reader)
    
    # stratified split to training and validation sets
    list_0 = [(sample[0], 0) for sample in samples if sample[1] == "0"]
    list_1 = [(sample[0], 1) for sample in samples if sample[1] == "1"]
    train = (random.sample(list_0, int(np.floor(ratio_train * len(list_0)))) +
        random.sample(list_1, int(np.floor(ratio_train * len(list_1)))))
    val = list((set(list_0) | set(list_1)) - set(train))

    return train, val

def make_lr_scheduler(base_lr, decay_rate, epoch_rate):

    def lr_schedule(epoch):
        if epoch + 1 < epoch_rate:
            lr = base_lr
        else:
            lr = base_lr / (decay_rate * np.floor(epoch + 1 / rate_epochs))
            
        return lr

    return keras.callbacks.LearningRateScheduler(lr_schedule)

def train(train, val, path_data, path_session, hyper_param, seed):
    """Train models per specified hyper param.

    Args:
        train, val: training and validation sets from `split_train_val()`
        path_data: /path/to/train_detections.hdf5
        path_session: string specifying the session's output path
        hyper_param: dictionary with entries as follows -
                        * epochs: number of epochs
                        * batch_sz: batch size in training
                        * batch_norm: do batch normalization?
                        * optimizer: a keras.optimizers beast
                        * lr_scheduler: a keras.callback.LearningRateScheduler

    """

    for batch_sz in hyper_param["batch_sz"]:
        for batch_norm in hyper_param["batch_norm"]:
            for optimizer in hyper_param["optimizers"]:
                for lr_scheduler_param in hyper_param["lr_scheduler_param"]:

                    models_i.append(kaggle.classifier.train(
                        train,
                        val,
                        os.path.join(PATH_DATASETS, "detections_train.hdf5"),
                        path_session,
                        hyper_param)

    # train
    models = []
    for i, do_batch_norm in enumerate(batch_norm):
        for j, optimizer in enumerate(optimizers):
            models.append(_train(train, val, do_batch_norm, optimizer,
                                 lr_scheduler, "{}_{}_{}".format(path_session, i, j)))
    
    # sort by validation loss
    return models.sort(key=lambda tuple: tuple[1])
