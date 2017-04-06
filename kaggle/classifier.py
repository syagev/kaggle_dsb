"""Train a classifier on the Kaggle DSB2017 dataset"""
import random
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
import keras.backend as K
import keras.callbacks
from keras.layers import Input, Conv3D, MaxPool3D, Lambda, Flatten, Dropout, Dense, BatchNormalization, Concatenate
import csv
import h5py
import threading
import pickle
import sys

INPUT_SZ = 48
N_SLICES = 9
# TODO: check with our initialized models!
PATH_UNET = "/home/omer/unet.hdf5"



def _get_model(optimizer, do_batch_norm, pool_type="max", dropout_rate=0.5):
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

    # set up the architecture
    inputs = Input((INPUT_SZ, INPUT_SZ, None, 1))
  
    # in: (B, 48, 48, N_SLICES x N_DETECTIONS, 1)
    conv1 = Conv3D(32, (3, 3, 1), activation="relu", padding="same")(inputs)
    conv1 = Conv3D(32, (3, 3, 1), activation="relu", padding="same")(conv1)
    pool1 = MaxPool3D(pool_size=(2, 2, 1))(conv1)

    # in: (B, 24, 24, N_SLICES x N_DETECTIONS, 32)
    conv2 = Conv3D(64, (3, 3, 1), activation="relu", padding="same")(pool1)
    conv2 = Conv3D(64, (3, 3, 1), activation="relu", padding="same")(conv2)
    pool2 = MaxPool3D(pool_size=(2, 2, 3))(conv2)

    # in: (B, 12, 12, N_SLICES/3 x N_DETECTIONS, 64)
    conv3 = Conv3D(128, (3, 3, 1), activation="relu", padding="same")(pool2)
    conv3 = Conv3D(128, (3, 3, 1), activation="relu", padding="same")(conv3)
    pool3 = MaxPool3D(pool_size=(2, 2, 3))(conv3)

    # in: (B, 6, 6, N_DETECTIONS, 128)
    conv4 = Conv3D(256, (3, 3, 1), activation="relu", padding="same")(pool3)
    conv4 = Conv3D(256, (3, 3, 1), activation="relu", padding="same")(conv4)
    pool4 = MaxPool3D(pool_size=(2, 2, 1))(conv4)

    # in: (B, 3, 3, N_DETECTIONS, 256)
    conv5 = Conv3D(512, (3, 3, 1), activation="relu", padding="same")(pool4)
    conv5 = Conv3D(512, (3, 3, 1), activation="relu", padding="same")(conv5)
    pool5 = MaxPool3D(pool_size=(3, 3, 1))(conv5)

    # in: (B, 1, 1, N_DETECTIONS, 512)
    # # --- ugly fix, see https://github.com/fchollet/keras/issues/4609
    # def K_mean(x, **arguments):
        # from keras import backend as K
        # return K.mean(x, **arguments)
    # def K_max(x, **arguments):
        # from keras import backend as K
        # return K.max(x, **arguments)
    maxpool_det = Lambda((lambda x: K.max(x, axis=3)))(pool5)
    meanpool_det = Lambda((lambda x: K.mean(x, axis=3)))(pool5)
    if pool_type == "both":
        pool_det = Concatenate(-1)([maxpool_det, meanpool_det])   
    else:
        if pool_type == "max":
            pool_det = maxpool_det
        elif pool_type == "mean":
            pool_det = meanpool_det
    pool_det = Flatten()(pool_det)

    # in: (B, 512) for "max"/"mean" pool and (B, 1024) for "both"
    dropout = Dropout(dropout_rate)(pool_det)
    if pool_type == "both":
        fc = Dense(32, activation="sigmoid")(dropout)
        dropout = Dropout(dropout_rate)(fc)
        fc = Dense(1, activation="sigmoid")(dropout)
    else:
        fc = Dense(1, activation="sigmoid")(dropout)

    model = keras.models.Model(inputs=inputs, outputs=fc)

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

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.

    [om]: from https://github.com/fchollet/keras/issues/1638
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def _sample_generator(samples, path_data, batch_sz=8, mode="train"):
    """Generates batches for training and validation.

    Args:
        samples: list of (patient_id, label) for sampling
        batch_size: required batch size (default 8)

    Returns:
        (data, labels): batch with data and labels
    """
    if mode == "train":
        inds_shuffled = np.random.permutation(len(samples))
    elif mode == "predict":
        inds_shuffled = range(0, len(samples))
    
    with h5py.File(path_data, "r") as fh5:
        while True:

            # reshuffle when necessary
            if len(inds_shuffled) < batch_sz:
                inds_shuffled = np.random.permutation(len(samples))

            # figure out batch's maximal 3-rd dim size
            max_sz = 1 * N_SLICES
            for i in range(0, batch_sz):
                if samples[inds_shuffled[i]][0] in fh5:
                    max_sz = max(fh5[samples[inds_shuffled[i]][0]].shape[-1]
                                 * N_SLICES, max_sz)

            # load the data from file (with augmentation and permutation)
            data = np.zeros((batch_sz, INPUT_SZ, INPUT_SZ, max_sz), np.float32)
            labels = []
            for i in range(0, batch_sz):
                
                # 4D-cube (L, L, L, NUM_SAMPLES)
                if samples[inds_shuffled[i]][0] in fh5:
                    cube_4d = fh5.get(samples[inds_shuffled[i]][0]).value
                    
                    # cube_i is list of (L, L, max_sz) arrays                
                    cube_i = []
                    for i_cube in range(0, cube_4d.shape[-1]):

                        crop_inds = np.random.randint(0, cube_4d.shape[0] -
                                                      INPUT_SZ, 3)
                        # the "sliced" cube
                        cube_2d = cube_4d[crop_inds[0]:crop_inds[0] + INPUT_SZ,
                                          crop_inds[1]:crop_inds[1] + INPUT_SZ,
                                          crop_inds[2]:crop_inds[2] + INPUT_SZ,
                                          i_cube]

                        cube_i.append(_normalize_hu(_slice_cube(cube_2d)))
                    
                    cube_i = np.dstack(cube_i)
                    data[i, :, :, 0:cube_i.shape[2]] = cube_i

                    if mode == "train":
                        labels.append(samples[inds_shuffled[i]][1])
                
                else:
                    if mode == "train":
                        labels.append(0)
                
            inds_shuffled = np.delete(inds_shuffled, range(0, batch_sz))

            if mode == "train":
                output = (np.expand_dims(data, axis=5), labels)
            elif mode == "predict":
                output = np.expand_dims(data, axis=5)

            yield output

def train(trainset, valset, path_data, path_session, hyper_param):
    """Execute a single training task.


    Returns:
        model: /path/to/best_model as measured by validation's loss
        loss: the loss computed on the validation set
        acc: the accuracy computed on the validation set
    """

    session_id = os.path.basename(path_session)
    model_cp = keras.callbacks.ModelCheckpoint(
        os.path.join(path_session, "{}_model.hdf5".format(session_id)),
        monitor="val_loss",
        save_best_only=True)
    

    # train
    model = _get_model(hyper_param["optimizer"],
                       hyper_param["batch_norm"],
                       pool_type=hyper_param["pool_type"],
                       dropout_rate=hyper_param["dropout_rate"])
    history = model.fit_generator(
        _sample_generator(trainset, path_data, hyper_param["batch_sz"]),
        steps_per_epoch=int(len(trainset) / hyper_param["batch_sz"]),
        epochs=hyper_param["epochs"],
        validation_data=_sample_generator(valset, path_data, 2),
        validation_steps=int(len(valset) / 2),
        callbacks=[model_cp, hyper_param["lr_schedule"]],
        verbose=1,
        workers=4)

    # plot training curves
    def plot_history(metric):
        plt.ioff()
        str_metric = "accuracy" if metric == "acc" else "loss"
        plt.plot(history.history[metric])
        plt.plot(history.history["val_{}".format(metric)])
        plt.title("model {}".format(str_metric))
        plt.ylabel(str_metric)
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig(os.path.join(path_session, 
                                 "{}_{}.png".format(session_id, str_metric)))
    
    plot_history("loss")
    plt.cla()
    plot_history("acc")   
    with open(os.path.join(path_session,
                           "{}_history.pkl".format(session_id)),
              'wb') as output:
        pickle.dump(history.history, output, pickle.HIGHEST_PROTOCOL)
    
    # output model and performance measures
    ind_min_loss = np.argmin(history.history["val_loss"])
    return (os.path.join(path_session, "{}.hdf5".format(session_id)),
            history.history["val_loss"][ind_min_loss],
            history.history["val_acc"][ind_min_loss])



def split_train_val(path_train_labels, ratio_train=0.7, seed=1):
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

def load_ensemble(path):
    """Loads an ensemble of classifiers and meta from a cross-validation run.

    Args:
        path: /path/to/session_dir

    Returns:
        models: list of list of sorted models, same as in `train_ensemble()`

    """
    models = []
    for crossval_id in os.listdir(path):
        if os.path.isdir(os.path.join(path, crossval_id)):
            models_i = []
            for task_id in os.listdir(os.path.join(path, crossval_id)):
                task_prefix = os.path.join(path, crossval_id, task_id, task_id)
                with open("{}_history.pkl".format(task_prefix), "rb") as f:
                    if sys.version[0:3] == "2.7":
                        history = pickle.load(f)
                    else:
                        history = pickle.load(f, encoding="latin1")
                    ind_min_loss = np.argmin(history["val_loss"])
                    models_i.append(
                        ("{}_model.hdf5".format(task_prefix),
                        history["val_loss"][ind_min_loss],
                        history["val_acc"][ind_min_loss]))

            models_i.sort(key=lambda tuple: tuple[1])
            models.append(models_i)

    return models

def predict_ensemble(models, path_data, test_ids, path_output):
    """Predict an ensemble of models.

    Args:
        models: output from `train_ensemble` or `load_ensemble`
        path_data: /path/to/detections.hdf5
        test_ids: predict on these scan IDs
        path_output: save predictions.csv to this path

    Returns:
        saves a collection of prediction.csv files to path_output
    """

    if not os.path.exists(path_output):
        os.mkdir(path_output)
    session_id = os.path.basename(path_output)

    # predict all models
    predictions = []
    models_inds = []
    models_loss = []
    with h5py.File(path_data, "r") as fh5:
        for i, models_crossval in enumerate(models):

            print("*** predictions on {}/{}".format(i + 1, len(models)))
            for j, model_task in enumerate(models_crossval):
                
                # load the model
                model = keras.models.load_model(model_task[0])

                # predict
                preds = np.zeros(len(test_ids))
                for i in range(0, len(test_ids)):
                    preds[i] = model.predict_generator(
                        _sample_generator([(id,) for id in test_ids], # req. tuple
                                          path_data,
                                          batch_sz=1,
                                          mode="predict"),
                        steps = 1)
                predictions.append(preds)

                # stats about model
                models_inds.append((i, j))
                models_loss.append(model_task[1])

    # ----- several bagging regimes
    predictions = np.stack(predictions, axis=1)
    def csv_write_helper(filename, ids, vals):
        filename = "{}_{}".format(session_id, filename)
        with open(os.path.join(path_output, filename), "w") as f:
            wr = csv.writer(f)
            wr.writerow(("id", "cancer"))
            for id, val in zip(ids, vals):
                wr.writerow((id, "%.2f" % np.round(val, 2)))

    # mean over all predictions
    preds = np.mean(predictions, axis=1)
    csv_write_helper("ensemble_mean.csv", test_ids, preds)

    # median over all predictions
    preds = np.median(predictions, axis=1)
    csv_write_helper("ensemble_median.csv", test_ids, preds)

    # mean over top models from each cross-validation round
    inds_top_xval = [i for i, inds in enumerate(models_inds) if inds[1] == 0]
    preds = np.mean(predictions[:, inds_top_xval], axis=1)
    csv_write_helper("crossval_top_mean.csv", test_ids, preds)
    
    # median over top models from each cross-validation round
    preds = np.median(predictions[:, inds_top_xval], axis=1)
    csv_write_helper("crossval_top_median.csv", test_ids, preds)

    # mean over top percentile
    top_prc = np.percentile(models_loss, 20)
    inds_top_prc = [i for i, val in enumerate(models_loss) if val <= top_prc]
    preds = np.mean(predictions[:, inds_top_prc], axis=1)
    csv_write_helper("prctle_top_mean.csv", test_ids, preds)

    # median over top percentile
    preds = np.median(predictions[:, inds_top_prc], axis=1)
    csv_write_helper("prctle_top_median.csv", test_ids, preds)

    # top model
    ind_best = np.argmin(models_loss)
    csv_write_helper("best_model.csv", test_ids, predictions[:, ind_best])

def train_ensemble(trainset, valset, path_data, path_session, hyper_param):
    """Train an ensemble of models per set of hyper param.

    Args:
        trainset, valset: training and validation sets from `split_train_val()`
        path_data: /path/to/train_detections.hdf5
        path_session: string specifying the session's output path
        hyper_param: dictionary with entries as follows -
                        * epochs: number of epochs
                        * batch_sz: batch size in training
                        * batch_norm: do batch normalization?
                        * optimizer: a keras.optimizers beast
                        * lr_scheduler: a keras.callback.LearningRateScheduler

    """

    models = []
    for i, batch_sz in enumerate(hyper_param["batch_sz"]):
        for j, optimizer in enumerate(hyper_param["optimizers"]):
            for k, lr_param in enumerate(hyper_param["lr_scheduler_param"]):
                for l, dropout_rate in enumerate(hyper_param["dropout_rate"]):
                    for m, batch_norm in enumerate(hyper_param["batch_norm"]):
                        for n, pool_type in enumerate(hyper_param["pool_type"]):

                            # prepare the tasks' hyper param
                            hyper_param_ = {
                                "epochs": hyper_param["epochs"],
                                "batch_sz": batch_sz,
                                "optimizer": optimizer,
                                "lr_schedule": make_lr_scheduler(*lr_param),
                                "dropout_rate": dropout_rate,
                                "batch_norm": batch_norm,
                                "pool_type": pool_type
                                }

                            # task's path
                            session_id_ = "{}.{}_{}_{}_{}_{}_{}". \
                            format(os.path.basename(path_session),
                                   i, j, k, l, m, n)
                            path_session_ = os.path.join(path_session,
                                                         session_id_)
                            if not os.path.exists(path_session_):
                                os.mkdir(path_session_)

                            # train
                            models.append(train(
                                trainset,
                                valset,
                                path_data,
                                path_session_,
                                hyper_param_))
   
    # sort by validation loss
    return models.sort(key=lambda tuple: tuple[1])
