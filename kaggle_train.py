import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kaggle.train as kgtrain
import kaggle.process_luna as kgluna
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import subprocess
import os
import h5py     # temp, for loading LUNA's processed h5 files for # samples

OUTPUT_DIR = "/razberry/workspace/luna2016"

#import ptvsd
#ptvsd.enable_attach(None, address = ('0.0.0.0', 3000))
#print("Waiting for attach...")
#ptvsd.wait_for_attach()

version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
version = version.strip().decode("ascii")
output_dir_ver = os.path.join(OUTPUT_DIR, version)
os.mkdir(output_dir_ver)

# load luna dataset, split to train and validation
d0, d1 = kgluna.get_candidates()
ids_train = [item for item in
             random.sample(list(d1.keys()), int(np.round(0.7 * len(d1))))]
ids_test = set(d1.keys()).difference(set(ids_train))

# TMP, for figuring out the training/test sets size
batch_size = 8
def load_samples(d, label, ids):
    with h5py.File(os.path.join(kgluna.PATH_OUTPUT, \
                                "{}.h5".format(label))) as f:
        return [(entry, label, i)
                for entry in d if entry in ids
                for i in range(0, f[entry].shape[-1])]
samples = load_samples(d0, 0, ids_train) + load_samples(d1, 1, ids_train)
steps_per_epoch = int(len(samples) / batch_size)
samples = load_samples(d0, 0, ids_test) + load_samples(d1, 1, ids_test)
validation_steps = int(len(samples) / batch_size)

model = kgtrain.get_unet()
model_cp = ModelCheckpoint(os.path.join(output_dir_ver, 
                                        "unet_{}.hdf5".format(version)),
                           monitor="val_loss",
                           save_best_only=True)
def lr_schedule(epoch):
    base = 5e-5
    rate_factor = 2
    rate_epochs = 5
    lr = base if epoch < rate_epochs \
              else base / (rate_factor * np.floor(epoch / rate_epochs))
    return lr
lr_scheduler = LearningRateScheduler(lr_schedule)
    
history = model.fit_generator(kgtrain.luna_generator(d0, d1, 8, ids_train),
                              steps_per_epoch=steps_per_epoch,
                              epochs=50,
                              validation_data=
                              kgtrain.luna_generator(d0, d1, 8, ids_test),
                              validation_steps=validation_steps,
                              callbacks=[model_cp, lr_scheduler],
                              verbose=1)

# TODO: print also the starting point
# summarize history for accuracy
plt.ioff()
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig(os.path.join(output_dir_ver, "accuracy.png"))
# summarize history for loss
plt.cla()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig(os.path.join(output_dir_ver, "loss.png"))
