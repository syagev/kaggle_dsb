import numpy as np
import kaggle.train
import kaggle.process_luna as kgluna

#train, val = split_train_val()
#train = random.sample(train, 30)
#val = random.sample(val, 10)
#gen_simulated_data(train + val)

print("imported!")

# load luna dataset, split to train and validation
d0, d1 = kgluna.get_candidates()
ids_train = [item[0] for item in
             np.sample(list(d1.keys()), int(np.round(0.7 * len(d1))))]
ids_val = set(d1.keys).difference(set(ids_train))

model = get_unet(PATH_UNET)
model_checkpoint = ModelCheckpoint("unet_trained.hdf5", monitor="loss",
                                   save_best_only=True)
history = model.fit_generator(kgluna.luna_generator(d0, d1, 8, ids_train),
                              steps_per_epoch=25,
                              epochs=3,
                              validation_data=
                              kgluna.luna_generator(d0, d1, 8, ids_val),
                              validation_steps=5,
                              callbacks=[model_checkpoint],
                              verbose=1)


# ------------------------------------------------
# DEFINE TRAINING SESSION
#print("-"*30)
#print("Creating and compiling model...")
#print("-"*30)
#model = get_mock_model()
#model_checkpoint = ModelCheckpoint("unet.hdf5", monitor="loss", save_best_only=True)

#print("-"*30)
#print("Fitting model...")
#print("-"*30)


#history = model.fit_generator(mnist_generator("training"),
#                              steps_per_epoch=250,
#                              epochs=5,
#                              validation_data=mnist_generator("testing"),
#                              validation_steps=20)


#history = model.fit(x, y, batch_size=2, validation_split=0.2, nb_epoch=1, verbose=1,
#          shuffle=True, callbacks=[model_checkpoint])


# TODO: print also the starting point
# summarize history for accuracy
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig("accuracy.png")
# summarize history for loss
plt.cla()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig("loss.png")
