#!/usr/bin/env python

import sys
import logging
import click
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.layers import *
from keras.models import Model, Sequential
from keras import backend as K
from keras.datasets import mnist
from keras import optimizers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
frmtr = logging.Formatter(
  '[%(levelname)-1s/%(processName)-1s/%(name)-1s]: %(message)s')

class DCAutoencoder():
    def __init__(self, nrow, ncol, nchan):
        self.img_rows = nrow
        self.img_cols = ncol
        self.channels = nchan
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.n_latent_features = 100

        optimizer = keras.optimizers.Adadelta()

        self.encoder = self.build_encoder(nrow, ncol, nchan)
        self.encoder_output_shape = self.encoder.layers[-2].output_shape[1]
        self.decoder = self.build_decoder(
        	nrow, ncol, nchan, self.encoder_output_shape)

        orig_image = Input(shape=(nrow, ncol, nchan))
        latent = self.encoder(orig_image)
        restored_image = self.decoder(latent)

        self.autoencoder = Model(orig_image, restored_image)
        self.autoencoder.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"])

    def build_encoder(self, nrow, ncol, nchan):
        input_img = Input(shape=(nrow, ncol, nchan))
        x  = Conv2D(16, kernel_size=5, strides=(1, 5), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((1, 5), padding='same')(x)
        x = Conv2D(8, kernel_size=5, strides=(1, 5), activation='relu', padding='same')(x)
        x = MaxPooling2D((1, 5), padding='same')(x)
        x = Conv2D(4, kernel_size=5, strides=(1, 5), activation='relu', padding='same')(x)
        x = MaxPooling2D((1, 5), padding='same')(x)
        x = Flatten()(x)
        encoded = Dense(self.n_latent_features)(x)

        model = Model(input_img, encoded)
        model.summary(print_fn=logger.info)

        return model

    def build_decoder(self, nrow, ncol, nchan, input_shape):
        input_feat = Input(shape=(self.n_latent_features, ))
        x = Dense(input_shape)(input_feat)
        x = Reshape((nrow, 1, 4))(x)
        x = Conv2D(4, kernel_size=5, strides=(1, 1), activation='relu', padding='same')(x)
        x = UpSampling2D((1, 2))(x)
        x = Conv2D(4, kernel_size=5, strides=(1, 1), activation='relu', padding='same')(x)
        x = UpSampling2D((1, 4))(x)
        x = Conv2D(8, kernel_size=5, strides=(1, 1), activation='relu', padding='same')(x)
        x = UpSampling2D((1, 5))(x)
        x = Conv2D(16, kernel_size=5, strides=(1, 1), activation='relu', padding='same')(x)
        x = UpSampling2D((1, 5))(x)
        decoded= Conv2D(nchan, kernel_size=3, strides=(1, 1), activation='sigmoid', padding='same')(x)

        model = Model(input_feat, decoded)
        model.summary(print_fn=logger.info)

        return model 


    def train(self, X, fl, epochs, batch_size):
        lamb_log = keras.callbacks.LambdaCallback(
            on_epoch_end = lambda epoch, logs: \
                logger.info("Epoch {} / loss {} / validation loss {}".format(
                    epoch, logs["loss"], logs["val_loss"]))
        )

        self.autoencoder.fit(
           X[:9000], X[:9000],
           epochs=epochs,
           batch_size=batch_size,
           shuffle=True,
           callbacks = [lamb_log],
           validation_data=(X[9000:], X[9000:]))
        self.autoencoder.save(fl + "_weights.h5")

    def predict_images(self, X, fl):
        test_idxs = np.random.choice(np.arange(9000, X.shape[0]), 10, replace=False)
        restored_images = self.autoencoder.predict(X[np.ix_(test_idxs)])

        f, ax = plt.subplots(ncols=10, nrows=2, figsize=(20, 5))
        for i in range(10):
            ax[0][i].imshow(X[test_idxs[i], :, :, 0], cmap="magma")
            ax[1][i].imshow(restored_images[i, :, :, 0], cmap="magma")
            ax[0][i].axis('off')
            ax[1][i].axis('off')
        f.savefig("{}-predicted_images.png".format(fl, epoch))
        plt.close()


@click.command()
@click.argument("file", type=str)
@click.argument("channels", type=int)
@click.argument("epochs", type=int)
@click.argument("outfile", type=str)
def run(file, channels, epochs, outfile):
    """
    Estimate a Wasserstein-DCGAN from images.
    """

    hdlr = logging.FileHandler(outfile + ".log")
    hdlr.setFormatter(frmtr)
    logger.addHandler(hdlr)

    X = np.load(file)
    if int(channels) == 1:
    	logger.info("Learning with a single channel")
        X = X[:,:,:,0]
        X = np.expand_dims(X, axis=3)
    nrow, ncol, nchan = X.shape[1:]
    X = X.astype(np.float32) / np.max(X)

    logger.info("Dimensionality of data: {} x {} x {}".format(nrow, ncol, nchan))

    from datetime import datetime as dt
    start = dt.now()

    gan = DCAutoencoder(nrow, ncol, nchan)
    gan.train(X, outfile, epochs=int(epochs), batch_size=64)
    gan.predict_images(X, outfile)

    logger.info("Time needed: {} seconds".format((dt.now() - start).seconds))

if __name__ == '__main__':
    run()

