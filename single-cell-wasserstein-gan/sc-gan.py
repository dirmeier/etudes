#!/usr/bin/env python

import sys
import logging
import click
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import keras.backend as K


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
frmtr = logging.Formatter(
  '[%(levelname)-1s/%(processName)-1s/%(name)-1s]: %(message)s')

class SCGAN():
    """
    This architecture is largely taken from https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py
    """    
    def __init__(self, nrow, ncol, nchan):
        self.img_rows = nrow
        self.img_cols = ncol
        self.channels = nchan
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.critic.trainable = False

        valid = self.critic(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * self.img_rows * self.img_cols, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.img_rows, self.img_cols, 128)))
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary(print_fn=logger.info)

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary(print_fn=logger.info)

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X, fl, epochs, batch_size=128, sample_interval=500):

        X = (X.astype(np.float32) - 127.5) / 127.5

        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):
            for _ in range(self.n_critic):
                idx = np.random.randint(0, X.shape[0], batch_size)
                imgs = X[idx]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            if epoch % sample_interval == 0:
                self.sample_images(fl, epoch)

        for epoch in range(epochs + 1, epochs + 101):
            self.sample_images(fl, epoch)


    def sample_images(self, fl, epoch):
        r = 5
        noise = np.random.normal(0, 1, (r, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, 4)
        for i in range(r):
            axs[i, 0].imshow(gen_imgs[i, :, :, :])
            axs[i, 1].imshow(gen_imgs[i, :, :, 0] / 255)
            axs[i, 2].imshow(gen_imgs[i, :, :, 1] / 255)
            axs[i, 3].imshow(gen_imgs[i, :, :, 2] / 255)
            for j in range(4):
                axs[i,j].axis('off')
        fig.savefig("{}-{}.png".format(fl, epoch))
        np.save("{}-{}.npy".format(fl, epoch), gen_imgs)
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
    if channels == 1:
        X = X[:,:,:,0]
        X = np.expand_dims(X, axis=3)
    nrow, ncol, nchan = X.shape[1:]

    logger.info("Dimensionality of data: {} x {} x {}".format(nrow, ncol, nchan))

    from datetime import datetime as dt
    start = dt.now()
    
    gan = SCGAN(nrow, ncol, nchan)
    gan.train(X, outfile, epochs=epochs, batch_size=64, sample_interval=200)
    
    logger.info("Time needed: {} seconds".format((dt.now() - start).seconds))

if __name__ == '__main__':
    run()
