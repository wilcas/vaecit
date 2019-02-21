"""
An implementation of a Variational Autoencoder (VAE) that trains on a genotype matrix. The
goal of the autoencoder is to learn a latent representation of genotypes from which it can
reconstruct genotypes with similar nonlinear relationships with each other between samples.
"""
import numpy as np
import tensorflow as tf


class vae(tf.keras.Model):
    def __init__(self, n_hidden=1, n_latent=2):
        super(vae, self).__init__()
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        # encoder network
        self.encode_net = tf.keras.Sequential()
        self.encode_net.add(tf.keras.layers.Flatten())
        for i in range(n_hidden):
            if i == (n_hidden - 1):
                tmp_layer = tf.keras.layers.Dense(2 * n_latent)
            else:
                tmp_layer = tf.keras.layers.Dense(2 * n_latent, activation = tf.nn.relu)
            self.encode_net.add(tmp_layer)
            
        # decoder network    
        self.decode_net = tf.keras.Sequential()
        self.decode_net.add(tf.keras.layers.Flatten())
        for i in range(n_hidden):
            if i == (n_hidden - 1):
                tmp_layer = tf.keras.layers.Dense(2 * n_latent)
            else:
                tmp_layer = tf.keras.layers.Dense(2 * n_latent, activation = tf.nn.relu)
            self.decode_net.add(tmp_layer)
    
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encode_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid = False):
        logits = self.decode_net(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        else:
            return logits


loss_object = tf.keras.losses.binary_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis
    )

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)    
    x_logit = model.decode(z)
    cross_ent = loss_object(x, tf.reduce_sum(x_logit,1))
    logpx_z = -cross_ent
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
        optimizer.apply_gradients(zip(gradients, variables))