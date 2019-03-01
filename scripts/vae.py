"""
An implementation of a Variational Autoencoder (VAE) that trains on a genotype matrix. The
goal of the autoencoder is to learn a latent representation of genotypes from which it can
reconstruct genotypes with similar nonlinear relationships with each other between samples.
"""
import numpy as np
import tensorflow as tf


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis
    )


class VAE(tf.keras.Model):
    def __init__(self, output_size, n_latent=2, n_hidden=2): # default 2 latent dimensions
        super(VAE,self).__init__() # inherit Model functions
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.encode_net = tf.keras.Sequential()
        for i in range(n_hidden):
            self.encode_net.add(tf.keras.layers.Dense(128 * (i+1),activation=tf.nn.relu))
        self.encode_net.add(tf.keras.layers.Dense(2*n_latent)) #no Activation
        self.decode_net = tf.keras.Sequential()
        self.decode_net.add(tf.keras.layers.InputLayer(input_shape=(n_latent,)))
        for i in range(n_hidden)
            self.decode_net.add(tf.keras.layers.Dense(128 * (n_hidden - i + 1),activation=tf.nn.relu))        
        tf.keras.layers.Dense(output_size) #no Activation
        
    
    def call(self, data):
        mean,logvar = self.encode(data)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

    
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.n_latent))
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
    
    
    def total_loss(self, x,y):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent)
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    