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
    def __init__(self, output_size=100, n_latent=2, n_hidden=2): # default 2 latent dimensions
        super(VAE,self).__init__() # inherit Model functions
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.encode_net = tf.keras.Sequential()
        self.encode_net.add(tf.keras.layers.InputLayer(input_shape=(output_size,)))
        for i in range(n_hidden):
            self.encode_net.add(tf.keras.layers.Dense(128 * (i+1),activation=tf.nn.relu))
        self.encode_net.add(tf.keras.layers.Dense(2*n_latent)) #no Activation
        self.decode_net = tf.keras.Sequential()
        self.decode_net.add(tf.keras.layers.InputLayer(input_shape=(n_latent,)))
        for i in range(n_hidden):
            self.decode_net.add(tf.keras.layers.Dense(128 * (n_hidden - i + 1),activation=tf.nn.relu))        
        self.decode_net.add(tf.keras.layers.Dense(output_size)) #no Activation
        
    
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


class MMD_VAE(tf.keras.Model):
    
    def __init__(self, output_size=100, n_latent=2, n_hidden=2): # default 2 latent dimensions
        super(MMD_VAE,self).__init__() # inherit Model functions
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.encode_net = tf.keras.Sequential()
        self.encode_net.add(tf.keras.layers.InputLayer(input_shape=(output_size,)))
        for i in range(n_hidden):
            self.encode_net.add(tf.keras.layers.Dense(128 * (i+1),activation=tf.nn.relu))
        self.encode_net.add(tf.keras.layers.Dense(2*n_latent)) #no Activation
        self.decode_net = tf.keras.Sequential()
        self.decode_net.add(tf.keras.layers.InputLayer(input_shape=(n_latent,)))
        for i in range(n_hidden):
            self.decode_net.add(tf.keras.layers.Dense(128 * (n_hidden - i + 1),activation=tf.nn.relu))        
        self.decode_net.add(tf.keras.layers.Dense(output_size)) #no Activation
    
    def call(self, data):
        z = self.encode(data)
        return self.decode(z)

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.n_latent))
        return self.decode(eps)

    def encode(self, x):
        return self.encode_net(x)

    def decode(self, z):
        logits = self.decode_net(z)
        return logits
        
    def total_loss(self, x,y):
        train_z = self.encode_net(x)
        train_xr = self.decode_net(train_z)
        samples = tf.random_normal(tf.stack([200,self.n_latent]))
        loss_mmd = compute_mmd(samples,train_z)
        loss_nll = tf.reduce_mean(tf.square(train_xr - x))
        return loss_mmd + loss_nll
    
    
def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)