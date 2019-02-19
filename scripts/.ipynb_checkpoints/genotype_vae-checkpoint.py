"""
An implementation of a Variational Autoencoder (VAE) that trains on a genotype matrix. The
goal of the autoencoder is to learn a latent representation of genotypes from which it can
reconstruct genotypes with similar nonlinear relationships with each other between samples.
"""
import functools
import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions


class vae(object):
    
    
    def __init_(self, data, n_hidden=1, n_latent=2):
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        raise NotImplemented
        
    
    def build_encoder(self):
        net_layers = [tf.keras.layers.Flatten()]
        for i in range(self.n_hidden):
            tmp_layer = tf.keras.layers.Dense(2 * latent_size, activation=None)
            net_layers.append(tmp_layer)
        encode_net = tf.keras.Sequential(net_layers)
        
        def encoder(X):
            X = 2 * tf.cast(X, tf.float32) - 1
            NN = encode_net(X)
            return tfd.MultivariateNormalDiag(
                loc=NN[..., :self.n_latent],
                scale_diag=tf.nn.softplus(net[..., self.n_latent:] + _softplus_inverse(1.0)),
                name="encode"
            )        
        return encoder
    
    
    def make_decoder(self, output_shape):
        net_layers = [tf.keras.layers.Flatten()]
        for i in range(self.n_hidden):
            tmp_layer = tf.keras.layers.Dense(2 * latent_size, activation=None)
            net_layers.append(tmp_layer)
        decode_net = tf.keras.Sequential(net_layers)
        def decoder(codes):
            orig_shape = tf.shape(input=codes)
            codes = tf.reshape(codes, (-1, 1, 1, self.n_latent))
            norm_samples = decode_net(codes)
            norm_samples tf.reshape(
                norm_samples,
                shape= tf.concat([orig_shape[:-1], output_shape])
            )
            raise NotImplemented
        raise NotImplemented
       
    
    def encode(self):
        raise NotImplemented
    
    def decode(self):
        raise NotImplemented

    def train(self):
        raise NotImplemented