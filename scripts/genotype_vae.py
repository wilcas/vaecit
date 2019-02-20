"""
An implementation of a Variational Autoencoder (VAE) that trains on a genotype matrix. The
goal of the autoencoder is to learn a latent representation of genotypes from which it can
reconstruct genotypes with similar nonlinear relationships with each other between samples.
"""
import numpy as np
import tensorflow as tf

tfd = tfp.distributions


class vae(object):
    
    
    def __init__(self, n_hidden=1, n_latent=50):
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        
    
    def build_encoder(self):
        net_layers = [tf.keras.layers.Flatten()]
        for i in range(self.n_hidden):
            tmp_layer = tf.keras.layers.Dense(2 * self.n_latent, activation=None)
            net_layers.append(tmp_layer)
        encode_net = tf.keras.Sequential(net_layers)
        
        def encoder(X):
            X = 2 * tf.cast(X, tf.float32) - 1
            NN = encode_net(X)
            return tfd.MultivariateNormalDiag(
                loc=NN[..., :self.n_latent],
                scale_diag=tf.nn.softplus(NN[..., self.n_latent:] + _softplus_inverse(1.0)),
                name="encode"
            )        
        return encoder
    
    
    def build_decoder(self, output_shape):
        net_layers = [tf.keras.layers.Flatten()]
        for i in range(self.n_hidden):
            tmp_layer = tf.keras.layers.Dense(2 * self.n_latent, activation=None)
            net_layers.append(tmp_layer)
        decode_net = tf.keras.Sequential(net_layers)
        def decoder(codes):
            orig_shape = tf.shape(input=codes)
            codes = tf.reshape(codes, (-1, 1, 1, self.n_latent))
            logits = decode_net(codes)
            logits = tf.reshape(
                logits,
                shape= tf.concat([orig_shape[:-1], output_shape], axis=0)
            )
            return tfd.Independent(
                tfd.Binomial(logits=logits, total_count=2),
                reinterpreted_batch_ndims=len(output_shape),
                name='genotypes'
            )
        return decoder
       