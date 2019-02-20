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
    
    
    def __init_(self, n_hidden=1, n_latent=2):
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.data = data
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
    
    
    def make_decoder(self, data):
        output_shape = tf.shape(data)
        net_layers = [tf.keras.layers.Flatten()]
        for i in range(self.n_hidden):
            tmp_layer = tf.keras.layers.Dense(2 * latent_size, activation=None)
            net_layers.append(tmp_layer)
        decode_net = tf.keras.Sequential(net_layers)
        def decoder(codes):
            orig_shape = tf.shape(input=codes)
            codes = tf.reshape(codes, (-1, 1, 1, self.n_latent))
            logits = decode_net(codes)
            logits = tf.reshape(
                norm_samples,
                shape= tf.concat([orig_shape[:-1], output_shape])
            )
            return tfd.Independent(
                tfd.Binomial(logits = logits, total_count=2),
                reinterpreted_batch_ndims=len(output_shape),
                name='genotypes'
            )
        return decoder
       

    def encode(self):
        raise NotImplemented
    
    def decode(self):
        raise NotImplemented

    def model_fn(self,data,mode):
        encoder = self.make_encoder()
        decoder = self.make_decoder(data)
        approx_posterior = encoder(data)
        approx_posterior_sample = approx_posterior.sample(100)
        decoder_likelihood = decoder(approx_posterior_sample)
        distortion = -decoder_likelihood.log_prob(data)
        avg_distortion = tf.reduce_mean(input_tensor=distortion)
        rate = tfd.kl_divergence(approx_posterior, tfd.MultivariateNormalDiag(loc=tf.zeros([self.n_latent]), scale_identity_multiplier=1.0))
        avg_rate = tf.reduce_mean(input_tensor=rate)
        elbo_local = -(rate + distortion)
        elbo = tf.reduce_mean(input_tensor=elbo_local)
        loss = -elbo
        global_step = tf.compat.v1.train.get_or_create_global_step()
        learning_rate = tf.compat.v1.train.cosine_decay(
            0.001,
            global_step,
            1000
        )
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            train_op=train_op,
            eval_metric_ops ={
                "elbo": tf.compat.v1.metrics.mean(elbo),
                "rate": tf.compat.v1.metrics.mean(avg_rate),
                "distortion":  tf.compat.v1.metrix.mean(avg_distortion)
            }
        )
        
        
        