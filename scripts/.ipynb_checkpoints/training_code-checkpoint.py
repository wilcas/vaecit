import tensorflow as tf
import numpy as np 
import time
import os
import glob
import matplotlib.pyplot as plt
import PIL


from IPython import display
from genotype_vae import *
tf.enable_eager_execution()
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], -1).astype('float32')
train_images = train_images
test_images = test_images.reshape(test_images.shape[0], -1).astype('float32')
test_images = test_images
# Normalizing the images to the range of [0., 1.]
train_images /= 255.
test_images /= 255.

# Binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.
TRAIN_BUF = 1000
BATCH_SIZE = 100

TEST_BUF = 200
train_dataset = train_images[0:1000,]#tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF)
test_dataset = test_images[0:1000,]#tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF)
epochs = 100
latent_dim = 50

#test_dataset = [tf.cast(tf.convert_to_tensor(np.random.binomial(2,0.25,1000)),tf.float32) for i in range(500)]
#train_dataset = [tf.cast(tf.convert_to_tensor(np.random.binomial(2,0.25,1000)),tf.float32) for i in range(500)]
# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[100,2])
model = vae(n_hidden=2)

def generate_and_save_images(model, epoch, test_input):
    print(model.sample(test_input).shape)
    predictions = tf.reshape(model.sample(test_input),(28,14))

    plt.imshow(predictions, cmap='gray')
    plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.draw()
    


for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        gradients, loss = compute_gradients(model, train_x)
        apply_gradients(optimizer, gradients, model.trainable_variables)
    end_time = time.time()
    train_error = loss

    if epoch % 1 == 0:
        for test_x in test_dataset:
            elbo = -compute_loss(model, test_x)
    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'
          'Training error: {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time,
                                     train_error))
    display.clear_output(wait=False)
    generate_and_save_images(
        model, epoch, random_vector_for_generation)
