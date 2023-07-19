from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv1D, Conv2D, \
    Flatten, GRU
from tensorflow.keras.layers import Reshape, Conv2DTranspose, UpSampling1D
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate, Flatten
from tensorflow.keras.models import Model
import numpy as np
import math
import matplotlib.pyplot as plt

"""Model Design
The archeticture closely follows Huang, X., Li, Y., Poursaeed, O., Hopcroft, J.,
& Belongie, S. (2017). Stacked generative adversarial networks.

The stacked GAN implementation by Atienza, Rowel has been invaluable:
Advanced Deep Learning with Keras: Apply deep learning techniques, autoencoders, GANs, variational autoencoders, deep
reinforcement learning, policy gradients, and more. Packt Publishing Ltd, 2018.
"""


def generator(inputs, SHAPE,
              activation='sigmoid',
              labels=None,
              codes=None):
    """Build a Generator Model
    Output activation is sigmoid instead of tanh in as Sigmoid converges easily.
    [1] Radford, Alec, Luke Metz, and Soumith Chintala.
    "Unsupervised representation learning with deep convolutional
    generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015)..
    Arguments:
        inputs (Layer): Input layer of the generator (the z-vector)
        activation (string): Name of output activation layer
        codes (list): 2-dim disentangled codes
    Returns:
        Model: Generator Model
    """

    if codes is not None:
        # generator 0 of MTSS
        inputs = [inputs, codes]
        x = concatenate(inputs, axis=1)
        # noise inputs + conditional codes
    else:
        # default input is just a noise dimension (z-code)
        x = inputs ##

    x = Dense(SHAPE[0]*SHAPE[1])(x)
    x = Reshape((SHAPE[0], SHAPE[1]))(x)
    x = GRU(72, return_sequences=False, return_state=False,unroll=True)(x)
    x = Reshape((int(SHAPE[0]/2), 6))(x)
    x = Conv1D(128, 4, 1, "same")(x)
    x = BatchNormalization(momentum=0.8)(x) # adjusting and scaling the activations
    x = ReLU()(x)
    x = UpSampling1D()(x)
    x = Conv1D(6, 4, 1, "same")(x)
    x = BatchNormalization(momentum=0.8)(x)

    if activation is not None:
        x = Activation(activation)(x)

    # generator output is the synthesized data x
    return Model(inputs, x,  name='gen1')



def discriminator(inputs, SHAPE,
                  activation='sigmoid',
                  num_labels=None,
                  num_codes=None):
    """Build a Discriminator Model
    The network does not converge with batch normalisation so it is not used here
    Arguments:
        inputs (Layer): Input layer of the discriminator (the sample)
        activation (string): Name of output activation layer
        num_codes (int): num_codes-dim Q network as output
                    if MTSS-GAN or 2 Q networks if InfoGAN

    Returns:
        Model: Discriminator Model
    """
    ints = int(SHAPE[0]/2)
    x = inputs
    x = GRU(SHAPE[1]*SHAPE[0] , return_sequences=False, return_state=False,unroll=True, activation="relu")(x)
    x = Reshape((ints, ints))(x)
    x = Conv1D(16, 3,2, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(32, 3, 2, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(64, 3, 2, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(128, 3, 1, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    # default output is probability that the time series array is real
    outputs = Dense(1)(x)

    if num_codes is not None:
        # MTSS-GAN Q0 output
        # z0_recon is reconstruction of z0 normal distribution
        # eventually two loss functions from this output.
        z0_recon =  Dense(num_codes)(x)
        z0_recon = Activation('tanh', name='z0')(z0_recon)
        outputs = [outputs, z0_recon]

    return Model(inputs, outputs, name='discriminator')


def build_encoder(inputs, SHAPE, num_labels=6, feature0_dim=6*24):
    """ Build the Classifier (Encoder) Model sub networks
    Two sub networks:
    1) Encoder0: time series array to feature0 (intermediate latent feature)
    2) Encoder1: feature0 to labels
    # Arguments
        inputs (Layers): x - time series array, feature1 -
            feature1 layer output
        num_labels (int): number of class labels
        feature0_dim (int): feature0 dimensionality
    # Returns
        enc0, enc1 (Models): Description below
    """


    x, feature0 = inputs

    y = GRU(SHAPE[0]*SHAPE[1], return_sequences=False, return_state=False,unroll=True)(x)
    y = Flatten()(y)
    feature0_output = Dense(feature0_dim, activation='relu')(y)
    # Encoder0 or enc0: data to feature0
    enc0 = Model(inputs=x, outputs=feature0_output, name="encoder0")

    # Encoder1 or enc1
    y = Dense(num_labels)(feature0)
    labels = Activation('softmax')(y)
    # Encoder1 or enc1: feature0 to class labels
    enc1 = Model(inputs=feature0, outputs=labels, name="encoder1")

    # return both enc0 and enc1
    return enc0, enc1


def build_generator(latent_codes, SHAPE, feature0_dim=144):
    """Build Generator Model sub networks
    Two sub networks: 1) Class and noise to feature0
        (intermediate feature)
        2) feature0 to time series array
    # Arguments
        latent_codes (Layers): dicrete code (labels),
            noise and feature0 features
        feature0_dim (int): feature0 dimensionality
    # Returns
        gen0, gen1 (Models): Description below
    """

    # Latent codes and network parameters
    labels, z0, z1, feature0 = latent_codes

    # gen0 inputs
    inputs = [labels, z0]      # 6 + 50 = 62-dim
    x = concatenate(inputs, axis=1)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    fake_feature0 = Dense(feature0_dim, activation='relu')(x)

    # gen0: classes and noise (labels + z0) to feature0
    gen0 = Model(inputs, fake_feature0, name='gen0')


    # gen1: feature0 + z0 to feature1 (time series array)
    # example: Model([feature0, z0], (steps, feats),  name='gen1')
    gen1 = generator(feature0, SHAPE= SHAPE, codes=z1)

    return gen0, gen1


def build_discriminator(inputs, SHAPE, z_dim=50):
    """Build Discriminator 1 Model
    Classifies feature0 (features) as real/fake time series array and recovers
    the input noise or latent code (by minimizing entropy loss)
    # Arguments
        inputs (Layer): feature0
        z_dim (int): noise dimensionality
    # Returns
        dis0 (Model): feature0 as real/fake recovered latent code
    """

    # input is 256-dim feature1
    x = Dense(SHAPE[0]*SHAPE[1], activation='relu')(inputs)

    x = Dense(SHAPE[0]*SHAPE[1], activation='relu')(x)


    # first output is probability that feature0 is real
    f0_source = Dense(1)(x)
    f0_source = Activation('sigmoid',
                           name='feature1_source')(f0_source)

    # z0 reonstruction (Q0 network)
    z0_recon = Dense(z_dim)(x)
    z0_recon = Activation('tanh', name='z0')(z0_recon)

    discriminator_outputs = [f0_source, z0_recon]
    dis0 = Model(inputs, discriminator_outputs, name='dis0')
    return dis0

