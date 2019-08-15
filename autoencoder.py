from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
en_dim = 32

input_image = Input(shape=(784,))
encoded = Dense(en_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_image)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_image, decoded)

encoder = Model(input_image, encoded)
encoded_input = Input(shape=(en_dim,))
decoded_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoded_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

# print(x_train.shape)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# print(x_train.shape)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train.shape)

autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True
                , validation_data=(x_test, x_test))

encoded_images=encoder.predict(x_test)
decoded_images=decoder.predict(encoded_images)

import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()