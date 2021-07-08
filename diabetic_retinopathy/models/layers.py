'''Model Architecture'''

import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from input_pipeline.preprocessing import N_img_width, N_img_height

print('------------------')
print('Model Architecture')
print('------------------')

def model(input_shape, kernel_size, pool_size, n_classes, dropout_rate_1, dropout_rate_2):
    """Defining the model architecture.
       Parameters:
           input_shape (tuple: 3): input shape of the neural network
           kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)
           pool_size   (tuple: 2): pool size used for the MaxPooling layers, e.g. (2, 2)
           n_classes   (int): number of classes, corresponding to the number of output neurons
           dropout_rate (float): dropout rate
       Returns:
           (keras.Model): keras model object
    """

    model = Sequential()
    model.add(Conv2D(8, kernel_size, strides=2, activation='relu',
                     input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = pool_size))

    model.add(Conv2D(16, kernel_size, strides=2, activation='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))

    model.add(Conv2D(32, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))

    model.add(Conv2D(128, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_rate_1))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout_rate_2))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

mdl = model((N_img_height, N_img_width, 3), (3, 3), (2, 2), 2, 0.4, 0.5)
print('------------------------------------------Start------------------------------------------')
print('Printing the Model Summary')
print(mdl.summary())
print('------------------------------------------End------------------------------------------')
