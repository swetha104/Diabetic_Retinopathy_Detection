'''Start Hyper-parameter tuning
- HP_OPTIMIZER
- HP_EPOCHS
- HP_DENSE_LAYER
- HP_DROPOUT

Visualize the results on tensorboard'''

import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers
from tensorboard.plugins.hparams import api as hp
from input_pipeline.preprocessing import train_generator, val_generator, N_img_height, N_img_width
from evaluation.metrics import test_images_list, test_labels

#Define the hyperparameters
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([256, 512]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.3, 0.4, 0.5]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam','sgd']))
HP_EPOCHS = hp.HParam('epochs',hp.Discrete([100, 150, 200]))

METRIC_ACCURACY = 'accuracy'

path_hparams = input('Enter the path to save the tuning logs: ')

with tf.summary.create_file_writer(path_hparams + 'logs/hparam_tuning').as_default():
  hp.hparams_config(
      hparams = [HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_EPOCHS],
      metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def train_test_model(hparams):
  model = Sequential()
  model.add(Conv2D(8, kernel_size=(3, 3),strides =2 ,
                 activation='relu', 
                 input_shape=(N_img_height, N_img_width, 3)))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(16, kernel_size=(3, 3),strides =2,
                 activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(hparams[HP_DROPOUT]))
  model.add(Flatten())
  model.add(Dense(hparams[HP_NUM_UNITS], activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='softmax'))

  model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )

  #Obtain the train_generator and val_generator from the created input pipeline
  model.fit(train_generator, validation_data=val_generator, epochs=hparams[HP_EPOCHS])

  #Evaluate the model and obtain the test results
  _, accuracy = model.evaluate(test_images_list, test_labels)
  return accuracy

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0
for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in HP_DROPOUT.domain.values:
    for optimizer in HP_OPTIMIZER.domain.values:
      for epochs in HP_EPOCHS.domain.values:

        hparams = {
            HP_NUM_UNITS: num_units,
            HP_DROPOUT: dropout_rate,
            HP_OPTIMIZER: optimizer,
            HP_EPOCHS: epochs,
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run(path_hparams + '/hparam_tuning/' + run_name, hparams)
        print(session_num)
        session_num += 1
        


# Visualizing on tensorboard
# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir /content/drive/MyDrive/logs/hparam_tuning
