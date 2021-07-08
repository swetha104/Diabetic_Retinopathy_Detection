'''The model is trained and saved according to the number of epochs.
Also the model checkpoints are saved are running each epoch.
Please adapt the path to save the model accordingly'''

import tensorflow as tf
import os
from input_pipeline.preprocessing import train_generator, val_generator
from models.layers import mdl

#Path to save the model and checkpoints
path_to_save_model = input("Enter the path for saving the model and checkpoints: ")

checkpoint_path = path_to_save_model + '/' + 'cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

if not os.path.exists(path_to_save_model):
    pass
else:
    try:
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    except:
        print('Path Exception raised') 
        
def Trainer(epochs):
    """Function to train the compiled model based on the dataset inputs and number of epochs"""
    epochs = EPOCHS
    history = mdl.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[cp_callback]        
    )

    if not os.path.exists(path_to_save_model):
        pass
    else:
        try:
            mdl.save(path_to_save_model + '/' + 'DRD_Model.h5')
        except:
            print('Path Exception raised')

    return history
print(''' ***************************Start Training************************''')
EPOCHS = 150

history = Trainer(EPOCHS)
print(''' ***************************End Training************************''')
