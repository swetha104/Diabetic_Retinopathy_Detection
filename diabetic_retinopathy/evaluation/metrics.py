'''Metrics : Confusion Matrix
Displays the true postives and true negatives'''

import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from input_pipeline.datasets import load
import numpy as np
from evaluation.eval import mdl
import matplotlib.pyplot as plt
from input_pipeline.preprocessing import N_img_width, N_img_height
from input_pipeline.datasets import dataset_path
import seaborn as sb
import os
import glob
import pandas as pd

"""Metrics : Confusion Matrix to predict the model test results"""

#The test images are loaded along with their ground truths
test_images = glob.glob(dataset_path + "/images/test/*.jpg")
print('Total number of test images:', len(test_images))
df_test = pd.read_csv(dataset_path + '/labels/test.csv')
df_test['Retinopathy grade'] = df_test['Retinopathy grade'].replace([0, 1, 2, 3, 4], [0, 0, 1, 1, 1])
df_test['Image name'] = df_test['Image name'] + '.jpg'
df_test = df_test.drop_duplicates()
df_test = df_test.iloc[:, : 2]
#Debug
#print(df_test)

test_images_list = []
test_labels = []
predicted_label_list = []

for iname, iclass in df_test.itertuples(index=False):
    for file in test_images:
      if os.path.basename(file) == iname:
        img = tf.io.read_file(file)
        img = tf.io.decode_jpeg(img)
        img = tf.cast(img,tf.float32) / 255
        img = tf.image.resize_with_pad(img, N_img_height, N_img_width, antialias=True)
        test_images_list.append(img)
        test_labels.append(iclass)
        img = tf.reshape(img, [1, N_img_height, N_img_width,3])
        x = mdl.predict(img)
        predicted_label = np.argmax(x)
        predicted_label_list.append(predicted_label)

test_images_list = tf.convert_to_tensor(test_images_list)
test_labels = tf.convert_to_tensor(test_labels)

#Evaluate the model and print the test accuracy
mdl.evaluate(test_images_list, test_labels)

#Append a predicted label column
df_test['Predicted Class'] = predicted_label_list
df_test['Result'] = np.where(df_test['Retinopathy grade'] == df_test['Predicted Class'], 'Correct Prediction', 'Incorrect Prediction')
#print(df_test)
df_test['Result'].value_counts() #The correct and incorrect labels are listed

#Confusion Matrix Plot
cm = confusion_matrix(df_test['Retinopathy grade'],df_test['Predicted Class']) #Plot the confusion matrix
plt.figure(figsize = (10,5))
plt.title('Confusion Matrix')
sb.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})
plt.show()
print('Test Accuracy:',metrics.accuracy_score(df_test['Retinopathy grade'], df_test['Predicted Class'])) #Obtain the test accuracy
