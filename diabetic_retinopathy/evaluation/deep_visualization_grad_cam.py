import cv2
from google.colab.patches import cv2_imshow
from models.layers import mdl
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow import keras
import tensorflow as tf
from input_pipeline.preprocessing import N_img_width, N_img_height

#Debug
print('The input of the model: ', mdl.input)
print('The output of the model: ', mdl.output)

#A mapping that outputs target convolution and output
activation_grad_model = tf.keras.models.Model([mdl.input], [mdl.get_layer('conv2d_1').output, mdl.output])

#Using the second layer of the model to visualize the results
layers_name = ['conv2d_1']
#Debug
#layers_name = ['conv2d_2']
#layers_name = ['conv2d_3']

#Output of the model
outputs = [
    layer.output for layer in mdl.layers
    if layer.name in layers_name
]
print(outputs)

#Printing the layers in the model
for ilayer, layer in enumerate(mdl.layers):
    print("{:3.0f} {:10}".format(ilayer, layer.name))
	
#GradCam Visualization for few images
#Enter the image path that has to be visualized
img_path = input('Enter the path for the test image that has to be visualized: ')
img = tf.io.read_file(img_path)
img = tf.io.decode_jpeg(img)
img = tf.cast(img,tf.float32) / 255
img = tf.image.resize_with_pad(img, N_img_height, N_img_width, antialias=True)

img = tf.reshape(img, [1, N_img_height, N_img_width, 3])
x = mdl.predict(img) #Predict the label of the image from the lab
predicted_label = np.argmax(x)

print('Predicted Label:', predicted_label)

#Computing the loss between predicted label and true label for the above image
with tf.GradientTape() as tape:
      conv_outputs, predictions = activation_grad_model(img)
      loss = predictions[:, predicted_label]

#Computing the gradient loss obtained with respect to the output of the last convolution layer
output = conv_outputs[0]
grads = tape.gradient(loss,conv_outputs)[0]

#Only the positive gradient is chosen here
gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

#The average of the gradients are taken here
weights = tf.reduce_mean(guided_grads, axis=(0, 1))
weights_gradcam = tf.reduce_mean(grads, axis=(0, 1))

#Based on the prominence of the gradients, a heatmap is generated
cam = np.ones(output.shape[0:2], dtype=np.float32)
cam_gradcam = np.ones(output.shape[0:2], dtype=np.float32)

for index, w in enumerate(weights):
  cam += w * output[:,:,index]

for index, w in enumerate(weights_gradcam):
  cam_gradcam += w * output[:,:,index]

cam = cv2.resize(cam.numpy(), (256, 256))
cam_gradcam = cv2.resize(cam_gradcam.numpy(), (256, 256))

cam = np.maximum(cam, 0)
cam_gradcam = np.maximum(cam_gradcam, 0)

#Heatmap Visualization
heatmap = (cam - cam.min()) / (cam.max() - cam.min())
heatmap_gradcam = (cam_gradcam - cam_gradcam.min()) / (cam_gradcam.max() - cam_gradcam.min())

img = tf.reshape(img, (256,256,3))
img = np.asarray(img).astype('float32')
print(img.shape, img.dtype)

cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
  
cam_gradcam = cv2.applyColorMap(np.uint8(255*heatmap_gradcam), cv2.COLORMAP_JET)
output_image_gradcam = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam_gradcam, 1.0, 0)

b, g, r = cv2.split(output_image)
output_image = cv2.merge([r, g, b])

b, g, r = cv2.split(output_image_gradcam)
output_image_gradcam = cv2.merge([r, g, b])

#Plot the images
plt.figure(figsize=(20,20))
plt.subplot(1, 3, 1)
plt.title('Test Image with Diabetic Retinopathy')
plt.imshow(img)

plt.subplot(1, 3, 2)
plt.title('Grad-CAM ')
plt.imshow(output_image)

plt.subplot(1, 3, 3)
plt.title('Grad-CAM + Guided Backpropagation')
plt.imshow(output_image_gradcam)
