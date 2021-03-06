# Project 1 - Diabetic Retinopathy Detection




# How to run the code
Run the **main.py** file.

Here you can find the different options for debugging the code. Please select the necessary option according to your choice. 
These options enables the user for displaying the images, logs, etc..
Also, please make sure to enter the correct dataset directory path.

The sequence of the codeflow in main.py is as follows:

**Dataset used :  Indian Diabetic Retinopathy Image Dataset (IDRiD)**

- An input pipeline is set-up initially  
- A model architecture is built
- Training of the model (Also, the saved model(DRD_model.h5) can be found in the experiments folder)  
- Evaluation of the model (Test accuracy is computed here)  
- Metrics - Confusion Matrix
- Deep Visualization (Available here - evaluation/deep_visualization_grad_cam.py)
- Other experimental results, logs and images are attached here

- The **tune.py** file can be executed separately to configure and analyze the hyper-parameter tuning.  

# Results

**--------------------------------------------------------------------**  
**The overall test accuracy obtained is 72.81%.**  
**--------------------------------------------------------------------**  


**1.  Input Pipeline**  

The following operations are performed on the input image,
- Resizing the image to 256x256(img_height x img_width) without any distortion.  
- Cropping the image borders  

![alt text](diabetic_retinopathy/experiments/images/Resized.png)

Binarization and balancing the dataset with **label 0(NRDR)** and **label 1(RDR)**,

| ![alt text](diabetic_retinopathy/experiments/images/hist1.png) | ![alt text](diabetic_retinopathy/experiments/images/hist2.png) | ![alt text](diabetic_retinopathy/experiments/images/hist3.png) |
|------------------------------------|------------------------------------|------------------------------------|

**2.  Data Augmentation**

Techniques used,  
- Rotation  
- Zoom  
- Shift  
- Horizontal and Vertical Flipping  

![alt text](diabetic_retinopathy/experiments/images/Augmented_Images.png)

**3. Hyperparameter Parameter Tuning using HParams**  

Hyperparameter tuning is performed to obtain a consistent model architecture,  

- HP_OPTIMIZER 
- HP_EPOCHS  
- HP_DENSE_LAYER  
- HP_DROPOUT  

| ![alt text](diabetic_retinopathy/experiments/images/Acc_hparams.png) | ![alt text](diabetic_retinopathy/experiments/images/acc_Hparams.png) |
|--------------------------------------|------------------------------------------|

**4. Model Architecture**  

The following architecture has been used, 

![alt text](diabetic_retinopathy/experiments/images/Model_Architecture.png)

**Model Summary**

![alt text](diabetic_retinopathy/experiments/images/Model_Summary.png)

**5. Evaluation and Metrics**

The model is evaluated and the training and validation accuracy and loss is as shown,  
x-axis : No of epochs | y-axis : Train/Validation Accuracy and Loss

![alt text](diabetic_retinopathy/experiments/images/Train_Val_728.png)

**Metrics : Confusion Matrix**

![alt text](diabetic_retinopathy/experiments/images/CM_728.jpg)

**6. Deep Visualization**

The following two techniques have been used to visualize the images,  
- Grad-CAM
- Grad-CAM + Guided Backpropagation  

![alt text](diabetic_retinopathy/experiments/images/grad_cam_3.png)  

![alt text](diabetic_retinopathy/experiments/images/grad_cam_2.png)  

![alt text](diabetic_retinopathy/experiments/images/grad_cam_4.png) 
