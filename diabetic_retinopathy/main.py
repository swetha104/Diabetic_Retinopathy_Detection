import logging
from absl import app, flags
import os
Project_Header = '''|--------------------------------------------------------------------------------------------|
|Diabetic Retinopathy Detection Project - Team05                                             |
|--------------------------------------------------------------------------------------------|
|Team Members :                                                                              |
|1.  Ram Sabarish Obla Amar Bapu     |st169693|  email ID:  st169693@stud.uni-stuttgart.de   |
|2.  Swetha Lakshmana Murthy         |st169481|  email ID:  st169481@stud.uni-stuttgart.de   |
|--------------------------------------------------------------------------------------------|
The dataset used here is IDRID dataset. This contains a total of 516 fundus images of which 413 belongs to training and the remaining 103 belongs to the test dataset.
The entire project is divided into sequence of events. Please select the necessary ones to view their corresponding results.
Please install the necessary packages.
Folder structure of the dataset
    ROOT_FOLDER(/home/user/.../IDRID_dataset)
       |-------- images
       |            |------ train
       |            |           |------ IDRiD_001.jpg
       |            |                   etc...
       |            |------ test
       |                        |------ IDRiD_001.jpg
       |                                etc...
       |
       | -------- labels                 
       |             |
       |             | ----- test.csv
       |             | ----- train.csv
       |
    ```
Creation of an input pipeline for the IDRID dataset
Changes to be made according to your requirements is mentioned with '**TO_BE_CHANGED**' macro.
Please adapt it accordingly.
Folder structure of the dataset
    ROOT_FOLDER(/home/user/.../IDRID_dataset)
       |-------- images
       |            |------ train
       |            |           |------ IDRiD_001.jpg
       |            |                   etc...
       |            |------ test
       |                        |------ IDRiD_001.jpg
       |                                etc...
       |
       | -------- labels                 
       |             |
       |             | ----- test.csv
       |             | ----- train.csv
       |
    ```
'''

def main(argv):
    print('Main Function')

if __name__ == "__main__":
    # Importing the programs according to the sequence as mentioned in the README.md
    print(Project_Header)
    import input_pipeline.preprocessing
    import models.layers
    import evaluation.eval
    import evaluation.metrics
    import evaluation.deep_visualization_grad_cam
    app.run(main)
