'''''''''This file helps us in loading the required dataset.
The data is split into training, validation and test sets.
This is further sent for preprocessing.'''

#Importing
import logging
import pandas as pd
import zipfile
import glob, os
import sys
import matplotlib.pyplot as plt

#Class functions
class data_preparation:
    '''Class containing functions for loading the dataset'''

    def load(self, data_dir, DEBUG_INPUT_PIPELINE):
        '''Function to check the dataset path and load the ground truths.
        Classifies the labels into 0's and 1's
        Plots the label histograms in Debug mode.'''

        self.train_images = glob.glob(data_dir + "/images/train/*.jpg") #Lists all the train images
        print('Total number of training images:', len(self.train_images))

        self.df_train = pd.read_csv(data_dir + '/labels/train.csv')
        self.df_train = self.df_train.drop_duplicates()
        self.df_train = self.df_train.iloc[:, : 2]

        #Histograms of the labels
        if DEBUG_INPUT_PIPELINE == 1:
            print(self.df_train.head())
            self.df_train[['Retinopathy grade']].hist(figsize=(10, 5))
            plt.title('Groundtruth Labels for Diabetic Retinopathy before binarization')
            plt.xlabel('Label')
            plt.ylabel('Number of Images')
            plt.show()

        '''Classifying the labels [0,1] to 0(NRDR) and labels [2,3,4] to 1(RDR)
        NRDR is considered as having no or mild non-proliferative DR (labels 0 and 1). 
        RDR is considered as having moderate, severe, or proliferative DR (labels 2 and up).'''

        self.df_train['Retinopathy grade'] = self.df_train['Retinopathy grade'].replace([0, 1, 2, 3, 4], [0, 0, 1, 1, 1])
        if DEBUG_INPUT_PIPELINE == 1:
            print(self.df_train.head())
            self.df_train[['Retinopathy grade']].hist(figsize=(10, 5))
            self.df_train = self.df_train.sample(frac=1).reset_index(drop=True)
            plt.title('Groundtruth Labels for Diabetic Retinopathy after binarization')
            plt.xlabel('Label')
            plt.ylabel('Number of Images')
            plt.show()


    def balancing_dataset(self, DEBUG_INPUT_PIPELINE):
        ''' Function to balance the dataset by random over-sampling the minority class '''

        N_Training = round(len(self.df_train) * 0.8) #80% is used for training
        train = self.df_train[:N_Training]
        self.validation = self.df_train[N_Training:] #20% is used for validation
        print('---------------------------------------------------------')
        print('Splitting the train samples into train and validation set')
        print('---------------------------------------------------------')
        print('Number of training samples:', len(train))
        print('Number of validation samples:', len(self.validation))

        label_0 = train[train['Retinopathy grade'] == 0]
        label_1 = train[train['Retinopathy grade'] == 1]
        print('---------------------------------------------------------')
        print('Number of images with Label 0 before sampling:', len(label_0))
        print('Number of images with Label 1 before sampling:', len(label_1))
        print('---------------------------------------------------------')

        label_count_1, label_count_0 = train['Retinopathy grade'].value_counts()
        label_0 = label_0.sample(label_count_1, replace=True)
        self.df_train_sampled = pd.concat([label_0, label_1])
        self.df_train_sampled = self.df_train_sampled.sample(frac=1, random_state=0)
        print('Number of images with Label 0 after over-sampling:', len(label_0))
        print('Number of images with Label 1 after over-sampling:', len(label_1))
        print('---------------------------------------------------------')

        #Histogram of the sampled and balanced class
        if DEBUG_INPUT_PIPELINE == 1:
            print(self.df_train_sampled)
            self.df_train_sampled[['Retinopathy grade']].hist(figsize = (10, 5))
            plt.show()

    def to_preprocessing(self):
        '''Function to send the data for further preprocessing.'''
        
        train_images_list = []
        train_labels_list = []
        val_images_list = []
        val_labels_list = []
        
        #Appending .jpg extension to the images
        self.df_train_sampled.loc[:,'Image name'] = self.df_train_sampled.loc[:,'Image name'] + '.jpg'
        self.validation.loc[:,'Image name'] = self.validation.loc[:,'Image name'] + '.jpg'

        for tname, tclass in self.df_train_sampled.itertuples(index=False):
            for ft in self.train_images:
                if os.path.basename(ft) == tname:
                    # print(fp,iname,iclass)
                    train_images_list.append(ft)
                    train_labels_list.append(tclass)

        for vname, vclass in self.validation.itertuples(index=False):
            for fv in self.train_images:
                if os.path.basename(fv) == vname:
                    # print(fv,vname,vclass)
                    val_images_list.append(fv)
                    val_labels_list.append(vclass)
        #Debug
        #print(len(train_images_list), len(train_labels_list), len(val_images_list), len(val_labels_list))
        return train_images_list, train_labels_list, val_images_list, val_labels_list


""" Class Instantiation"""
'''Flag: debug_mode_input_pipeline helps in visualising the necessary images'''

load = data_preparation()
dataset_path = input("Enter the path for the dataset. Please unzip the contents of the dataset before loading the folder path: ")
Debug = '''The Debug option here helps in analysing the histograms of the dataframes(train and test).
It also displays the format of the data in the .csv files'''
print(Debug)
debug_mode_input_pipeline = int(input('Enter 1 for enabling debug option(DEBUG_INPUT_PIPELINE) else enter 0 : '))
load.load(dataset_path,debug_mode_input_pipeline)
load.balancing_dataset(debug_mode_input_pipeline)
