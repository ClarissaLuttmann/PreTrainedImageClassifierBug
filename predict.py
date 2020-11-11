#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                         
# PROJECT: Image Classifier 
# FILE: Prediction File 
# PROGRAM: UDACITY, Introduction to Machine Learning with TensorFlow
# PROGRAMMER: Clarissa Luttmann
# DATE CREATED: 11/6/2020                                  
# REVISED DATE: -
# PURPOSE: Python application that can train an image classifier on a dataset, then predict new images # using the trained model. 
#
##

# ------------------------------------------------------------------------------- #
# Import Libraries
# ------------------------------------------------------------------------------- #

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import time 
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# ------------------------------------------------------------------------------- #
# Define Functions
# ------------------------------------------------------------------------------- #

def arg_parser():
    """Defines parser with required arguments. 
   
        Args: 
            None
            
        Returns: 
            args : parser and keyboard arguments 
        
    """
    
    # defines the parser
    parser = argparse.ArgumentParser(description="predict.py")
    
    ### IMAGE PROCESSING
    # adds image file directory to parser
    parser.add_argument('--img_path',
                        dest = "img_path",
                        action = "store",
                        type = str,
                        default = "./test_images/cautleya_spicata.jpg")
    
    ### LOADING MODEL
    # adds checkpoint path to parser
    parser.add_argument('--save_dir',
                        dest = "save_dir",
                        action = "store",
                        type = str,
                        default = "my_image_classifier_model.h5")
    
    ### PREDICTION 
    # add number of highest probability classifications for display to parser 
    parser.add_argument('--top_k',
                        dest = "top_k",
                        action = "store",
                        type = int,
                        default = 5)
    
    # adds json category names to parser 
    parser.add_argument('--category_names',
                        dest = "category_names",
                        action = "store",
                        type = str,
                        default = "label_map.json")

    # parses arguments 
    args = parser.parse_args()
    
    return args

def load_model(save_dir):
    """Loads saved model from directory path. 
   
        Args: 
            save_dir (str) : model checkpoint path
            
        Returns: 
            model : loaded model 
        
    """
    
    model = tf.keras.models.load_model(save_dir, custom_objects = {'KerasLayer': hub.KerasLayer})

    return model

def load_class_names(category_names):
    """Loads class names from image directory path. 
   
        Args: 
            img_path (str) : image file directory
            
        Returns: 
            flower_classes (dic) : dictionary containing class names 
        
    """
    
    with open(category_names, 'r') as f:
        names = json.load(f)
    
    class_names = {}
    for key in names.keys():
        class_names[str(int(key))] = names[key]

    return class_names

def process_image(image):
    """processes image for prediction. 
   
        Args: 
            image : image to be processed
            
        Returns: 
            image : processed image 
        
    """
    
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255 # normalizes image 
    return image.numpy()

def predict(img_path, model, top_k): 
    """Takes an image, a model, and then returns the top  𝐾  most likely class labels along with the probabilities.
   
        Args: 
            image_path (str) : image file directory to parser as specified by args
            model (str) : model checkpoint path as specified by args 
            top_k (int) : umber of highest probability classifications for display to parser as specified by args
            
        Returns: 
            top_k_probs : array of top-k probabilities 
            top_k_classes : array of top-k image classifications 
        
    """
    
    # Load the image.
    loaded_image = Image.open(img_path)  
    
    # Transform image into np array format. 
    array_image = np.asarray(loaded_image) 
    
    # Process image with process_image function 
    processed_image = process_image(array_image) 
    
    # Expand image dimensions from (224, 224, 3) to (1, 224, 224, 3)
    expanded_image = np.expand_dims(processed_image, axis = 0)
    
    # Calculate all probabilities for image class. 
    probs = model.predict(expanded_image)
    
    # Store top-k probabilities and corresponding image classes. 
    top_k_probs, top_k_classes = tf.nn.top_k(probs, k = top_k)
    
    return top_k_probs.numpy(), top_k_classes.numpy() 

# =============================================================================
# Main Function
# =============================================================================

def main(): 
    """Executes relevant functions 
   
        Args: None 
            
        Returns: None 
        
    """
    # gets keyword args with arg_parser 
    args = arg_parser()
    
    # load model 
    model = load_model(args.save_dir)
    
    # get class names 
    class_names = load_class_names(args.category_names)
    
    # predict top-k classes and corresponding probabilities
    top_k_probs, top_k_classes = predict(args.img_path, model, args.top_k)
    
    # print top-k classes and corresponding probabilities
    print(top_k_probs, top_k_classes) 
   
# =============================================================================
# Run Program 
# =============================================================================

if __name__ == '__main__': main()

