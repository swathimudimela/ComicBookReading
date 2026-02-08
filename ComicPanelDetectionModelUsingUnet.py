# -*- coding: utf-8 -*-
"""
Comic Panel Detection Model using VGG16 U-Net

Tried to implement the paper : https://arxiv.org/pdf/1902.08137
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# --- Configuration ---
IMG_HEIGHT = 768
IMG_WIDTH = 512
CHANNELS = 3
BATCH_SIZE = 2
EPOCHS = 10
TRAIN_IMAGES_PATH = "images/trainImages"
TRAIN_PANELS_PATH = "images/trainPanels"
TEST_IMAGE_PATH = "images/testImages/TestComicPage3.jpg"
MODEL_FILENAME = "comic_panel_model_unet.h5"

def load_data(image_dir, mask_dir, size_h, size_w):
    """
    Loads images and masks from directories.
    Note: For large datasets, consider using DataGenerators.
    """
    images = []
    masks = []

    print("Loading Images...")
    # Using sorted to ensure alignment between images and masks if filenames match
    # Assuming standard glob order or matching filenames for safety
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
   
    # Assuming 1:1 match 
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))

    if len(image_paths) != len(mask_paths):
        print(f"Warning: Number of images ({len(image_paths)}) and masks ({len(mask_paths)}) do not match!")

    for img_path in image_paths:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (size_w, size_h)) # resize expects (width, height)
            #VGG expects RGB.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    print("Loading Masks...")
    for mask_path in mask_paths:
        try:
            mask = cv2.imread(mask_path, 0) # Read as grayscale
            mask = cv2.resize(mask, (size_w, size_h))
            masks.append(mask)
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")

    images = np.array(images)/255.0
    masks = np.array(masks)/255.0
    masks = np.expand_dims(masks, axis=3) # Add channel dimension for mask

    return images, masks

def build_unet_vgg16(input_shape):
    """
    Builds a U-Net model using VGG16 as the encoder.
    """
    # Encoder
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze encoder layers
    for layer in vgg_base.layers:
        layer.trainable = False

    # vgg_base.summary()

    # Decoder / Expansion Path
    
    b5_conv = vgg_base.get_layer('block5_conv3').output # /16
    b4_conv = vgg_base.get_layer('block4_conv3').output # /8
    b3_conv = vgg_base.get_layer('block3_conv3').output # /4
    b2_conv = vgg_base.get_layer('block2_conv2').output # /2
    b1_conv = vgg_base.get_layer('block1_conv2').output # Full
    
    input_to_decoder = vgg_base.output # /32 (Pooling layer)

    # u6: 32 -> 16
    u6 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(input_to_decoder)
    u6 = Concatenate()([u6, b5_conv])
    c6 = Conv2D(512, (3, 3), padding='same', activation='relu')(u6)
    BatchNormalization()

    # u7: 16 -> 8
    u7 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, b4_conv])
    c7 = Conv2D(512, (3, 3), padding='same', activation='relu')(u7)
    BatchNormalization()

    # u8: 8 -> 4
    u8 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = Concatenate()([u8, b3_conv])
    c8 = Conv2D(256, (3, 3), padding='same', activation='relu')(u8)
    BatchNormalization()

    # u9: 4 -> 2
    u9 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = Concatenate()([u9, b2_conv])
    c9 = Conv2D(128, (3, 3), padding='same', activation='relu')(u9)
    BatchNormalization()

    # u10: 2 -> 1 (Full Res)
    u10 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c9)
    u10 = Concatenate()([u10, b1_conv])
    c10 = Conv2D(64, (3, 3), padding='same', activation='relu')(u10)
    BatchNormalization()
    
    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer = regularizers.l2(0.001) )(c10)
    BatchNormalization()


    model = Model(inputs=vgg_base.input, outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train(model, X_train, Y_train):
    print("Starting training...")
    checkpointer = ModelCheckpoint(MODEL_FILENAME, verbose=1, save_best_only=True)
    
    callbacks = [
        EarlyStopping(patience=2, monitor='val_loss'),
        TensorBoard(log_dir='logs'),
        checkpointer
    ]

    history = model.fit(
        X_train, 
        Y_train, 
        validation_split=0.1, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        callbacks=callbacks
    )
    return history

def predict_panel(model_path, image_path):
    print(f"Predicting for {image_path}...")
    try:
        if not os.path.exists(model_path):
            print("Model file not found.")
            return

        model = tf.keras.models.load_model(model_path)
        
        img = cv2.imread(image_path)
        if img is None:
            print("Could not read image.")
            return
            
        
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_input = np.expand_dims(img_resized, axis=0) # Batch dim
        img_input = cv2.cvtColor(img_input[0], cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_input, axis=0)/255

        prediction = model.predict(img_input)
        pred_mask = prediction[0]
        
        # Display/Save
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        
        plt.subplot(1, 2, 2)
        plt.title("Predicted Mask")
        plt.imshow(pred_mask[:, :, 0], cmap='gray')
        plt.show()
        
              
        print("Prediction complete.")

    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    # Check if data exists
    if not os.path.exists(TRAIN_IMAGES_PATH) or not os.path.exists(TRAIN_PANELS_PATH):
        print(f"Data directories not found at {TRAIN_IMAGES_PATH} or {TRAIN_PANELS_PATH}")
        print("Please ensure 'images/trainImages' and 'images/trainPanels' exist.")
    else:
        # Load Data
        X_train, Y_train = load_data(TRAIN_IMAGES_PATH, TRAIN_PANELS_PATH, IMG_HEIGHT, IMG_WIDTH)
        print(f"Data Loaded. X: {X_train.shape}, Y: {Y_train.shape}")

        if len(X_train) > 0:
            # Build Model
            model = build_unet_vgg16((IMG_HEIGHT, IMG_WIDTH, CHANNELS))
            model.summary()

            # Train
            train(model, X_train, Y_train)
            
            # Predict
            predict_panel(MODEL_FILENAME, TEST_IMAGE_PATH)
        else:
            print("No training data found.")
