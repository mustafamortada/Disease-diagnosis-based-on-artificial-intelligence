# Import Package
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Model
from keras import layers
from keras.layers import Input, Dense, ReLU, BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
import numpy as np
from keras.metrics import F1Score, Accuracy, Precision, Recall
import matplotlib.pyplot as plt
import cv2
from keras.applications import (InceptionV3, InceptionResNetV2, Xception, MobileNet, MobileNetV2, ResNet50V2, MobileNetV3Large, EfficientNetB0, DenseNet201, NASNetMobile, EfficientNetB4, ResNet50, VGG16)
import random
import os

SD = 42
K.clear_session()
np.random.seed(SD)
random.seed(SD)
tf.random.set_seed(SD)
os.environ['PYTHONHASHSEED'] = str(SD)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


TEST_DIR = "D://project code thesis/pythonProject2/OCT/test/"
VAL_SPLIT = 0.032
test_datagen = ImageDataGenerator(validation_split=VAL_SPLIT)
TRAIN_DIR = "D://project code thesis/pythonProject2/OCT/train/"

categories = os.listdir(TRAIN_DIR)
category_counts = {category: len(os.listdir(os.path.join(TRAIN_DIR, category))) for category in categories}

training_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
)

train_generator = training_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    color_mode="grayscale",
    seed = SD
)

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    color_mode="grayscale",
    subset='validation',
    seed = SD
)

test_generator = test_datagen.flow_from_directory(
	TEST_DIR,
	target_size=(224, 224),
	class_mode='categorical',
    batch_size=32,
    color_mode = "grayscale",
    subset='training',
    seed = SD
)


def create_model(base_model, input_shape=(224, 224, 1), num_classes= 4):
    input = Input(shape=input_shape)
    base = base_model(input_shape=input_shape, include_top=False, weights=None)
    x = layers.Flatten()(base(input))

    x = Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Activation('relu')(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    output = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)
    return model

models = [VGG16, InceptionV3]
model_names = ["VGG16", "InceptionV3"]

# Callbacks

reduce_lr_cb = ReduceLROnPlateau(factor=0.1, patience=5)
tensorboard_cb = TensorBoard(log_dir="logs/")
callbacks = [ tensorboard_cb, reduce_lr_cb]



y_true = test_generator.classes


for i, model_n in enumerate(models):

    model = create_model(model_n)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=25,
        verbose=1,
        batch_size=32,

        callbacks=callbacks
             )

    final_test_results = model.evaluate(test_generator)
    final_test_accuracy = final_test_results[1]
    print(f"Accuracy of {model_names[i]}: {final_test_accuracy * 100:.2f}")

    model_name = model_n.__name__
    model.save(f'{model_name}_best_weights.h5')

    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)


