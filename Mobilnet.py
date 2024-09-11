import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import tensorflow as tf
from keras.src.callbacks import history
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import cohen_kappa_score
from keras.applications import MobileNetV2

# Optionally, you can create a new model using the Functional API
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Ensuring the model is not trainable if that's your intent
base_model.trainable = False
# Create the output layer for the new model
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output_layer = tf.keras.layers.Dense(4, activation='softmax')(x)
# Create the new model
model = tf.keras.Model(inputs=base_model.input, outputs=output_layer)
# Create the new output layers, or continue using the existing structure
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output_layer = tf.keras.layers.Dense(4, activation='softmax')(x)

# Rebuild the model
model_vgg = tf.keras.Model(inputs=base_model.input, outputs=output_layer)

# Compile the model as before
model_vgg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Check the summary to verify the model structure
model.summary()


def build_model(base_model):
    # Assuming base_model outputs a tensor named `x`
    x = base_model.output
    # Flatten the output layer to 1 dimension
    x = tf.keras.layers.Flatten()(x)
    # Add a fully connected layer with 4 outputs and softmax activation
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    # Create the new model
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model_vgg = build_model(base_model)


# Example usage:
# Build the model function
def build_model(base_model):
    # Assuming you are adding layers or modifying the base_model
    base_model.add(tf.keras.layers.Flatten())
    base_model.add(tf.keras.layers.Dense(4, activation='softmax'))
    base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return base_model


# Example usage of VGG19
base_dir = "D:\\work thisis\\10GB\\OCT2017\\data\\"
print('Base directory --> ', os.listdir(base_dir))

train_dir = os.path.join(base_dir, "D:\\work thisis\\10GB\\OCT2017\\data\\train\\")
print("Train Directory --> ", os.listdir(train_dir))

validation_dir = os.path.join(base_dir, "D:\\work thisis\\10GB\\OCT2017\\data\\val\\")
print("Validation Directory --> ", os.listdir(validation_dir))

test_dir = os.path.join(base_dir, "D:\\work thisis\\10GB\\OCT2017\\data\\test\\")
print("Test Directory --> ", os.listdir(test_dir))


class CohenKappa(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='cohen_kappa', **kwargs):
        super(CohenKappa, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.cohen_kappa = self.add_weight(name='ck', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=1)
        y_pred = tf.math.argmax(y_pred, axis=1)
        current_kappa = tf.py_function(cohen_kappa_score, [y_true, y_pred], Tout=tf.float32)
        self.cohen_kappa.assign(current_kappa)

    def result(self):
        return self.cohen_kappa

    def reset_states(self):
        self.cohen_kappa.assign(0.0)


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, \
    Reshape, Dropout, Dense
from keras.models import Model


def build_mobilenetv2(input_shape, num_classes):
    # Define the input tensor
    input_tensor = Input(shape=input_shape)

    # Initial Conv2D layer
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Depthwise separable convolutions
    x = mobilenet_v2_block(x, 16, 3, 1, 1)
    x = mobilenet_v2_block(x, 24, 3, 2, 6)
    x = mobilenet_v2_block(x, 24, 3, 1, 6)
    x = mobilenet_v2_block(x, 32, 3, 2, 6)
    x = mobilenet_v2_block(x, 32, 3, 1, 6)
    x = mobilenet_v2_block(x, 32, 3, 1, 6)
    x = mobilenet_v2_block(x, 64, 3, 2, 6)
    x = mobilenet_v2_block(x, 64, 3, 1, 6)
    x = mobilenet_v2_block(x, 64, 3, 1, 6)
    x = mobilenet_v2_block(x, 64, 3, 1, 6)

    x = mobilenet_v2_block(x, 96, 3, 1, 6)
    x = mobilenet_v2_block(x, 96, 3, 1, 6)
    x = mobilenet_v2_block(x, 96, 3, 1, 6)

    x = mobilenet_v2_block(x, 160, 3, 2, 6)
    x = mobilenet_v2_block(x, 160, 3, 1, 6)
    x = mobilenet_v2_block(x, 160, 3, 1, 6)

    x = mobilenet_v2_block(x, 320, 3, 1, 6)

    # Last convolutional layer
    x = Conv2D(1280, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Global average pooling and output layer
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model


def mobilenet_v2_block(input_tensor, filters, kernel_size, strides, expansion_factor):
    # Expansion phase
    expand_channels = input_tensor.shape[-1] * expansion_factor
    x = Conv2D(expand_channels, (1, 1), strides=(1, 1), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Depthwise convolution
    x = DepthwiseConv2D(kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Linear projection
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    # Skip connection if input and output shapes are the same
    if input_tensor.shape[-1] == filters:
        x = tf.keras.layers.add([x, input_tensor])

    return x


# Example usage
input_shape = (224, 224, 3)  # Image shape
num_classes = 1000  # Number of classes for classification
model = build_mobilenetv2(input_shape, num_classes)
model.summary()

# Usage
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[CohenKappa(num_classes=4)])

# Data generators
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), class_mode='categorical',
                                                    batch_size=500)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(224, 224),
                                                              class_mode='categorical', batch_size=16)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), class_mode='categorical',
                                                  batch_size=50)

# Fit model
history_vgg = model_vgg.fit(
    train_generator,
    steps_per_epoch=int(83484 / 500),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=int(32 / 16),
    verbose=1
)

# Plots and evaluation using corrected history object
acc = history_vgg.history['accuracy']
val_acc = history_vgg.history['val_accuracy']
loss = history_vgg.history['loss']
val_loss = history_vgg.history['val_loss']

epochs = range(len(acc))
plt.figure(figsize=(12, 12))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training & validation accuracy')
plt.legend()

plt.figure(figsize=(12, 12))
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training & validation loss')
plt.legend()
plt.show()

model_vgg.evaluate(test_generator)
predictions = model_vgg.predict(test_generator, steps=np.math.ceil(test_generator.samples / test_generator.batch_size))
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = sklearn.metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)
cm = sklearn.metrics.confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, fmt='.0f', annot=True, linewidths=0.2, linecolor='purple')
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.show()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title('Loss')
plt.legend()
import matplotlib.pyplot as plt

# Assuming 'history' is the History object returned by model.fit()
plt.plot(history.History['loss'], label='Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

plt.subplot(1, 2, 2)
plt.plot(history.History['accuracy'], label='Accuracy')
plt.plot(history.History['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(history.History['loss'], label='Loss')
ax[0].plot(history.History['val_loss'], label='Val Loss')
ax[0].set_title('Loss')
ax[0].legend()
ax[1].plot(history.History['accuracy'], label='Accuracy')
ax[1].plot(history.History['val_accuracy'], label='Val Accuracy')
ax[1].set_title('Accuracy')
ax[1].legend()
plt.show()


def display_examples(class_names, images, labels):
    """
        Display 25 images from the images array with its corresponding labels
    """

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.Binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()


import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.applications.resnet50 import preprocess_input

# Assuming you have a trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

X_train = []
y_train = []
X_test = []
y_test = []
y_true = []

X_train_preprocessed = preprocess_input(X_train)
X_test_preprocessed = preprocess_input(X_test)

# Generate predictions for the test set
y_pred = model.predict(X_test_preprocessed)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert one-hot encoded labels to integers
y_test_classes = np.argmax(y_test, axis=1)

# Create the confusion matrix

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Print confusion matrix
print(conf_matrix)

#*********************************
#**************** End Code Mustafa
