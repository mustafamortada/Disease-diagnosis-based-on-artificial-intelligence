import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report
import os
import cv2
import numpy as np
from tqdm import tqdm
from keras.src.legacy.preprocessing.image import ImageDataGenerator




for dirname, _, filenames in os.walk('D:\\work thisis\\10GB\\OCT2017\\data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# # Color

# In[2]:


colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C', '#4B6F44', '#4F7942', '#74C365', '#D0F0C0']

# ---

# In[3]:




# labels = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
labels = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
image_size = 224
target_per_class = 1500

X_train = []
y_train = []

# Define an ImageDataGenerator with data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop through each class
for i in labels:

    folderPath = os.path.join('D:\\work thisis\\10GB\\OCT2017\\data\\train', i)

    # Count the current number of images in the class
    current_count = 0

    # Loop through the images in the class
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))

        # Apply data augmentation and add augmented images
        for _ in range(5):  # Apply augmentation 5 times per image to reach the target
            augmented_img = datagen.random_transform(img)
            X_train.append(augmented_img)
            y_train.append(i)
            current_count += 1

            # Stop when the target number of images is reached for this class
            if current_count >= target_per_class:
                break

    # Fill the class with additional copies of existing images if needed
    while current_count < target_per_class:
        for j in range(len(X_train)):
            if y_train[j] == i:
                img = X_train[j]
                X_train.append(img)
                y_train.append(i)
                current_count += 1
                if current_count >= target_per_class:
                    break
        # ---
    folderPath = os.path.join('D:\\work thisis\\10GB\\OCT2017\\data', 'test', i)

    # Count the current number of images in the class
    current_count = 0
    # Loop through the images in the class
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))

        # Apply data augmentation and add augmented images
        for _ in range(5):  # Apply augmentation 5 times per image to reach the target
            augmented_img = datagen.random_transform(img)
            X_train.append(augmented_img)
            y_train.append(i)
            current_count += 1

            # Stop when the target number of images is reached for this class
            if current_count >= target_per_class:
                break

    # Fill the class with additional copies of existing images if needed
    while current_count < target_per_class:
        for j in range(len(X_train)):
            if y_train[j] == i:
                img = X_train[j]
                X_train.append(img)
                y_train.append(i)
                current_count += 1
                if current_count >= target_per_class:
                    break

X_train = np.array(X_train)
y_train = np.array(y_train)

# In[4]:


len(X_train)

# In[5]:


X_train.shape

# In[6]:


# Plot the pie chart
for label in labels:
    count = np.sum(y_train == label)
    plt.bar(label, count)

# In[12]:


# Initialize an empty dictionary to store counts for each label
label_counts = {label: np.sum(y_train == label) for label in labels}

# Plot the pie chart using the accumulated counts for all labels
plt.pie(label_counts.values(), labels=[f'{label}  Retinal Diseases ' for label in labels], autopct='%1.1f%%',
        startangle=90)
plt.title('Pie Chart: Diagnosing Retinal Diseases ')
plt.show()

# Print the counts for each label
print("Counts for each label:", label_counts)

# In[13]:


k = 0
fig, ax = plt.subplots(1, 4, figsize=(20, 20))
fig.text(s='Sample Image From Each Label', size=18, fontweight='bold',
         fontname='monospace', color=colors_dark[1], y=0.62, x=0.4, alpha=0.8)
for i in labels:
    j = 0
    while True:
        if y_train[j] == i:
            ax[k].imshow(X_train[j])
            ax[k].set_title(y_train[j])
            ax[k].axis('off')
            k += 1
            break
        j += 1

# In[14]:


X_train, y_train = shuffle(X_train, y_train, random_state=101)

# In[15]:


X_train.shape

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=101)

# In[47]:


y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

# In[48]:


########################################### Model VGG16 ######################################################
from keras.applications import VGG16
from keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
num_classes = 4
# Load the VGG16 model pre-trained on ImageNet data
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)  # Adjust num_classes to match your dataset

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# In[53]:


tensorboard = TensorBoard(log_dir='logs')
# checkpoint = ModelCheckpoint("effnet.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
checkpoint = ModelCheckpoint("effnet.keras", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001,
                              mode='auto', verbose=1)

# In[54]:


history = model.fit(X_train, y_train, validation_split=0.1, epochs=10, verbose=1, batch_size=32,callbacks=[tensorboard, checkpoint, reduce_lr])

# In[56]:


# import matplotlib.pyplot as plt

# Extracting metrics from the history object
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Generate list of epoch numbers
epochs = range(1, len(train_acc) + 1)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot accuracy
ax1.plot(epochs, train_acc, label='Training Accuracy', marker='o')
ax1.plot(epochs, val_acc, label='Validation Accuracy', marker='x')
ax1.set_title('Training and Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Plot loss
ax2.plot(epochs, train_loss, label='Training Loss', marker='o')
ax2.plot(epochs, val_loss, label='Validation Loss', marker='x')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

# Show the plots
plt.show()

# In[57]:


pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
y_test_new = np.argmax(y_test, axis=1)

# In[58]:


print(classification_report(y_test_new, pred))

# In[59]:


from sklearn.metrics import accuracy_score
import numpy as np

# In[60]:


# Get the model predictions
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)

# Assuming y_test is one-hot encoded, if it's not, you don't need the next line
y_test_new = np.argmax(y_test, axis=1)

# Calculate the accuracy
test_accuracy = accuracy_score(y_test_new, pred)

# Print the accuracy
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")


# In[61]:


def display_random_image(class_names, images, labels):
    """
        Display a random image from the images array and its correspond label from the labels array.
    """

    index = np.random.randint(images.shape[0])
    plt.figure()
    plt.imshow(images[index])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Image #{} : '.format(index) + class_names[labels[index]])
    plt.show()


# ---------------------------------
predictions = model.predict(X_test)  # Vector of probabilities
pred_labels = np.argmax(predictions, axis=1)  # We take the highest probability

display_random_image(labels, X_test, pred_labels)


# In[63]:


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
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()



import numpy as np
from keras.applications import VGG16
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a dataset X_test with images and y_test with corresponding labels
# Assuming you have a VGG16 model loaded and trained

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate confusion matrix


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap( annot=True, fmt='d', cmap='Blues', xticklabels=['class_0', 'class_1'], yticklabels=['class_0', 'class_1'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# *********************************
# **************** End Code Mustafa