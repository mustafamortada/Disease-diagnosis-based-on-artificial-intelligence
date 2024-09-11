import os
import glob
import numpy as np
from PIL import Image
from keras.metrics import F1Score, Precision, Recall
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.src.metrics import AUC
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Direct paths (Assuming you're running the script from a sensible directory)
path_base = 'D://work thisis/10GB/OCT2017/data/train/'
categories = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
paths = [os.path.join(path_base, cat) for cat in categories]

# Load and prepare images
data = []
labels = []
image_dims = (64, 64)  # Height x Width

for idx, path in enumerate(paths):
    for file in glob.glob(path + '/*.jpeg'):
        with Image.open(file) as im:
            im_resized = im.resize(image_dims)
            data.append(np.array(im_resized))
            labels.append(idx)

data = np.array(data)
data = data.reshape((-1, 64, 64, 1)) / 255.0
labels = to_categorical(labels, num_classes=4)

# Splitting data into training and test sets
np.random.seed(42)
indices = np.random.permutation(data.shape[0])
train_idx, test_idx = indices[:8000], indices[8000:]
tr_images, te_images = data[train_idx], data[test_idx]
tr_target, te_target = labels[train_idx], labels[test_idx]

# Define model
model = Sequential([
    Conv2D(16, (3, 3), input_shape=(64, 64, 1), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(tr_images, tr_target, epochs=20, batch_size=64, validation_data=(te_images, te_target))

# Evaluate with custom metrics
#f1_metric = F1Score(num_classes=4, average='macro')
f1_metric = F1Score(average='macro')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall(), AUC(), F1Score(average='macro')])

precision_metric = Precision()
recall_metric = Recall()

pr_target = model.predict(te_images)
f1_metric.update_state(te_target, pr_target)
precision_metric.update_state(te_target, pr_target)
recall_metric.update_state(te_target, pr_target)

print(f'F1 Score: {f1_metric.result().numpy()}')
print(f'Precision: {precision_metric.result().numpy()}')
print(f'Recall: {recall_metric.result().numpy()}')

# Plotting
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(history.history['loss'], label='Loss')
ax[0].plot(history.history['val_loss'], label='Val Loss')
ax[0].set_title('Loss')
ax[0].legend()

ax[1].plot(history.history['accuracy'], label='Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Val Accuracy')
ax[1].set_title('Accuracy')
ax[1].legend()

plt.show()


# In[93]:


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



#*********************************
#**************** End Code Mustafa