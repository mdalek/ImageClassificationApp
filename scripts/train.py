from matplotlib import pyplot as plt
from tensorflow.keras import models, layers, datasets
import os
import random
import tensorflow as tf


# Constants for class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def plot_images(train_images, train_labels):
    # Create a new figure
    plt.figure(figsize=(10,10))

    # Plot 25 images at random
    for i in range(25):
        n = random.randint(0, len(train_images))
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[n])
        plt.xlabel(CLASS_NAMES[train_labels[n][0]])

    # Save the figure
    plt.savefig(os.path.join("output","sample_images.png"))

def plot_training_history(history):
    """
    Plot training and validation accuracy over epochs.

    Args:
        history: The training history of the model.
    """

    plt.figure(figsize=(10, 10))
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join("output", "training_history.png"))

def create_model():
    
    """
    Create a convolutional neural network model for image classification.

    Returns:
        The CNN model.
    """

    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Flattening for dense layers
    model.add(layers.Flatten())

    # Dense layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    return model
        




if __name__ == "__main__":
    # Create outupt directory
    if not os.path.exists("output"):
        os.mkdir("output")


    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Plot sample images
    plot_images(train_images, train_labels)

     # Normalise images
    train_images, test_images = train_images / 255.0, test_images / 255.0

     # Model creation and summary
    model = create_model()
    model.summary()

     # Model compilation and training
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    # Plot training history
    plot_training_history(history)

    # Model evaluation
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc}")

    # Save the model
    model.save(os.path.join("output", "model.h5"))
