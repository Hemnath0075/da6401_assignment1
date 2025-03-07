import wandb
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist


# wandb.init(project="fashion-mnist-visualization")

# Load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Get one sample per class
sample_images = []
sample_labels = []

for i in range(10):  # 10 classes in Fashion MNIST
    index = np.where(train_labels == i)[0][0]  # Find the first occurrence of each class
    sample_images.append(train_images[index])
    sample_labels.append(class_names[i])

# Create a figure and log to wandb
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Fashion-MNIST Samples", fontsize=16)

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i], cmap="gray")
    ax.set_title(sample_labels[i])
    ax.axis("off")

plt.show()
