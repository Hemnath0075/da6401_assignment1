# Fashion MNIST Neural Network

## Overview

This project implements a Neural Network from scratch to classify images from the Fashion MNIST dataset. The dataset consists of 28x28 grayscale images of clothing items, categorized into 10 classes.

Github Repo link :- https://github.com/Hemnath0075/da6401_assignment1/tree/main

**Wandb Report Link** :- https://wandb.ai/ch24s016-iitm/fashion_mnist_sweep/reports/CH24S016-Assignment-1--VmlldzoxMTgzOTIyMw?accessToken=nqq8sg6ggo037u2ntqs5a3tvf9feq3otyjkfibd9cfq1b8m54z7cfbo07sddhecj

## Dataset Details

* Each image is a 28x28 pixel grayscale image.
* Training set: 54,000 images
* Validation set:6000
* Test set: 10,000 images
* The input layer of the neural network consists of 784 neurons (flattened 28x28 pixels).
* Labels are one-hot encoded for classification.

## Preprocessing Steps

1. **Flattening the Images:** The 2D image arrays are reshaped into 1D arrays of 784 elements.
   ```python
   train_images = train_images.reshape(train_images.shape[0], -1) / 255.0
   test_images = test_images.reshape(test_images.shape[0], -1) / 255.0
   ```
2. **One-Hot Encoding Labels:** Labels are converted into a one-hot encoded format using NumPy.
   ```python
   import numpy as np

   def one_hot_encode(labels, num_classes=10):
       return np.eye(num_classes)[labels]

   train_labels = one_hot_encode(train_labels)
   test_labels = one_hot_encode(test_labels)
   ```

## Neural Network Implementation

The neural network is implemented with the following features:

* **Configurable architecture:** Input, multiple hidden layers, and output layers.
* **Weight Initialization Methods:** Random and Xavier initialization.
* **Activation Functions:** ReLU, Sigmoid, Tanh, and Softmax.
* **Optimization Algorithms:** SGD, Momentum, Nesterov, RMSprop, Adam.
* **Forward and Backward Propagation:** Computes activations and gradients for backpropagation.

### Example Usage

```python
from neural_network import NeuralNetwork

nn = NeuralNetwork(
    input_neurons=784,
    hidden_layers=[128, 64],
    output_neurons=10,
    init_wb_method='xavier',
    optimizer='adam',
    activation='relu',
    learning_rate=0.001,
    weight_decay=0.0001,
    beta1=0.9,
    beta2=0.999
)
```

## Training and Evaluation

The network is trained using a specified optimizer and loss function (categorical cross-entropy). The performance is evaluated using accuracy on the test set.

## Dependencies

* Python 3.x
* NumPy

## Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fashion-mnist-nn.git
   cd fashion-mnist-nn
   ```
2. Install dependencies:
   ```bash
   pip install numpy
   ```
3. Run the training script:
   ```bash
   python train.py
   ```

## License

This project is licensed under the MIT License.
