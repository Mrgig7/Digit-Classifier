# Handwritten Digit Classification using Convolutional Neural Networks (CNN)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)


This project utilizes a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the popular MNIST dataset. The network is implemented using Keras with TensorFlow as the backend.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

## Overview

This project is a simple implementation of a CNN model to classify handwritten digits. The MNIST dataset is used, which contains 60,000 training images and 10,000 testing images, each of size 28x28 pixels.

## Installation

To run this project, you'll need Python 3.x and the following libraries:

```bash
pip install numpy matplotlib keras tensorflow
```

## Dataset

The MNIST dataset is automatically downloaded when running the code. It contains grayscale images of handwritten digits (0-9), each image being 28x28 pixels.

## Model Architecture

The CNN model used in this project consists of the following layers:

1. **Conv2D Layer**: 64 filters, kernel size 3x3, ReLU activation, input shape (28,28,1).
2. **Conv2D Layer**: 32 filters, kernel size 3x3, ReLU activation.
3. **MaxPooling2D Layer**: Pool size 2x2.
4. **Flatten Layer**: Converts the matrix to a vector.
5. **Dense Layer**: 10 neurons, softmax activation.

## Training

The model is trained for 10 epochs with the following parameters:

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

```python
hist = model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=10)
```

## Evaluation

The model is evaluated using the test set. After 10 epochs, the model achieves an accuracy of approximately 98.15% on the test set.

```python
val_loss, val_acc = model.evaluate(X_test, y_test_one_hot)
```
## Results

The model demonstrates strong performance on the MNIST dataset, achieving high accuracy and low loss.

| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |
|-------|-------------------|---------------------|-----------------|
| 1     | 91.60%            | 97.36%              | 0.0790          |
| 2     | 97.97%            | 97.69%              | 0.0765          |
| 3     | 98.47%            | 97.79%              | 0.0736          |
| 10    | 99.41%            | 98.15%              | 0.1006          |

## References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
