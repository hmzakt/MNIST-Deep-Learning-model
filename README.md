# MNIST Image Classification with PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.13%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8%2B-EE4C2C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Mlxtend](https://img.shields.io/badge/Mlxtend-0.23%2B-EE4C2C?style=for-the-badge&logo=python&logoColor=white)](http://rasbt.github.io/mlxtend/)
[![Tqdm](https://img.shields.io/badge/Tqdm-4.66%2B-EE4C2C?style=for-the-badge&logo=python&logoColor=white)](https://tqdm.github.io/)

This project demonstrates how to build, train, and evaluate a convolutional neural network (CNN) using PyTorch to classify images from the MNIST dataset.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Data Source](#data-source)
-   [Libraries Used](#libraries-used)
-   [Project Structure](#project-structure)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Results](#results)
-   [Model Saving and Loading](#model-saving-and-loading)

## Project Overview

The goal of this project is to train a CNN model to accurately classify handwritten digits from the MNIST dataset. The notebook walks through the entire process, from data loading and preprocessing to model training, evaluation, and making predictions.

## Data Source

The dataset used in this project is the **MNIST (Modified National Institute of Standards and Technology) database**. It is a large database of handwritten digits that is commonly used for training various image processing systems. The dataset consists of:

*   60,000 training images and labels.
*   10,000 testing images and labels.
*   Each image is a 28x28 pixel grayscale image.

The dataset is automatically downloaded using `torchvision.datasets.MNIST()`.

## Libraries Used

The following libraries are used in this project:

*   **PyTorch**: A deep learning framework for building and training neural networks.
*   **Torchvision**: A library for computer vision tasks, providing access to datasets, models, and transformations.
*   **Matplotlib**: A plotting library for visualizing data and results (e.g., sample images, confusion matrix).
*   **Mlxtend**: A library for data science and machine learning, used here for plotting the confusion matrix.
*   **Tqdm**: A library for displaying progress bars, used to visualize training progress.
*   **Timeit**: Python's built-in library for measuring execution time.
*   **Pathlib**: Python's built-in library for working with file paths.
*   **Requests**: A library for making HTTP requests, used to download helper functions.

## Project Structure

The project is implemented as a single Google Colab notebook (`03_pytorch_computer_vision_exercises.ipynb`). The notebook contains the following sections:

*   Importing libraries and setting up the device.
*   Loading and preparing the MNIST dataset.
*   Visualizing sample data.
*   Creating DataLoaders.
*   Defining the CNN model (`MNISTModel`).
*   Implementing training and testing steps (`train_step`, `test_step`).
*   Training the model.
*   Making and visualizing predictions.
*   Plotting the confusion matrix.
*   Experimenting with `nn.Conv2d`.
*   Saving and loading the trained model.

## Installation

If you are running this notebook in Google Colab, all necessary libraries are likely pre-installed.

If you are running this notebook locally, you can install the required libraries using pip

## Mentored by
Thanks to [Daniel Bourke](https://github.com/mrdbourke) and [learnpytorch.io](https://www.learnpytorch.io/) for resources and guidance
