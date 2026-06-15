# Hong Kong Transport Classifier

[![Hugging Face Space:](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/savanachan/HongKong-transport-classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a Computer Vision model that classifies images of Hong Kong public transport into three categories: **Minibuses**, **Trams**, and **Taxis**. It uses a Convolutional Neural Network (CNN) built with TensorFlow and provides a web interface powered by Gradio.

**Live Demo:** [Click here to try the classifier!](https://huggingface.co/spaces/savanachan/HongKong-transport-classifier)

## Model & Approach

- **Architecture:** A custom CNN built from scratch with TensorFlow/Keras.
- **Training Data:** ~80 images. Data augmentation (rotation, flipping, zoom) was used to artificially expand the dataset and prevent overfitting.
- **Performance:** The model achieves an **88% accuracy** on the validation set.
