
---

# Breast Cancer Detection using CNN

This project demonstrates a Convolutional Neural Network (CNN) model trained to detect the presence of breast cancer from medical images. The model is built using TensorFlow and Keras, and it is capable of classifying images as either indicating the presence of breast cancer or not.

## Requirements

Ensure you have the following libraries installed:

- numpy
- tensorflow

You can install the required packages using pip:

```
pip install numpy tensorflow
```

## Usage

1. Run the script to train the CNN model( Replace `'path_to_test_image'` with the actual path to the image you want to test for breast cancer.):

```bash
python3 predict_cancer.py
```

2. After training the model, it will be saved as "breast_cancer_detection_model.h5".


**Note: You must train your model before using the prediction script.**
