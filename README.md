# Optimisation of Grid Search for ResNet CNN Hyperparameter Tuning in Single class Image Classification

# Dataset:
CIFAR-10
CIFAR-100
ImageNet
COCO
Intel Image Classification dataset
Scene Understanding dataset
Open Images V7

# Library:
TensorFlow & Keras

# Chosen Hyperparameters:
Batch Size
Learning Rate
Filter Size
TBC

# Context:
Hyperparameter tuning is an essential part of the model development process in artificial intelligence. Aim to find the most influential hyperparameter in the generalised use case of image recognition.

# Model Measurement:
F1 score
Total runtime
Initial accuracy
Total memory consumption
Total power consumption

# Method:
Construct an image-recognition model using pre-defined libraries
Apply grid search and Bayesian optimisation to tune hyperparameters on the same model
Measure the trained model to establish runtime baseline and accuracy baseline
Map out grid search iteration vs. accuracy - One changing hyperparameter
Observe these trends to allow for the implementation of optimised grid search
Implement an optimised grid search
Measure the trained model based on the optimised grid search
Using the three models, run the model on a different device and a different dataset
Measure generalisability
