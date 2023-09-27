# VisionTransformer
Implementing the Vision Transformer for image classification tasks.


This project is for education and research purpose only.


# What We Have
* Supports training an image classifier from scratch.
* Supports training an image classifier using PyTorch pretrained weights.


# Environment and Requirements
* Python        3.10.6
* PyTorch       2.0.1
* Tensorboard   2.13.0


# Code Structure
* `config.py` contains all the training configurations, such as model type, number of epochs, batch size, learning rate, etc.
* `model.py` contains code for the Vision Transformer model class.
* `train.py` contains code for the training and evaluation, as well as the main loop.
* `schedule.py` contains code for a custom learning rate scheduler class.
* `download_pretrained_weights.py` contains code for downloading PyTorch pretrained weights.


# Project Setup

```
python3 -m pip install --upgrade pip setuptools

python3 -m pip install -r requirements.txt
```


# Run training

## Training on CIFAR10 from Scratch
Step 1: Run the training script
```
python3 train.py
```

## Training on CIFAR10 Using PyTorch Pretrained Weights

Step 1: Download the weights and maintain the file path in the `config.py` module
```
python3 download_pretrained_weights.py
```

Step 2: Run the training script, we found using fixed learning rate of 1e-5 achieves good results after 5 epochs of training.
```
python3 train.py
```


## Monitoring progress
We can monitor the progress by using Tensorboard:
```
tensorboard --logdir=./logs
```

![Tensorboard](/images/tensorboard_log.png)



# License
This project is licensed under the MIT License (the "License"), see the LICENSE file for details
