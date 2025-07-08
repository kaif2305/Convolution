import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as convolve

'''
# Load a simple gray scale image
load_image = np.random.rand(10,10)

# Defining Convolution Kernels (Filters)

edge_detection_kernel = np.array([[-1, -1, -1],
                                  [-1, 8, -1],
                                  [-1, -1, -1]])
   
blur_kernel = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]) / 9.0

#Applying Convolution Kernels to the Image
edge_detection_image = convolve.convolve(load_image, edge_detection_kernel)
blurred_image = convolve.convolve(load_image, blur_kernel)

# Visulaizing the Original and Filtered Images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(load_image, cmap='gray')
axes[0].set_title('Original Image') 
axes[1].imshow(edge_detection_image, cmap='gray')
axes[1].set_title('Edge Detected Image') 
axes[2].imshow(blurred_image, cmap='gray')
axes[2].set_title('Blurred Image')
plt.show()

'''


import tensorflow as tf

#Create a sample input Tensor (batch size, channels, height, width)
input_tensor = np.random.rand(1, 10, 10, 1) # Example shape (sample, height, width, channels)

#Defining a Convolutional Layer
conv_layer = tf.keras.layers.Conv2D(
    filters=1,  # Number of output filters
    kernel_size=(3, 3),  # Size of the convolution kernel
    strides=(1, 1),  # Stride of the convolution
    padding='same',  # Padding type
)

# Applying the convolutional layer to the input tensor
output_tensor = conv_layer(input_tensor)

# Displaying the output tensor shape
print("Original Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)


#Using PyTorch for Convolution
import torch
import torch.nn as nn

# Create a sample input tensor (batch size, channels, height, width)
input_tensor_torch = torch.rand(1, 1, 10, 10)  # Example shape (batch_size, channels, height, width)

# Defining a Convolutional Layer
conv_layer_torch = nn.Conv2d(
    in_channels=1,  # Number of input channels
    out_channels=1,  # Number of output channels
    kernel_size=3,  # Size of the convolution kernel
    stride=1,  # Stride of the convolution
    padding=1  # Padding type
)

# Applying the convolutional layer to the input tensor
output_tensor_torch = conv_layer_torch(input_tensor_torch)

# Displaying the output tensor shape
print("Original Tensor Shape (PyTorch):", input_tensor_torch.shape)
print("Output Tensor Shape (PyTorch):", output_tensor_torch.shape)
