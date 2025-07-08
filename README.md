# Understanding Convolutional Layers: NumPy, TensorFlow, and PyTorch

This project explores the concept of convolution, a fundamental operation in image processing and convolutional neural networks (CNNs). It demonstrates how to perform convolutions using raw NumPy and SciPy, and then illustrates the implementation of convolutional layers in both TensorFlow and PyTorch.

## Project Structure

The project code is divided into three main sections:

### 1. Manual Convolution with NumPy and SciPy (Commented Out)

This commented-out section provides a foundational understanding of how convolution works at a lower level. It performs image convolution using `scipy.ndimage.convolve` with predefined kernels.

**Key operations:**

* **Image Loading**: Loads a simple 10x10 grayscale image (randomly generated for demonstration purposes).
* **Kernel Definition**: Defines two common convolution kernels:
    * **Edge Detection Kernel**: A 3x3 kernel designed to highlight edges in an image.
    * **Blur Kernel**: A 3x3 kernel used to blur an image by averaging pixel values.
* **Applying Convolution**: Uses `scipy.ndimage.convolve` to apply these kernels to the image.
* **Visualization**: Displays the original, edge-detected, and blurred images using `matplotlib.pyplot`, allowing for a visual comparison of the effects of convolution.

To see this in action, uncomment the code block in your local environment.

### 2. Convolutional Layer with TensorFlow

This section demonstrates how to define and use a `Conv2D` layer in TensorFlow's Keras API.

**Key concepts:**

* **Input Tensor**: A sample input tensor is created with the shape `(batch_size, height, width, channels)`. For a single grayscale image, this would be `(1, 10, 10, 1)`.
* **`tf.keras.layers.Conv2D`**:
    * `filters=1`: Specifies the number of output filters (or feature maps). In this example, it's set to 1, meaning the output will have one channel.
    * `kernel_size=(3, 3)`: Defines the dimensions of the convolution window (3x3 pixels).
    * `strides=(1, 1)`: The step size of the convolution operation across the input. A stride of 1x1 means the kernel moves one pixel at a time horizontally and vertically.
    * `padding='same'`: Ensures that the output feature map has the same spatial dimensions (height and width) as the input by adding appropriate zero-padding.
* **Applying the Layer**: The `conv_layer` is applied to the `input_tensor`, producing an `output_tensor`.
* **Shape Display**: The original and output tensor shapes are printed, illustrating how the convolutional layer transforms the input's dimensions.

### 3. Convolutional Layer with PyTorch

This section illustrates the implementation of a convolutional layer in PyTorch using `torch.nn.Conv2d`.

**Key concepts:**

* **Input Tensor**: A sample input tensor is created with the shape `(batch_size, channels, height, width)`. This is the standard channel-first convention in PyTorch, so for a single grayscale image, it would be `(1, 1, 10, 10)`.
* **`torch.nn.Conv2d`**:
    * `in_channels=1`: The number of channels in the input image.
    * `out_channels=1`: The number of channels produced by the convolution (number of filters).
    * `kernel_size=3`: The size of the square convolution kernel (3x3).
    * `stride=1`: The step size of the convolution.
    * `padding=1`: The amount of zero-padding added to both sides of the input. A padding of 1 for a 3x3 kernel with stride 1 ensures that the output feature map has the same spatial dimensions as the input.
* **Applying the Layer**: The `conv_layer_torch` is applied to the `input_tensor_torch`, resulting in `output_tensor_torch`.
* **Shape Display**: The original and output tensor shapes are printed, highlighting PyTorch's channel-first convention and the dimension transformation.

## Prerequisites

To run this code, you will need:

* Python 3.x
* `numpy`
* `scipy` (for the commented-out section)
* `matplotlib` (for the commented-out section)
* `tensorflow`
* `torch`

You can install these libraries using pip:

```bash
pip install numpy scipy matplotlib tensorflow torch