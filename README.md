# Implement Stable Diffusion with Keras and Tensorflow

The following are the prerequisites and the steps to implement Stable Diffusion with Keras and Tensorflow.

## Prerequisites

* Provision a [Vultr Cloud GPU Server](https://my.vultr.com/deploy/?cloudgpu) using the [NVIDIA NGC](https://www.vultr.com/marketplace/apps/nvidia-ngc/) Marketplace Application

## Set Up the Environment

Deploy a container using the `tensorflow/tensorflow:latest-gpu-jupyter` image.

```console
# docker run -p 8888:8888 --gpus all -it --rm -v /root/notebooks:/tf/notebooks tensorflow/tensorflow:latest-gpu-jupyter
```

Open the Jupyter Notebook interface on your web browser using the link provided in the output.

Create a new blank notebook named **Stable Diffusion**.

Install the required Python packages.

```console
!pip install keras_cv tensorflow_datasets
```

## Implement the Stable Diffusion Model

Import the required modules.

```python
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt
```

Initialize the model.

```python
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
```

Generate an image batch using the `model.text_to_image()` function.

```python
images = model.text_to_image('cat in space', batch_size=3)
```

Declare a new fuction to plot the images in the notebook.

```python
def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()
```

Plot the generated image batch using the `plot_images()` function.

```python
plot_images(images)
```

Benchmark the performance.

```python
import time

t0 = time.time()
images = model.text_to_image('horse running on the moon', batch_size=3)
t1 = time.time()

print(f'Total Time Taken: {t1-t0}')

plot_images(images)
```

## Implement Mixed Precision

Restart the Python kernel and clear all output.

Add the following line before initializing the model.

```python
keras.mixed_precision.set_global_policy("mixed_float16")
```

Run the block containing the benchmarking code.

## Implement XLA Compilation

Restart the Python kernel and clear all output.

Add the `jit_compile` parameter while initializing the model.

```python
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=True)
```

Run the block containing the benchmarking code.
