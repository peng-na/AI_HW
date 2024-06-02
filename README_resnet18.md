# Violence Image Classification Interface

This repository contains the `ViolenceClass` interface for classifying images as either containing violence or not. The model is based on ResNet-18 and uses PyTorch for inference.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
    - [Interface Description](#interface-description)
    - [Example](#example)
- [Requirements](#requirements)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/violence-classification.git
    cd violence-classification
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Interface Description

The `ViolenceClass` interface provides a method to classify images. The main method is `classify`, which takes a batch of images as input and returns the predicted classes.

#### `ViolenceClass`

```python
class ViolenceClass:
    def __init__(self, model_checkpoint_path: str, device: str = 'cuda:0'):
        """
        Initialize the ViolenceClass.
        
        Args:
            model_checkpoint_path (str): Path to the model checkpoint file.
            device (str): Device to run the model on, default is 'cuda:0'.
        """
        ...

    def classify(self, imgs: torch.Tensor) -> list:
        """
        Classify the input images.
        
        Args:
            imgs (torch.Tensor): Input tensor of shape (n, 3, 224, 224) where n is the batch size.
        
        Returns:
            list: Predicted classes for each input image.
        """
        ...
```

### Example

Here is an example of how to use the ViolenceClass interface:
1.Ensure you have a trained model checkpoint file, e.g., resnet18_checkpoint.pth.
2.Create a script to use the ViolenceClass:

```python
import torch
from classify import ViolenceClass

# Path to the model checkpoint file
model_checkpoint_path = 'path/to/resnet18_checkpoint.pth'

# Instantiate the classifier
classifier = ViolenceClass(model_checkpoint_path)

# Create a batch of example images (n, 3, 224, 224)
example_images = torch.rand(4, 3, 224, 224)  # Example input: 4 randomly generated images

# Classify the images
predictions = classifier.classify(example_images)

# Print the predictions
print(predictions)  # Output: [0, 1, 0, 1] (example output)

```

3.Run the script

## Requiments

* Python 3.8+
* torch
* torchvision
* pillow



