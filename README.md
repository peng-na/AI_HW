# Violence Image Classification Interface

This repository contains the `ViolenceClass` interface for classifying images as either containing violence or not. The model is based on EfficientNet-B3 and uses PyTorch for inference.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Interface Description](#interface-description)
- [Example](#example)
- [Requirements](#requirements)
- [License](#license)

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
