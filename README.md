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
```

### Example

Here is an example of how to use the ViolenceClass interface:
1. Ensure you have a trained model checkpoint file, e.g., efficientnet_b3.ckpt.
2. Create a script to use the ViolenceClass:

```python
import torch
from classify import ViolenceClass

# Path to the model checkpoint file
model_checkpoint_path = 'path/to/efficientnet_b3.ckpt'

# Instantiate the classifier
classifier = ViolenceClass(model_checkpoint_path)

# Create a batch of example images (n, 3, 224, 224)
example_images = torch.rand(4, 3, 224, 224)  # Example input: 4 randomly generated images

# Classify the images
predictions = classifier.classify(example_images)

# Print the predictions
print(predictions)  # Output: [0, 1, 0, 1] (example output)
```

3. Run the script:

```sh
python example_script.py
```

## Requirements

* Python 3.8+
* torch
* torchvision
* efficientnet-pytorch
* pillow

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### `classify.py`

```python
import torch
from torch import nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import os

class ViolenceClass:
    def __init__(self, model_checkpoint_path: str, device: str = 'cuda:0'):
        # 设置设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2, advprop=True)
        self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 定义预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def classify(self, imgs: torch.Tensor) -> list:
        # 确保输入是一个torch.Tensor，并且已经在0-1范围内
        if not isinstance(imgs, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")
        
        if imgs.dim() != 4 or imgs.size(1) != 3 or imgs.size(2) != 224 or imgs.size(3) != 224:
            raise ValueError("Input tensor should be of shape (n, 3, 224, 224)")
        
        # 将输入tensor移动到目标设备
        imgs = imgs.to(self.device)
        
        # 关闭梯度计算
        with torch.no_grad():
            # 模型推理
            outputs = self.model(imgs)
            _, preds = torch.max(outputs, 1)
        
        # 将结果转换为python列表
        return preds.cpu().tolist()

```

### 创建示例脚本

创建一个 example_script.py 文件，用于示例调用：

```python
import torch
from classify import ViolenceClass

# Path to the model checkpoint file
model_checkpoint_path = 'path/to/efficientnet_b3.ckpt'

# Instantiate the classifier
classifier = ViolenceClass(model_checkpoint_path)

# Create a batch of example images (n, 3, 224, 224)
example_images = torch.rand(4, 3, 224, 224)  # Example input: 4 randomly generated images

# Classify the images
predictions = classifier.classify(example_images)

# Print the predictions
print(predictions)  # Output: [0, 1, 0, 1] (example output)
```
