# Pet Image Segmentation Project

This project implements and compares different deep learning models for pet image segmentation using the Oxford-IIIT Pet Dataset.

## Features

- Multiple model architectures support (UNet, UNet++, FPN, DeepLabV3)
- Training pipeline with data augmentation
- Visualization tools for segmentation results
- Metrics tracking (Dice coefficient, IoU, pixel accuracy)


## Requirements

- Python 3.8+
- PyTorch 2.0+
- segmentation-models-pytorch
- torchvision>=0.15.0
- Pillow
- tqdm
- torchmetrics
- tensorboard

## Usage

### Using the Training Notebook

The complete training pipeline is in `train_model.ipynb`. This notebook demonstrates:
- Data loading and preprocessing
- Model training
- Results visualization
- Model comparison


## Model Architectures

The project includes several segmentation architectures:

1. UNet (Default ResNet34 backbone)
2. UNet++ (DenseNet121 backbone)
3. FPN (Feature Pyramid Network)
4. DeepLabV3 (ResNet34/50 backbone)

## Results

Performance metrics for different models:

| Model | Dice Score | mIoU | Pixel Accuracy |
|-------|------------|------|----------------|
| UNet34  | 0.8639      | 0.7865 | 0.9230          |
| UNet50  | 0.8763      | 0.8045 | 0.9338          |
| FPN   | 0.8678      | 0.7941 | 0.9300          |
| DeepLabV3| 0.8716   | 0.7998 | 0.9315          |
| DeepLabV3-50| 0.8747   | 0.8038 | 0.9333          |

## License

This project is released under the MIT License.


