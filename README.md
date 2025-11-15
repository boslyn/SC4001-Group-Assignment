# SC4001 Group Assignment

Repository for Nanyang Technological University SC4001 Neural Network & Deep Learning Group Project.

## Group Members
- Pang Boslyn
- Sally Ngui Yu Ying

## Project Overview: Flowers Recognition
This project implements a deep learning pipeline for classifying flower species from the Oxford Flowers 102 dataset, a collection of 102 flower categories commonly found in the UK. In this project, our objective is to develop and implement a classification model to classify the flower images. Beyond the standard classification models, we will be exploring advanced techniques to further improve the models developed. 

We explored and discussed various existing techniques. We first implemented 2 well-known Convolutional Neural Network (CNN) architectures, ResNet18 and VGG16. Then, we introduced several enhancements such as architectural enhancements, data augmentation and use of advanced loss function to improve accuracy and generalisation. Next, we examined Vision Transformer (ViT) models, beginning with fine tuning a pretrained ViT on the Flowers data as our baseline model. We then explored model improvements with hyperparameters fine-tuning strategies. Further improvements include Visual Prompt Tuning and exploring the use of advanced loss functions.

## Dataset
- The dataset is available in TorchVision
https://pytorch.org/vision/main/generated/torchvision.datasets.Flowers102.html
- The Oxford Flowers 102 Dataset https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

## Data Preparation
The Oxford Flowers dataset contains 102 flower categories, with each class consisting between 40 to 258 images. The train and validation set each consist of 10 images per class while the test set consists of at least 20 images per class. Both CNN and Transformers follow the following standardised preprocessing and loading pipeline. Images were resized to 224 x 224 pixels to be aligned with the input dimensions used by ImageNet-pretrained networks, such as ResNet and ViT. Next, the pixel data of images are then converted into PyTorch tensors for compatibility. Normalisation was then applied using the mean and standard deviation values from the ImageNet dataset, to ensure consistency between the data input and the pretrained models, improving convergence during fine-tuning. Each split was loaded using PyTorch DataLoader objects for efficient mini-batch training. During training, data shuffling was applied to the training set to introduce randomness, reducing the risk of overfitting to specific images. As for validation and test sets, they remained unshuffled as no training will be carried out.

## Models Tested in Report
| Report Section | Notebook |
| ------------------ | ------------------- |
| 3.1.1 Baseline ResNet18 Transfer Learning and Hyperparameter Tuning | [ResNet18_TransferLearning_FineTune_Frozen.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/ResNet18_TransferLearning_FineTune_Frozen.ipynb) |
| 3.1.2 ResNet18 with Different Number of Frozen Layers | [ResNet18_TransferLearning_FineTune_Frozen.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/ResNet18_TransferLearning_FineTune_Frozen.ipynb) |
| 3.1.3 ResNet18 MixUp and CutMix | [ResNet18_MixUp_CutMix.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/ResNet18_MixUp_CutMix.ipynb) |
| 3.1.4 ResNet18 Triplet Loss | [ResNet18_TripletLoss.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/ResNet18_TripletLoss.ipynb) |
| 3.1.5 ResNet18 FewShot CNN Prototypical Classification | [ResNet18_FewShot_ProtoNet.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/ResNet18_FewShot_ProtoNet.ipynb) |
| 3.2.1 Baseline VGG16-BN Transfer Learning and Fine-Tuning | [VGG16BN_Baselines_GAPvsFC_FrozenPartialFull.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/VGG16BN_Baselines_GAPvsFC_FrozenPartialFull.ipynb) | 
| 3.2.2 VGG16-BN with Attention Blocks | [VGG16BN_SE_or_CBAM_Attention.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/VGG16BN_SE_or_CBAM_Attention.ipynb) |
| 3.3 ResNet18 and VGG16-BN Hybrid | [ResNet_VGG16_Hybrid.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/ResNet_VGG16_Hybrid.ipynb) |
| 4.1 Fine-tune Pretrained Vision Transformer: ViT-B/16 | [Transformers_ViT.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/Transformers_ViT.ipynb) |
| 4.2 Hyperparameters Tuning of ViT Model | [Transformers_ViT.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/Transformers_ViT.ipynb) |
| 4.3 Visual Prompt Tuning (VPT) | [Transformer_VPT.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/Transformer_VPT.ipynb) |
| 4.4 Advanced Loss Functions (Label Smoothing Cross Entropy & ArcFace) | [Transformers_LabelSmoothingandArcFace.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/Transformers_LabelSmoothingandArcFace.ipynb) |

## Additional Models Tested 
| Model | Notebook |
| ------------------ | ------------------- |
| ResNet18 with Deformable Convolutions | [ResNet18_DeformableConv.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/other%20models/ResNet18_DeformableConv.ipynb) |
| EfficientNetB0 Transfer Learning: Baseline, Fine‑Tuning | [EfficientNetB0_Baselines_FrozenPartialFull_LLRD.ipynb](https://github.com/boslyn/SC4001-Group-Assignment/blob/main/other%20models/EfficientNetB0_Baselines_FrozenPartialFull_LLRD.ipynb) |


## Installation and Setup

### 1. Clone the Repository
```shell
https://github.com/boslyn/SC4001-Group-Assignment.git
cd SC4001-Group-Assignment
```

### 2. Create virtual environment
```
python3.12 -m venv .venv
```

### 3. Activate virtual environment on Windows
```
.venv\Scripts\activate
```

### 4. Install dependencies
```
pip install -r requirements.txt
```
