# Image Classifier
Image classifier with PyTorch.

The project consists of two main file:
  - train.py: will train a new network on a dataset and save the model as a checkpoint.
  - predict.py: uses a trained network to predict the class for an input image.
  - cat_to_name.json: json object mapping flowers category names to the real names

# Running the project
1. To traing flower images in the command prompt type:
```
python train.py data_directory
```
#### Options:
Set directory to save checkpoints: 
```
python train.py data_dir --save_dir save_directory
```
Choose architecture: 
```
python train.py data_dir --arch "vgg13"
```
Set hyperparameters: 
```
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
```
Use GPU for training: 
```
python train.py data_dir --gpu
```

2. Predict flower name from an image in the command prompt type:
```
python predict.py /path/to/image checkpoint
```
#### Options:
Return top KK most likely classes: 
```
python predict.py input checkpoint --top_k 3
```

Use a mapping of categories to real names: 
```
python predict.py input checkpoint --category_names cat_to_name.json
```
Use GPU for inference: 
```
python predict.py input checkpoint --gpu
```

# Acknowlegdment
This work has been part of Udacity DataScience Nano Master's Degree.
