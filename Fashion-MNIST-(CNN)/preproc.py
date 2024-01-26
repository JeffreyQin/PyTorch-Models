# for data
import pandas as pd 
import numpy as np 
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt 

num_images = 2000
# load dataset
train = pd.read_csv('dataset/train_LbELtWX/train.csv')
test = pd.read_csv('dataset/test_ScVgIM0/test.csv')

train_img = []
for img_name in tqdm(range(1,num_images + 1)):
    img_path = 'dataset/train_LbELtWX/train/' + str(img_name) + '.png'
    img = imread(img_path, as_gray=True)
    img /= 255.0
    img = img.astype('float32')
    train_img.append(img)
trainX = np.array(train_img) # trainX.shape = [60000, 28, 28]
trainY = train['label'].values[0:num_images] # .values is required for accessing values in pandas DataFrame

"""
test_img = []
for img_name in tqdm(test['id']):
    img_path = 'dataset/test_ScVgIM0/test/' + str(img_name) + '.png'
    img = imread(img_path, as_gray=True)
    img /= 255.0
    img = img.astype('float32')
    test_img.append(img)
testX = np.array(test_img)
testY = test['labels'].values
"""

# split dataset into train and test
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size = 0.1)

# convert dataset into torch tensor format
trainX = trainX.reshape(int(0.9 * num_images), 1, 28, 28)
trainX = torch.from_numpy(trainX)
trainY = trainY.astype(int)
trainY = torch.from_numpy(trainY) # shape: [54000]

print('arrived')
valX = valX.reshape(int(0.1 * num_images), 1, 28, 28)
valX = torch.from_numpy(valX)
valY = valY.astype(int)
valY = torch.from_numpy(valY)
