import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
import pandas as pd
import os

def augment_transform_image(image, angle_range=(-5, 5), shift_range=(-3, 3)):
    # 무작위 회전 각도 선택 (회전 범위 내)
    angle = np.random.uniform(*angle_range)
    rotated = rotate(image, angle, reshape=False, order=1, mode='reflect')
    
    # 수평 이동 (세로 이동은 적용하지 않음)
    shift_x = np.random.uniform(*shift_range)
    shifted = shift(rotated, shift=(0, shift_x), order=1, mode='reflect')
    
    return shifted



def augment_data(X_train,y_train, aug_factor=1):
    if hasattr(X_train, "to_numpy"):
        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()
    else:
        X_train_np = X_train
        y_train_np = y_train

    augmented_images = []
    for i in range(X_train_np.shape[0]):
        img = X_train_np[i].reshape(28, 28)
        for j in range(aug_factor):
            img_aug = augment_transform_image(img, angle_range=(0, 0), shift_range=(-4, 4))
            augmented_images.append(img_aug.flatten())

    X_train_augmented = np.array(augmented_images)
    new_X_train = np.vstack([X_train_np, X_train_augmented])
    new_y_train = np.concatenate([y_train_np, np.tile(y_train_np, aug_factor)])
    return new_X_train, new_y_train


def load_data_mnist(dir_name):
    path = os.path.join('./', dir_name)
    X_train = pd.read_csv(f"{path}/train_x.csv")
    y_train = pd.read_csv(f"{path}/train_y.csv").squeeze()  
    X_val = pd.read_csv(f"{path}/val_x.csv")
    y_val = pd.read_csv(f"{path}/val_y.csv").squeeze() 
    X_test = pd.read_csv(f"{path}/test_x.csv")
    y_test = pd.read_csv(f"{path}/test_y.csv").squeeze()  
    return X_train , y_train ,X_val , y_val , X_test , y_test



def accuracy(h, y):
    """
    h : (N, ), predicted label
    y : (N, ), correct label
    """

    total = h.shape[0]
    correct = len(np.where(h==y)[0])

    acc = correct / total

    return acc