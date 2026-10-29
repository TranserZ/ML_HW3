import numpy as np
import torch
from models.LogisticRegression import LogisticRegression
import pandas as pd
import os
from utils.utils import load_data_mnist, augment_data , accuracy
np.random.seed(42)
torch.manual_seed(42)

# ======= Set these paths to your dataset =======
dir_name = 'data_mnist' #fix
epochs = 1000           #fix   
learning_rate  = 0.1    #fix


lambda_reg = 0.01   # [0 0.1 0.01 0.001]
augment= False # True or False
early_stopping=True   # True or False
patience = 5

X_train,y_train , X_val , y_val , X_test ,y_test =  load_data_mnist(dir_name)
model = LogisticRegression(num_features=784, num_classes= 10)



# Evaluate both models on the same original training samples.
X_train_original = X_train.copy()
y_train_original = y_train.copy()

if augment:
    X_train , y_train = augment_data(X_train,y_train)
    
history_overfit = model.train(X=X_train, y=y_train, X_val=X_val, y_val=y_val, learning_rate = learning_rate ,epochs=epochs , early_stopping=early_stopping, lambda_reg = lambda_reg, patience=patience)
train_acc = accuracy(y_train_original, model.predict(X_train_original))
test_acc = accuracy(y_test, model.predict(X_test))
print(f"Final Train Loss: {history_overfit['train_loss']:.4f} - Val Loss: {history_overfit['val_loss']:.4f}")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train-Test Gap: {train_acc - test_acc:.4f}")
