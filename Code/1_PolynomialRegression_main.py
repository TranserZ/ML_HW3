# main.py
import numpy as np
import pandas as pd
from sympy import false, true
import torch
from models.PolynomialRegression import PolynomialRegression
import os
import random

# Set seed for reproducibility
seed = 42
random.seed(seed)              # 파이썬 기본 random
np.random.seed(seed)           # NumPy random
torch.manual_seed(seed)        # PyTorch random

# ======= File paths =======
TRAIN_PATH = 'data_poly/train.csv'
VAL_PATH = 'data_poly/val.csv'
TEST_PATH = 'data_poly/test.csv'

# ======= Load data =======
def load_csv(path):
    data = pd.read_csv(path)
    x = data[['x']].values.astype(np.float32)
    y = data['y'].values.astype(np.float32)
    return x, y

train_x, train_y = load_csv(TRAIN_PATH)
val_x, val_y = load_csv(VAL_PATH)
test_x, test_y = load_csv(TEST_PATH)

# ======= Hyperparameters =======
batch_size = 128
num_epochs = 50000
learning_rate = 5e-2
degree = 100
lambda_reg = 0.0

# ======= Train model: No regularization, no early stopping =======
# print("\n[Training: No Regularization, No Early Stopping]")
# model_plain = PolynomialRegression(degree=degree)
# model_plain.train(train_x, train_y, batch_size, num_epochs, learning_rate)

# # ======= Train model: With early stopping =======
print("\n[Training: Early Stopping]")
model_early = PolynomialRegression(degree=degree)
model_early.train(train_x, train_y, batch_size, num_epochs, learning_rate,
                  val_x=val_x, val_y=val_y, early_stopping=true, patience=25)

# # ======= Train model: With regularization =======
# print("\n[Training: L2 Regularization]")
# model_reg = PolynomialRegression(degree=degree, reg_method='l2', reg_lambda=lambda_reg)
# model_reg.train(train_x, train_y, batch_size, num_epochs, learning_rate)

# ======= Evaluation =======
# You need to change model name
# rmse_train = model_plain.eval(train_x, train_y, batch_size)
# rmse_val = model_plain.eval(val_x, val_y, batch_size)
# rmse_test = model_plain.eval(test_x, test_y, batch_size)

rmse_train = model_early.eval(train_x, train_y, batch_size)
rmse_val = model_early.eval(val_x, val_y, batch_size)
rmse_test = model_early.eval(test_x, test_y, batch_size)

# ======= Print results =======
print("\n[Final RMSE Results]")
print(f"Train RMSE: {rmse_train:.4f}")
print(f"Val RMSE:   {rmse_val:.4f}")
print(f"Test RMSE:  {rmse_test:.4f}")