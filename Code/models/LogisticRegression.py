import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

def to_tensor(x, dtype=torch.float32):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x = x.values
    return torch.tensor(np.array(x), dtype=dtype)


class LogisticRegression(nn.Module):
    def __init__(self, num_features=784, num_classes=10):
        super(LogisticRegression, self).__init__()
        self.W = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.W(x)  # logits (no softmax here)

    def l2_regularization(self):
        loss = None
        # ========== STUDENT CODE HERE ==========
        loss = torch.sum(self.W.weight ** 2)
        # ========== STUDENT CODE HERE ==========
        return loss

    # X, y: training data and labels
    # X_val, y_val: validation data and labels. Used for early stopping judgement.
    # learning_rate: Learning rate for the optimizer (SGD) = 0.1
    # epochs: # of epochs to train for = 1000
    # early_stopping: whether to use early stopping or not = True
    # lambda_reg: L2 regularization strength = 0.01
    # patience: number of epochs to wait for improvement in validation loss
    def train(self, X, y, X_val, y_val, learning_rate=0.1, epochs=1000, 
                     early_stopping=True, lambda_reg= 0.01, patience=5 ):

        # Convert to tensors
        X_train = to_tensor(X, dtype=torch.float32)
        y_train = to_tensor(y, dtype=torch.long)
        X_val = to_tensor(X_val, dtype=torch.float32)
        y_val = to_tensor(y_val, dtype=torch.long)
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        # CrossEntropyLoss (softmax + negative log-likelihood)
        criterion = nn.CrossEntropyLoss()

        # 지금까지 가장 낮았던 검증 손실. 처음엔 inf
        best_val_loss = float('inf')
        # 검증 손실이 개선되지 않은 연속 epoch 수
        no_improve = 0
        # 검증 손실이 가장 낮았을 때의 파라미터 스냅샷
        best_weights = None
        # epoch별 손실 기록
        train_losses, val_losses = [], []
        lambda_reg  = float(lambda_reg)
        for epoch in range(epochs):
            optimizer.zero_grad()
            # training data의 logits
            outputs = self.forward(X_train)
            # training loss
            loss = criterion(outputs, y_train)
            if lambda_reg>0:
                l2_norm = self.l2_regularization()
                loss += lambda_reg * l2_norm

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Validation
            with torch.no_grad():
                # validation data의 logits
                val_outputs = self.forward(X_val)
                # validation loss
                val_loss = criterion(val_outputs, y_val)
                if lambda_reg>0:
                    val_loss += lambda_reg * self.l2_regularization()
                val_losses.append(val_loss.item())

            if early_stopping:
                # ========== STUDENT CODE HERE ==========
                #use val_loss, best_val_loss, best_weights, no_imporve
                if(val_loss < best_val_loss):
                    best_val_loss = val_loss
                    best_weights = {i: v.clone() for i, v in self.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if(no_improve >= patience):
                        break
                # ========== STUDENT CODE HERE ==========


        if early_stopping and best_weights is not None:
            self.load_state_dict(best_weights)

        # Report losses for the final (possibly restored) model.
        with torch.no_grad():
            train_loss = criterion(self.forward(X_train), y_train)
            val_loss = criterion(self.forward(X_val), y_val)
            if lambda_reg > 0:
                penalty = lambda_reg * self.l2_regularization()
                train_loss = train_loss + penalty
                val_loss = val_loss + penalty

        return {
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item(),
        }

    # 각 샘플에서 확률이 가장 높은 클래스를 골라 numpy 배열로 돌려준다. 
    def predict(self, X):
        X_tensor = to_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.forward(X_tensor)
            return torch.argmax(F.softmax(logits, dim=1), dim=1).numpy()
