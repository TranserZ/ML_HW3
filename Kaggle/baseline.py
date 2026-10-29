# 1. 필요한 라이브러리와 데이터 읽기

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# 같은 코드를 실행하면 같은 결과가 나오도록 난수를 고정합니다.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 현재 Python 파일이 있는 폴더에서 데이터를 찾습니다.
DATA = Path(__file__).resolve().parent
OUTPUT = DATA / "outputs"
OUTPUT.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "train_data.csv")
test = pd.read_csv(DATA / "test_data.csv")

feature_columns = [f"z_{i:02d}" for i in range(64)]
target_columns = [f"y_{i:03d}" for i in range(256)]

X = train[feature_columns].to_numpy(dtype=np.float32)
Y = train[target_columns].to_numpy(dtype=np.float32)
X_test = test[feature_columns].to_numpy(dtype=np.float32)
groups = train["group"].to_numpy()

print("학습 입력 / 정답:", X.shape, Y.shape)
print("평가 입력:", X_test.shape)


# 2. DNN 정의

class FeatureDNN(nn.Module):
    """64개 입력을 ECG 복원에 유용한 32개 feature로 바꾸는 DNN."""

    def __init__(self):
        super().__init__()

        # DNN 실험에서 가장 좋았던 64 → 128 → 64 → 32 구조입니다.
        # LeakyReLU는 음수 값도 조금 남겨 뉴런이 완전히 꺼지는 것을 막습니다.
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # 학습할 때만 32개 feature로 256개 ECG 값을 예측합니다.
        self.output_layer = nn.Linear(32, 256)

    def forward(self, x):
        new_features = self.encoder(x)
        return self.output_layer(new_features)


def train_with_early_stopping(X_train, Y_train, X_val, Y_val):
    """검증 오차가 좋아지지 않으면 학습을 일찍 멈춥니다."""

    model = FeatureDNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_function = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_epoch = 1
    best_weights = None
    waiting = 0
    patience = 50

    for epoch in range(1, 5001):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_function(model(X_train_tensor), Y_train_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_function(model(X_val_tensor), Y_val_tensor).item()

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break

    # 검증 결과가 가장 좋았던 시점으로 되돌립니다.
    model.load_state_dict(best_weights)
    return model, best_epoch, np.sqrt(best_val_loss)


def train_with_all_data(X_train, Y_train, epochs):
    """검증에서 찾은 epoch 수만큼 전체 학습 데이터로 다시 학습합니다."""

    torch.manual_seed(SEED)
    model = FeatureDNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_function = nn.MSELoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_train, dtype=torch.float32)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(X_tensor), Y_tensor)
        loss.backward()
        optimizer.step()

    return model


def make_dnn_features(model, X_data):
    """학습된 encoder에서 새로운 32개 feature만 꺼냅니다."""

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        return model.encoder(X_tensor).numpy()


# 3. 사람 단위로 학습 데이터와 검증 데이터 나누기

# 같은 사람의 데이터가 양쪽에 섞이지 않도록 group을 기준으로 나눕니다.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_index, val_index = next(splitter.split(X, Y, groups=groups))

# scaler는 학습 부분으로만 학습하여 검증 정보가 새어 들어가는 것을 막습니다.
validation_scaler = StandardScaler()
X_train_scaled = validation_scaler.fit_transform(X[train_index]).astype(np.float32)
X_val_scaled = validation_scaler.transform(X[val_index]).astype(np.float32)

dnn, best_epoch, dnn_val_rmse = train_with_early_stopping(
    X_train_scaled,
    Y[train_index],
    X_val_scaled,
    Y[val_index],
)

print(f"DNN 학습 epoch: {best_epoch}")
print(f"DNN 검증 RMSE: {dnn_val_rmse:.6f} mV")


# 4. DNN feature에 2차 다항 회귀 적용하기

train_dnn_features = make_dnn_features(dnn, X_train_scaled)
val_dnn_features = make_dnn_features(dnn, X_val_scaled)

# 32개 feature에 제곱항과 교차항을 추가하면 총 560개가 됩니다.
validation_poly = PolynomialFeatures(degree=2, include_bias=False)
train_poly_features = validation_poly.fit_transform(train_dnn_features)
val_poly_features = validation_poly.transform(val_dnn_features)

# Ridge는 큰 가중치에 벌점을 주어 과적합을 줄이는 회귀 모델입니다.
# 몇 가지 alpha 중 검증 RMSE가 가장 낮은 값을 고릅니다.
best_alpha = None
best_ridge_rmse = float("inf")

for alpha in [0.1, 1.0, 10.0, 100.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_poly_features, Y[train_index])
    val_prediction = ridge.predict(val_poly_features)
    val_rmse = np.sqrt(np.mean((Y[val_index] - val_prediction) ** 2))

    if val_rmse < best_ridge_rmse:
        best_alpha = alpha
        best_ridge_rmse = val_rmse

print(f"선택된 Ridge alpha: {best_alpha}")
print(f"DNN feature + 2차 Ridge 검증 RMSE: {best_ridge_rmse:.6f} mV")


# 5. 전체 학습 데이터로 최종 모델 학습 및 테스트 예측

# 최종 모델에서는 100개의 학습 데이터를 모두 사용합니다.
final_scaler = StandardScaler()
X_all_scaled = final_scaler.fit_transform(X).astype(np.float32)
X_test_scaled = final_scaler.transform(X_test).astype(np.float32)

final_dnn = train_with_all_data(X_all_scaled, Y, best_epoch)
all_dnn_features = make_dnn_features(final_dnn, X_all_scaled)
test_dnn_features = make_dnn_features(final_dnn, X_test_scaled)

final_poly = PolynomialFeatures(degree=2, include_bias=False)
all_poly_features = final_poly.fit_transform(all_dnn_features)
test_poly_features = final_poly.transform(test_dnn_features)

final_ridge = Ridge(alpha=best_alpha)
final_ridge.fit(all_poly_features, Y)
predictions = final_ridge.predict(test_poly_features)

train_predictions = final_ridge.predict(all_poly_features)
train_rmse = np.sqrt(np.mean((Y - train_predictions) ** 2))

print(f"최종 Train RMSE: {train_rmse:.6f} mV")
print("예측 결과:", predictions.shape)


# 6. 새 feature와 Kaggle 제출 파일 저장

new_feature_columns = [f"dnn_feature_{i:02d}" for i in range(32)]

train_feature_file = pd.DataFrame(all_dnn_features, columns=new_feature_columns)
train_feature_file.insert(0, "id", train["id"].to_numpy())
train_feature_file.to_csv(OUTPUT / "dnn_train_features.csv", index=False)

test_feature_file = pd.DataFrame(test_dnn_features, columns=new_feature_columns)
test_feature_file.insert(0, "id", test["id"].to_numpy())
test_feature_file.to_csv(OUTPUT / "dnn_test_features.csv", index=False)

# test의 각 행에서 나온 256개 예측값을 한 줄로 길게 펼칩니다.
submission_ids = (
    test["id"].to_numpy()[:, None] * 256 + np.arange(256)
).ravel()

submission = pd.DataFrame({
    "id": submission_ids,
    "prediction": predictions.ravel(),
})
submission.to_csv(OUTPUT / "dnn_submission.csv", index=False, float_format="%.12g")

print("새 feature 저장:", OUTPUT / "dnn_train_features.csv")
print("새 feature 저장:", OUTPUT / "dnn_test_features.csv")
print("Kaggle 제출 파일 저장:", OUTPUT / "dnn_submission.csv")
print("제출 행 수:", len(submission))
