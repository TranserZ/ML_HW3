"""DNN feature 생성 설정을 단계별로 탐색하는 실험 코드.

테스트 데이터는 사용하지 않습니다. 각 설정은 사람(group) 단위로 나눈
검증 데이터에서 평가하며, 결과가 좋은 후보만 다음 단계로 넘깁니다.
"""

from datetime import datetime
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# 작은 행렬에서는 CPU thread를 하나만 쓰는 편이 더 빠르고 재현하기 쉽습니다.
torch.set_num_threads(1)

SEED = 42
DATA = Path(__file__).resolve().parent
RUN_NAME = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RESULT_DIR = DATA / "outputs" / "dnn_experiments" / RUN_NAME
RESULT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "train_data.csv")
feature_columns = [f"z_{i:02d}" for i in range(64)]
target_columns = [f"y_{i:03d}" for i in range(256)]

X = train[feature_columns].to_numpy(dtype=np.float32)
Y = train[target_columns].to_numpy(dtype=np.float32)
groups = train["group"].to_numpy()


def activation_layer(name):
    """문자열 이름에 맞는 활성화 함수를 만듭니다."""

    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "linear":
        return nn.Identity()
    raise ValueError(f"알 수 없는 activation: {name}")


class FeatureDNN(nn.Module):
    """설정값에 따라 encoder의 크기와 모양을 바꿀 수 있는 DNN."""

    def __init__(self, config):
        super().__init__()
        layers = []
        input_size = 64

        # hidden_layers가 [64, 32]라면 64 → 64 → 32 순서로 만듭니다.
        for hidden_size in config["hidden_layers"]:
            layers.append(nn.Linear(input_size, hidden_size))
            if config["layer_norm"]:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(activation_layer(config["hidden_activation"]))
            if config["dropout"] > 0:
                layers.append(nn.Dropout(config["dropout"]))
            input_size = hidden_size

        # 이 층의 출력이 새롭게 생성되는 bottleneck feature입니다.
        layers.append(nn.Linear(input_size, config["bottleneck_size"]))
        layers.append(activation_layer(config["bottleneck_activation"]))
        self.encoder = nn.Sequential(*layers)

        # DNN을 학습시키기 위한 임시 ECG 예측층입니다.
        self.output_layer = nn.Linear(config["bottleneck_size"], 256)

    def forward(self, x):
        return self.output_layer(self.encoder(x))


def train_one_model(X_train, Y_train, X_val, Y_val, config, seed):
    """DNN 하나를 early stopping으로 학습합니다."""

    torch.manual_seed(seed)
    model = FeatureDNN(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_function = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_epoch = 1
    best_weights = None
    waiting = 0

    validation_interval = 5

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(X_train_tensor), Y_train_tensor)
        loss.backward()
        optimizer.step()

        # 검증을 5 epoch마다 하여 넓은 탐색의 실행 시간을 줄입니다.
        if epoch % validation_interval != 0 and epoch != config["max_epochs"]:
            continue

        model.eval()
        with torch.no_grad():
            val_loss = loss_function(model(X_val_tensor), Y_val_tensor).item()

        if val_loss < best_val_loss - config["min_delta"]:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            waiting = 0
        else:
            waiting += validation_interval
            if waiting >= config["patience"]:
                break

    model.load_state_dict(best_weights)
    return model, best_epoch, np.sqrt(best_val_loss)


def make_features(model, X_data):
    """학습된 encoder에서 bottleneck feature를 꺼냅니다."""

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        return model.encoder(X_tensor).numpy()


def evaluate_config(config, n_splits, seeds):
    """하나의 설정을 GroupKFold로 평가합니다."""

    rows = []
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(X, Y, groups=groups))

    for seed in seeds:
        for fold, (train_index, val_index) in enumerate(splits, start=1):
            start_time = time.perf_counter()

            # 각 fold의 학습 데이터만 이용해 표준화합니다.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_index]).astype(np.float32)
            X_val = scaler.transform(X[val_index]).astype(np.float32)

            model, best_epoch, dnn_rmse = train_one_model(
                X_train,
                Y[train_index],
                X_val,
                Y[val_index],
                config,
                seed + fold,
            )

            train_features = make_features(model, X_train)
            val_features = make_features(model, X_val)

            # DNN feature에 현재의 2차 다항 Ridge 설정을 그대로 적용합니다.
            poly = PolynomialFeatures(degree=2, include_bias=False)
            train_poly = poly.fit_transform(train_features)
            val_poly = poly.transform(val_features)

            ridge = Ridge(alpha=100.0)
            ridge.fit(train_poly, Y[train_index])
            val_prediction = ridge.predict(val_poly)
            ridge_rmse = np.sqrt(np.mean((Y[val_index] - val_prediction) ** 2))

            rows.append({
                "seed": seed,
                "fold": fold,
                "best_epoch": best_epoch,
                "dnn_val_rmse": float(dnn_rmse),
                "ridge_val_rmse": float(ridge_rmse),
                "training_seconds": time.perf_counter() - start_time,
            })

    return rows


def config_to_columns(config):
    """설정 dictionary를 CSV에 저장하기 쉬운 형태로 바꿉니다."""

    return {
        "hidden_layers": "-".join(map(str, config["hidden_layers"])),
        "bottleneck_size": config["bottleneck_size"],
        "hidden_activation": config["hidden_activation"],
        "bottleneck_activation": config["bottleneck_activation"],
        "layer_norm": config["layer_norm"],
        "dropout": config["dropout"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "patience": config["patience"],
        "min_delta": config["min_delta"],
        "max_epochs": config["max_epochs"],
    }


def config_key(config):
    """같은 설정이 중복 평가되지 않도록 비교용 문자열을 만듭니다."""

    return json.dumps(config, sort_keys=True)


def unique_configs(configs):
    result = []
    seen = set()
    for config in configs:
        key = config_key(config)
        if key not in seen:
            seen.add(key)
            result.append(config)
    return result


def changed(config, **updates):
    """기존 설정을 복사하고 원하는 항목만 바꿉니다."""

    new_config = config.copy()
    new_config.update(updates)
    return new_config


def run_stage(stage_name, configs, n_splits=3, seeds=(SEED,)):
    """한 단계의 후보들을 평가하고 RMSE 순서로 정렬합니다."""

    configs = unique_configs(configs)
    detail_rows = []
    summary_rows = []
    print(f"\n[{stage_name}] 후보 {len(configs)}개 평가 시작")

    for number, config in enumerate(configs, start=1):
        rows = evaluate_config(config, n_splits=n_splits, seeds=seeds)
        ridge_scores = [row["ridge_val_rmse"] for row in rows]
        dnn_scores = [row["dnn_val_rmse"] for row in rows]
        settings = config_to_columns(config)

        for row in rows:
            detail_rows.append({
                "stage": stage_name,
                "candidate": number,
                **settings,
                **row,
            })

        summary = {
            "stage": stage_name,
            "candidate": number,
            **settings,
            "mean_ridge_rmse": float(np.mean(ridge_scores)),
            "std_ridge_rmse": float(np.std(ridge_scores)),
            "mean_dnn_rmse": float(np.mean(dnn_scores)),
            "mean_best_epoch": float(np.mean([r["best_epoch"] for r in rows])),
            "mean_training_seconds": float(
                np.mean([r["training_seconds"] for r in rows])
            ),
            "config": config,
        }
        summary_rows.append(summary)

        print(
            f"  {number:>3}/{len(configs)} "
            f"Ridge RMSE={summary['mean_ridge_rmse']:.6f} "
            f"DNN RMSE={summary['mean_dnn_rmse']:.6f}"
        )

        # 실행 중간에도 현재까지의 결과를 확인할 수 있도록 계속 저장합니다.
        pd.DataFrame(detail_rows).to_csv(
            RESULT_DIR / f"{stage_name}_details.csv", index=False
        )
        save_summary = [{k: v for k, v in item.items() if k != "config"}
                        for item in summary_rows]
        pd.DataFrame(save_summary).to_csv(
            RESULT_DIR / f"{stage_name}_summary.csv", index=False
        )

    summary_rows.sort(
        key=lambda row: (row["mean_ridge_rmse"], row["std_ridge_rmse"])
    )
    best = summary_rows[0]
    print(
        f"[{stage_name}] 최고 Ridge RMSE={best['mean_ridge_rmse']:.6f}, "
        f"설정={config_to_columns(best['config'])}"
    )
    return summary_rows


def sampled_variants(parent_results, choices, sample_count, seed):
    """가능한 조합 중 일부를 고르게 무작위 추출합니다."""

    parent_configs = [result["config"] for result in parent_results]
    candidates = []
    for parent in parent_results:
        base = parent["config"]
        values = [choices[name] for name in choices]
        for combination in product(*values):
            updates = dict(zip(choices, combination))
            candidates.append(changed(base, **updates))

    candidates = unique_configs(candidates)
    if len(candidates) <= sample_count:
        return candidates

    # 이전 단계의 좋은 설정은 반드시 포함하고, 나머지만 무작위로 고릅니다.
    selected_configs = unique_configs(parent_configs)
    selected_keys = {config_key(config) for config in selected_configs}
    remaining = [
        config for config in candidates if config_key(config) not in selected_keys
    ]
    rng = np.random.default_rng(seed)
    needed = sample_count - len(selected_configs)
    selected = rng.choice(len(remaining), size=needed, replace=False)
    selected_configs.extend(remaining[index] for index in selected)
    return selected_configs


# 현재 baseline.py의 DNN 설정입니다.
base_config = {
    "hidden_layers": [32],
    "bottleneck_size": 8,
    "hidden_activation": "relu",
    "bottleneck_activation": "relu",
    "layer_norm": False,
    "dropout": 0.0,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    # 초기 탐색에서는 짧게 비교하고, 최종 후보에서 충분히 오래 학습합니다.
    "patience": 50,
    "min_delta": 1e-7,
    "max_epochs": 1200,
}


# 1단계: 생성할 feature 개수를 넓게 비교합니다.
stage1_configs = [
    changed(base_config, bottleneck_size=size)
    for size in [2, 4, 6, 8, 12, 16, 24, 32]
]
stage1 = run_stage("stage1_bottleneck", stage1_configs, n_splits=2)


# 2단계: 좋은 bottleneck 상위 3개에서 DNN의 깊이와 너비를 비교합니다.
hidden_layer_choices = [
    [16],
    [32],
    [64],
    [128],
    [64, 32],
    [128, 64],
    [128, 64, 32],
]
stage2_configs = [
    changed(result["config"], hidden_layers=layers)
    for result in stage1[:3]
    for layers in hidden_layer_choices
]
stage2 = run_stage("stage2_architecture", stage2_configs, n_splits=2)


# 3단계: 활성화 함수, bottleneck 출력 형태, LayerNorm을 비교합니다.
stage3_configs = sampled_variants(
    stage2[:5],
    choices={
        "hidden_activation": ["relu", "leaky_relu", "tanh", "silu"],
        "bottleneck_activation": ["linear", "relu", "tanh"],
        "layer_norm": [False, True],
    },
    sample_count=30,
    seed=SEED,
)
stage3 = run_stage("stage3_activation", stage3_configs, n_splits=2)


# 4단계: 조합 수가 너무 커지므로 정규화 조합 40개를 고르게 추출합니다.
stage4_configs = sampled_variants(
    stage3[:8],
    choices={
        "dropout": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
        "weight_decay": [
            0.0,
            1e-6,
            3e-6,
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
        ],
    },
    sample_count=40,
    seed=SEED,
)
stage4 = run_stage("stage4_regularization", stage4_configs, n_splits=2)


# 5단계: 좋은 정규화 설정 상위 8개에서 learning rate를 세밀하게 봅니다.
stage5_configs = sampled_variants(
    stage4[:8],
    choices={
        "learning_rate": [
            1e-4,
            2e-4,
            3e-4,
            5e-4,
            1e-3,
            2e-3,
            3e-3,
            5e-3,
            1e-2,
        ],
    },
    sample_count=30,
    seed=SEED,
)
stage5 = run_stage("stage5_learning_rate", stage5_configs, n_splits=2)


# 6단계: 유망한 후보를 더 긴 학습 예산과 3-fold에서 비교합니다.
stage6_configs = sampled_variants(
    stage5[:5],
    choices={
        "patience": [50, 100, 200],
        "min_delta": [0.0, 1e-7, 1e-6],
        "max_epochs": [3000],
    },
    sample_count=30,
    seed=SEED + 1,
)
# 이전 단계 설정도 포함되므로 모든 후보의 최대 epoch를 3000으로 맞춥니다.
stage6_configs = [changed(config, max_epochs=3000) for config in stage6_configs]
stage6 = run_stage("stage6_early_stopping", stage6_configs, n_splits=3)


# 7단계: 최종 후보 3개를 긴 학습, 5-fold, 5개 seed에서 확인합니다.
stage7_configs = [
    changed(result["config"], max_epochs=5000)
    for result in stage6[:3]
]
stage7 = run_stage(
    "stage7_final",
    stage7_configs,
    n_splits=5,
    seeds=(7, 21, 42, 100, 2026),
)


best_config = stage7[0]["config"]
with open(RESULT_DIR / "best_dnn_config.json", "w", encoding="utf-8") as file:
    json.dump(best_config, file, ensure_ascii=False, indent=2)

print("\n실험 완료")
print("결과 폴더:", RESULT_DIR)
print("최고 설정:", json.dumps(best_config, ensure_ascii=False, indent=2))
print(f"최종 평균 Ridge RMSE: {stage7[0]['mean_ridge_rmse']:.6f}")
print(f"최종 Ridge RMSE 표준편차: {stage7[0]['std_ridge_rmse']:.6f}")
