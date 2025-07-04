# SimCLR 細胞圖像自監督學習專案

基於 SimCLR 的自監督學習系統，用於細胞圖像特徵學習。

## 功能特點

- **自監督預訓練**
  - 使用 SimCLR 框架進行特徵學習
  - 自動混合精度訓練 (AMP)
  - 支持多種數據增強方式
  - 可配置的學習率調度
  - 每 10 個 epoch 自動保存檢查點
  - 自動保存最佳模型

- **下游任務**
  - 支持 K-fold 交叉驗證
  - 兩階段訓練策略
  - 可配置的數據增強
  - 詳細的訓練指標記錄
  - 自動生成混淆矩陣和訓練曲線

## 安裝要求

使用 pip 安裝所需的依賴：

```bash
pip install -r requirements.txt
```

## 目錄結構

```
SimCLR/
├── config.yaml                 # 配置文件
├── requirements.txt           # 依賴包列表
├── downstream/               # 下游任務數據集
├── pretrain/                # 預訓練數據集
├── experiments_results_[timestamp]/  # 實驗結果目錄
│   ├── batch_8/            # batch_size=8 的實驗結果
│   │   ├── best_model/     # 最佳模型檢查點
│   │   │   └── best_model.ckpt
│   │   └── downstream_results/  # 該 batch 的下游任務結果
│   │       ├── logs/       # 訓練日誌
│   │       ├── plots/      # 訓練曲線和混淆矩陣
│   │       └── training_results.txt  # 訓練結果摘要
│   ├── batch_16/           # batch_size=16 的實驗結果
│   │   ├── best_model/
│   │   └── downstream_results/
│   └── ...                 # 其他 batch 大小的實驗結果
└── src/                    # 源代碼目錄
```

## 使用方法

### 1. 數據準備

首先運行 split.py 將數據集分割為預訓練集和下游任務集：

```bash
python split.py
```

### 2. 預訓練階段

運行 SimCLR 預訓練，支持指定批次大小：

```bash
# 使用單一批次大小
python simclr_schedule.py --batch_size 32

# 使用多個批次大小依序訓練
python simclr_schedule.py --batch_size 16 32 64
```

### 3. 下游任務訓練

運行下游任務訓練，支持 K-fold 交叉驗證和數據增強：

```bash
# 基本訓練
python downstream_integrated.py

# 使用 5-fold 交叉驗證和 3 倍數據增強
python downstream_integrated.py --k-fold 5 --augment 3
```

## 配置說明

在 `config.yaml` 中可以配置以下參數：

```yaml
base:
  seed: 42              # 隨機種子

data:
  batch_size: 88        # 批次大小
  num_workers: 4        # 數據加載線程數
  valid_size: 0.2       # 驗證集比例
  input_shape: [224, 224]  # 輸入圖像尺寸
  strength: 0.5         # 數據增強強度

training:
  epochs: 200             # 訓練輪數
  learning_rate: 0.001  # 學習率
  weight_decay: 1e-4    # 權重衰減
  temperature: 0.4     # 對比學習溫度係數
  early_stopping_patience: 10  # 早停耐心值
```
