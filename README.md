# GIGAKU - 降雪時視程分類 (Visibility Classification in Snowfall)

## 概要

降雪による視程劣化環境（DVE: Degraded Visual Environment）において、CCTVカメラ画像から道路の視程を3段階に自動分類するシステム。

**STI-Gigaku 2024 発表論文の再現実験リポジトリ**

### 視程階級の定義
| 階級 | 視認距離 | 意味 |
|------|---------|------|
| Level 1 | 500m以上 | 良好 |
| Level 2 | 250m〜500m | 注意 |
| Level 3 | 250m未満 | 危険 |

## 研究内容

2つのアプローチを比較分析:

1. **特徴量ベース機械学習**: 10種の画像特徴量 x 4種の分類器 = 4,092通りの組み合わせをラッパー法（逐次前進選択法）で探索
2. **ディープラーニング**: ResNet18, Xception, DenseNet121, EfficientNetB0 の4モデルをファインチューニング

### 使用する特徴量
- 光学的特徴: RMS Contrast, Edge Density, Dark Channel
- テクスチャ・形状: LBP, HOG
- 周波数: Spatial Frequency (SF), SF + mRMR
- 深層特徴: InceptionV3, ResNet50, EfficientNetB0

### 使用する分類器
- Random Forest, SVM, LightGBM, MLP

## ディレクトリ構成

```
gigaku-visibility-classification/
├── Dockerfile              # Docker環境定義
├── docker-compose.yml      # Docker Compose設定
├── requirements.txt        # Python依存パッケージ
├── .gitignore
├── README.md               # このファイル
├── src/
│   ├── DL.py              # ディープラーニングパイプライン
│   ├── AnoV3.py           # 画像アノテーションツール
│   └── feature_extraction/ # 特徴量抽出スクリプト（今後追加）
├── data/                   # 画像データ（Git管理外）
│   ├── lvl1/              # 階級1画像
│   ├── lvl2/              # 階級2画像
│   └── lvl3/              # 階級3画像
├── notebooks/              # 実験用Jupyterノートブック
├── models/                 # 学習済みモデル（Git管理外）
└── docs/                   # 論文・ポスター等の参考資料
```

## セットアップ手順

### 前提条件
- Docker Desktop がインストール済み
- Git がインストール済み

### 1. リポジトリのクローン
```bash
git clone https://github.com/<your-org>/gigaku-visibility-classification.git
cd gigaku-visibility-classification
```

### 2. データの配置
画像データはGit管理外のため、別途共有されたデータを `data/` フォルダに配置:
```bash
# 共有された画像データを解凍して配置
unzip ImageDataSetLoc1.zip -d data/
```

### 3. Docker環境の起動
```bash
docker compose up --build
```

### 4. Jupyter Labにアクセス
ブラウザで http://localhost:8888 を開く

### 5. 実験の実行
- `src/DL.py` を Jupyter上で実行、またはターミナルから:
```bash
docker compose exec gigaku-ml python src/DL.py
```

## 引き継ぎ元
- 研究者: 野崎 大夢
- 指導教員: 池田 啓
- 発表: STI-Gigaku 2024

## 引き継ぎ先
- 担当: 岡本
