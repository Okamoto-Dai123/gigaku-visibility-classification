# ============================================================
# GIGAKU - Visibility Classification (降雪時視程分類)
# Docker環境: TensorFlow + scikit-learn + Jupyter
# 担当: 岡本 (引き継ぎ元: 野崎)
# ============================================================
FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

LABEL maintainer="ikeda-lab"
LABEL description="GIGAKU visibility classification - ML/DL pipeline"

# システムパッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /workspace

# Python依存パッケージ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードをコピー
COPY src/ ./src/
COPY notebooks/ ./notebooks/

# Jupyter設定
EXPOSE 8888

# デフォルトコマンド: Jupyter Lab起動
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
