import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.applications import Xception, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("ライブラリのインポートが完了しました。")

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATASET_PATH = "/content/drive/MyDrive/ImageDataSet/"
    print(f"Google Driveのマウントに成功しました。")
    print(f"データセットパス: {DATASET_PATH}")
except Exception as e:
    print(f"Google Driveのマウント中にエラーが発生しました: {e}")

# 画像のパスとラベルをDataFrameに格納
filepaths = []
labels = []
class_names = ["lvl1", "lvl2", "lvl3"]
for i, name in enumerate(class_names):
    class_dir = os.path.join(DATASET_PATH, name)
    image_paths = glob.glob(os.path.join(class_dir, '*.jpg')) + \
                  glob.glob(os.path.join(class_dir, '*.png')) + \
                  glob.glob(os.path.join(class_dir, '*.JPG'))
    filepaths.extend(image_paths)
    labels.extend([name] * len(image_paths))

df = pd.DataFrame({'filepath': filepaths, 'label': labels})
df = df.sample(frac=1).reset_index(drop=True)
train_df, test_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=42, stratify=df['label'])
print("\n[データ準備完了]")
print(f"全画像数: {len(df)}")
print(f"学習用データ数: {len(train_df)}")
print(f"テスト用データ数: {len(test_df)}")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

print("\nセットアップとデータ準備が完了しました。")

INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 3

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    # ショートカット接続のチャンネル数を調整
    if stride > 1 or x.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def create_resnet18_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs, name="ResNet18")
    return model

def create_xception_model(input_shape, num_classes):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False # ベースモデルの重みは凍結
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs, name="Xception")
    return model

def create_densenet_model(input_shape, num_classes):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs, name="DenseNet121")
    return model

def create_alexnet_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x); x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x); x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs, name="AlexNet")
    return model

histories = {}
final_results = {}

def train_and_evaluate(model_name, create_model_func):
    print("\n" + "="*80)
    print(f"{model_name} の学習と評価を開始します...")
    print("="*80)

    # モデルの作成とコンパイル
    model = create_model_func(INPUT_SHAPE, NUM_CLASSES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # データ生成器の作成
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df, x_col='filepath', y_col='label',
        target_size=INPUT_SHAPE[:2], batch_size=32, class_mode='categorical', shuffle=True
    )
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df, x_col='filepath', y_col='label',
        target_size=INPUT_SHAPE[:2], batch_size=32, class_mode='categorical', shuffle=False
    )

    # 学習
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=test_generator,
        verbose=1
    )
    histories[model_name] = history

    print(f"\n{model_name} の最終性能評価...")
    loss, accuracy = model.evaluate(test_generator, verbose=0)

    y_pred_proba = model.predict(test_generator)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_generator.classes

    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)

    final_results[model_name] = {'accuracy': accuracy, 'loss': loss, 'report': report, 'cm': cm}

    print(f"\n {model_name} 完了！")
    print(f"テスト精度: {accuracy:.4f}")
    print("\n分類レポート:")
    print(report)

print(" モデル定義と共通関数の準備が完了しました。")

train_and_evaluate('ResNet18', create_resnet18_model)

train_and_evaluate('Xception', create_xception_model)

train_and_evaluate('DenseNet121', create_densenet_model)

train_and_evaluate('AlexNet', create_alexnet_model)

if not final_results:
    print("結果データがありません。先にモデルの学習セルを実行してください。")
else:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    for model_name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{model_name} Train')
        plt.plot(history.history['val_accuracy'], label=f'{model_name} Val', linestyle='--')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.legend()

    plt.subplot(1, 2, 2)
    for model_name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{model_name} Train')
        plt.plot(history.history['val_loss'], label=f'{model_name} Val', linestyle='--')
    plt.title('Model Loss Comparison')
    plt.ylabel('Loss'); plt.xlabel('Epoch'); plt.legend()
    plt.show()

    results_summary = {name: res['accuracy'] for name, res in final_results.items()}
    results_df = pd.DataFrame.from_dict(results_summary, orient='index', columns=['Test Accuracy'])
    results_df = results_df.sort_values(by='Test Accuracy', ascending=False)

    print("\n" + "="*50)
    print("最終比較")
    print("="*50)
    print(results_df)

    plt.figure(figsize=(20, 5))
    for i, (model_name, results) in enumerate(final_results.items()):
        plt.subplot(1, len(final_results), i + 1)
        sns.heatmap(results['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()