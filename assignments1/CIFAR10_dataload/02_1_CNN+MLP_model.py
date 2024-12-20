import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CIFAR-10 데이터셋 로드
# 학습용 데이터(X_train, y_train), 테스트 데이터(X_test, y_test)로 구분
cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# CIFAR-10 클래스 이름 정의
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# 데이터 정규화
# 각 픽셀 값을 0~1 사이의 값이 나오도록 255로 나누기
X_train, X_test = X_train / 255.0, X_test / 255.0

# 레이블 One-Hot Encoding -> MLP 학습에 필요
y_train_onehot = to_categorical(y_train, 10)
y_test_onehot = to_categorical(y_test, 10)

# CNN 모델 정의
# Sequential이란 : 레이어를 순차적으로 쌓아올리는 구조
cnn_model = models.Sequential([
    # Conv2D : 커널 크기 (3, 3)을 사용
    # ReLU 활성화 함수로 비선형성을 추가
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),     # 공간 크기를 절반으로 줄임

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

    layers.Flatten(),   # Flatten: 다차원 데이터를 1차원 벡터로 변환

    layers.Dense(128, activation='relu'),   # Fully Connected Layer로 특징을 학습
    layers.Dense(10, activation='softmax')  # 출력층 - 10개의 클래스 확률 출력
])

# CNN 모델 컴파일
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# CNN 모델 학습
# epochs=10 -> 데이터 10번 반복 학습
# batch_size=128 -> 한 번에 업데이트에 사용할 샘플 개수
history = cnn_model.fit(X_train, y_train_onehot, epochs=10, batch_size=128, verbose=2)


# 정확도와 손실 그래프
plt.title('CNN model')
plt.plot(history.history['loss'], 'b-', label='loss value')
plt.legend()
plt.plot(history.history['accuracy'], 'r-', label='accuracy')
plt.legend()
plt.show()

# CNN을 사용하여 특징 추출
# 학습된 CNN 모델로 이미지의 특징 벡터를 추출
X_train_features = cnn_model.predict(X_train, verbose=2)
X_test_features = cnn_model.predict(X_test, verbose=2)

# MLP 모델 정의
mlp_model = models.Sequential([
    # 특징 벡터를 입력 받아서 학습하기
    layers.Dense(256, activation='relu', input_shape=(X_train_features.shape[1],)),

    # 비선형 변환을 통해 데이터 패턴 학습하기
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),

    layers.Dense(10, activation='softmax')  # 최종적으로 각 클래스 확률 출력
])

# MLP 모델 컴파일
mlp_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# MLP 학습
# epochs=10 -> 데이터 10번 반복 학습
# batch_size=128 -> 한 번에 업데이트에 사용할 샘플 개수
# Validation Split -> 학습 데이터의 20%를 검증 데이터로 사용하겠다는 것
mlp_history = mlp_model.fit(X_train_features, y_train_onehot, epochs=10, batch_size=128, validation_split=0.2, verbose=2)

# MLP 학습 정확도와 손실 그래프
plt.figure(figsize=(12, 5))
plt.title('CNN+MLP model')
plt.plot(mlp_history.history['loss'], 'b-', label='loss value')
plt.legend()
plt.plot(mlp_history.history['accuracy'], 'r-', label='accuracy')
plt.legend()
plt.show()

# 테스트 데이터 예측
mlp_predictions = mlp_model.predict(X_test_features)
predicted_labels = np.argmax(mlp_predictions, axis=1)

# 정확도 평가
true_labels = y_test.flatten()
mlp_accuracy = accuracy_score(true_labels, predicted_labels)
print("\nCNN+MLP 테스트 정확도:", mlp_accuracy)

# 혼동 행렬 계산
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xticks(rotation=45)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('CNN+MLP Confusion Matrix')
plt.show()
