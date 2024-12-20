import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# CIFAR-10 클래스 이름 정의
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# CIFAR-10 데이터셋 로드
# 학습용 데이터(X_train, y_train), 테스트 데이터(X_test, y_test)로 구분
cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 데이터 정규화
# 각 픽셀 값을 0~1 사이의 값이 나오도록 255로 나누기
X_train, X_test = X_train / 255.0, X_test / 255.0

# 레이블 One-Hot Encoding -> 나중에 argmax로 정수 레이블 복구할거임
# 왜냐면, One-Hot Encoding은 CNN 학습에만 필요 (KNN에는 정수 레이블 필요함)
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

# CNN 모델 정의
# Sequential이란 : 레이어를 순차적으로 쌓아올리는 구조
cnn_model = models.Sequential([
    # Conv2D : 커널 크기 (3, 3)을 사용
    # ReLU 활성화 함수로 비선형성을 추가
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),    # 공간 크기를 절반으로 줄임

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

    # Flatten : CNN에서 추출된 2D 특징 맵을 1D 벡터로 변환
    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    # Softmax 활성화 함수 -> 각 클래스의 확률 반환
    layers.Dense(10, activation='softmax')
])

# CNN 모델 컴파일
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# CNN 모델 학습
# epochs=10 -> 데이터 10번 반복 학습
# batch_size=128 -> 한 번에 업데이트에 사용할 샘플 개수
history = cnn_model.fit(X_train, y_train_onehot, epochs=10, batch_size=128, verbose=2)

# CNN 부분 정확도와 손실 그래프
plt.plot(history.history['loss'], 'b-', label='loss value')
plt.legend()
plt.plot(history.history['accuracy'], 'r-', label='accuracy')
plt.legend()
plt.show()

# CNN을 사용하여 입력 데이터 -> 1D 특징 벡터로 변환
# 해당 벡터는 KNN에 입력으로 사용됨
X_train_features = cnn_model.predict(X_train, verbose=2)
X_test_features = cnn_model.predict(X_test, verbose=2)

# KNN 모델 정의
# n_neighbors=5: 가장 가까운 5개의 이웃을 사용하여 클래스를 예측
# metric='euclidean': 유클리드 거리를 사용하여 이웃 계산
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# KNN 모델 학습
# CNN에서의 one-hot encoding 레이블을 정수 레이블로 변환
y_train_labels = np.argmax(y_train_onehot, axis=1)
# CNN에서 추출된 특징 벡터 -> KNN 학습
knn_model.fit(X_train_features, y_train_labels)

# 테스트 데이터 예측
# 학습된 KNN 모델을 사용하여 테스트 데이터의 레이블을 예측
predicted_labels = knn_model.predict(X_test_features)

# 정확도 평가
# CNN에서의 one-hot encoding 레이블을 정수 레이블로 변환
true_labels = np.argmax(y_test_onehot, axis=1)
# accuracy_score : 테스트 데이터를 얼마나 정확하게 예측했는지 평가
knn_accuracy = accuracy_score(true_labels, predicted_labels)
print("\nCNN+KNN 테스트 정확도:", knn_accuracy)

# 혼동 행렬 계산
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xticks(rotation=45)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('KNN Confusion Matrix')
plt.show()
