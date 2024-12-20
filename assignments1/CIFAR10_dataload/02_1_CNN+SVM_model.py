import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
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

# 레이블 One-Hot Encoding -> 나중에 argmax로 정수 레이블 복구할거임
# 왜냐면, One-Hot Encoding은 CNN 학습에만 필요 (SVM에는 정수 레이블 필요)
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

# CNN 모델 정의
# Sequential이란 : 레이어를 순차적으로 쌓아올리는 구조
cnn_model = models.Sequential([
    # Conv2D : 커널 크기 (3, 3)을 사용
    # ReLU 활성화 함수로 비선형성을 추가
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),    # # 공간 크기를 절반으로 줄임

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),   # Fully Connected 레이어로 최종 특징 학습

    # Softmax 활성화 함수 -> 각 클래스의 확률 반환
    layers.Dense(10, activation='softmax')
])

# CNN 모델 컴파일
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# 모델 학습
# epochs=10 -> 데이터 10번 반복 학습
# batch_size=128 -> 한 번에 업데이트에 사용할 샘플 개수
history = cnn_model.fit(X_train, y_train_onehot, epochs=10, batch_size=128, verbose=2)

# CNN 부분의 정확도와 손실 그래프
plt.plot(history.history['loss'], 'b-', label='loss value')
plt.legend()
plt.plot(history.history['accuracy'], 'r-', label='accuracy')
plt.legend()
plt.show()

# 학습된 CNN 모델로 이미지 데이터를 입력받아 고차원 특징 벡터를 생성
X_train_features = cnn_model.predict(X_train, verbose=2)
X_test_features = cnn_model.predict(X_test, verbose=2)

# SVM 모델 정의
# 커널 : linear (선형 분류 경계)
# C=1 : 규제 강도, 값이 작을수록 규제가 강함
# 다중 클래스 분류 방식 : ovr (one-vs-rest)
svm_model = SVC(kernel='linear', C=1, decision_function_shape='ovr')

# SVM 모델 학습
y_train_labels = np.argmax(y_train_onehot, axis=1)  # One-Hot Encoding을 정수 레이블로 변환
svm_model.fit(X_train_features, y_train_labels)     # CNN이 추출한 특징(X_train_features)을 입력으로 사용

# 테스트 데이터 예측
predicted_labels = svm_model.predict(X_test_features)

# 정확도 평가
true_labels = np.argmax(y_test_onehot, axis=1)
svm_accuracy = accuracy_score(true_labels, predicted_labels)
print("\nSVM 테스트 정확도:", svm_accuracy)

# 혼동 행렬 계산
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xticks(rotation=45)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('SVM Confusion Matrix')
plt.show()
