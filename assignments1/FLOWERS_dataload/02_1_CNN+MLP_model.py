import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 이미지 크기 설정
IMAGE_SIZE = (128, 128)

# 데이터 로드 함수
def load_train_data(folder_path):
    X = []
    y = []
    class_names = os.listdir(folder_path)
    print("클래스 이름:", class_names)

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(folder_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = load_img(image_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            X.append(image)
            y.append(i)
    X = np.array(X)  # 이미지 배열
    y = np.array(y)  # 레이블
    return X, y, class_names  # class_names : 클래스 이름 리스트


# Flowers 데이터셋 경로 설정
train_folder = './flowers-dataset/train'

# 데이터 로드
X, y, class_names = load_train_data(train_folder)

# train-test 분할(학습용 / 테스트용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화 - 각 픽셀 값을 0~1 사이로 정규화
X_train = X_train / 255.0
X_test = X_test / 255.0

# 레이블 One-Hot Encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, len(class_names))
y_test_onehot = tf.keras.utils.to_categorical(y_test, len(class_names))

# CNN 모델 정의
# Sequential이란 : 레이어를 순차적으로 쌓아올리는 구조
cnn_model = models.Sequential([
    # Conv2D : 커널 크기 (3, 3)을 사용
    # ReLU 활성화 함수로 비선형성을 추가
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),  # 공간 크기를 절반으로 줄임

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

    layers.Flatten(),  # Flatten : CNN에서 추출된 2D 특징 맵을 1D 벡터로 변환

    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  # Softmax 활성화 함수 -> 각 클래스의 확률 반환
])

# CNN 모델 컴파일
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# CNN 모델 학습
# epochs=10 -> 데이터 10번 반복 학습
# batch_size=128 -> 한 번에 업데이트에 사용할 샘플 개수
cnn_history = cnn_model.fit(X_train, y_train_onehot, epochs=10, batch_size=128, verbose=2)

# CNN 부분 정확도와 손실 그래프
plt.figure(figsize=(12, 5))
plt.title('CNN+MLP - CNN model')
plt.plot(cnn_history.history['loss'], 'b-', label='Loss')
plt.plot(cnn_history.history['accuracy'], 'r-', label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()

# CNN을 사용하여 특징 추출
X_train_features = cnn_model.predict(X_train, verbose=2)
X_test_features = cnn_model.predict(X_test, verbose=2)

# MLP 모델 정의
mlp_model = models.Sequential([
    # 특징 벡터를 입력 받아서 학습하기
    layers.Dense(256, activation='relu', input_shape=(X_train_features.shape[1],)),

    # 비선형 변환을 통해 데이터 패턴 학습하기
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),

    layers.Dense(len(class_names), activation='softmax')    # 최종적으로 각 클래스 확률 출력
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

# CNN+MLP 학습 정확도와 손실 그래프
plt.figure(figsize=(12, 5))
plt.title('CNN+MLP model')
plt.plot(mlp_history.history['loss'], 'b-', label='Loss')
plt.plot(mlp_history.history['accuracy'], 'r-', label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
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
