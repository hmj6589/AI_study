import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
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
    X = np.array(X)     # 이미지 배열
    y = np.array(y)     # 레이블
    return X, y, class_names       # class_names : 클래스 이름 리스트

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

# MLP 모델 정의
mlp_model = Sequential([
    Flatten(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),  # 다차원 이미지를 1차원 벡터로 변환

    Dense(1024, activation='relu'),  # 히든 레이어 1

    Dense(512, activation='relu'),  # 히든 레이어 2

    Dense(512, activation='relu'),  # 히든 레이어 3

    Dense(len(class_names), activation='softmax')  # 출력 레이어
])

# MLP 모델 컴파일
mlp_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# MLP 모델 학습
# epochs=10 -> 데이터 10번 반복 학습
# batch_size=128 -> 한 번에 업데이트에 사용할 샘플 개수
history = mlp_model.fit(X_train, y_train_onehot, epochs=10, batch_size=128, verbose=2)

# 테스트 데이터 예측
y_pred_onehot = mlp_model.predict(X_test)
y_pred = np.argmax(y_pred_onehot, axis=1)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("MLP 모델 테스트 정확도:", accuracy)

# 혼동 행렬 계산
conf_matrix = confusion_matrix(y_test, y_pred)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('MLP model Confusion Matrix')
plt.show()

# MLP 학습 정확도와 손실 그래프
plt.figure(figsize=(12, 5))

# 정확도와 손실 그래프
plt.title('MLP model')
plt.plot(history.history['loss'], 'b-', label='loss value')
plt.legend()
plt.plot(history.history['accuracy'], 'r-', label='accuracy')
plt.legend()
plt.show()