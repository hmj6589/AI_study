import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 클래스 이름 정의
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
X_train, X_test = X_train / 255, X_test / 255

# 레이블을 one-hot encoding으로 변환
# 현재 CIFAR-10 레이블은 정수 형태로 되어 있음
# one-hot encoding으로 변환하여 모델의 출력 형식과 일치시키기
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# MLP 모델 정의
# Sequential이란 : 레이어를 순차적으로 쌓아올리는 구조
model = keras.Sequential([
    # Flatten 레이어 : 3차원 배열을 1차원 배열로 평탄화함
    keras.layers.Flatten(input_shape=(32, 32, 3)),

    # 히든 레이어 1(1024)
    keras.layers.Dense(1024, activation='relu'),

    # 히든 레이어 2(512)
    keras.layers.Dense(512, activation='relu'),

    # 히든 레이어 3(512)
    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dense(10, activation='softmax')    # softmax 활성화 함수 -> 출력값을 확률로 변환
])
model.summary()

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
# epochs=10 -> 데이터 10번 반복 학습
# batch_size=128 -> 한 번에 업데이트에 사용할 샘플 개수
mlp_model = model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=2)

# 모델 평가
# evaluate: 테스트 데이터를 사용하여 손실값과 정확도를 평가
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nMLP 테스트 정확도:', test_acc)

# 테스트 데이터에 대해 예측 수행
# predict: 테스트 데이터를 입력으로 하여 모델의 예측값을 생성
# np.argmax: one-hot Encoding된 예측값과 실제 레이블을 다시 정수형으로 변환
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# 혼동 행렬 계산
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 정확도와 손실 그래프
plt.plot(mlp_model.history['loss'], 'b-', label='loss value')
plt.legend()
plt.plot(mlp_model.history['accuracy'], 'r-', label='accuracy')
plt.legend()
plt.show()