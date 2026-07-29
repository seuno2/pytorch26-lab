# -*- coding: utf-8 -*-
"""
[실습 1] Keras — MNIST 손글씨 분류
===================================
동일 과제 3종 비교 실습 중 1번: 고수준 API (선언형)
모델 구조: 784 → 128(ReLU) → 10(Softmax)

핵심 관찰 포인트
  - 학습 루프가 코드에 보이지 않는다 → fit() 내부에서 자동 수행
  - "무엇을 만들지"만 선언하면 "어떻게 학습할지"는 프레임워크가 처리
  - 실행: pip install keras tensorflow  →  python 01_keras_mnist.py
"""

import keras
from keras import layers

# 1. 데이터 준비 -----------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0   # 0~1 정규화

# 2. 모델 정의 (선언형: 층을 쌓기만 한다) -----------------------------------
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),          # 28x28 → 784
    layers.Dense(128, activation='relu'),          # 은닉층
    layers.Dense(10, activation='softmax')         # 출력층 (확률)
])

# 3. 컴파일: 옵티마이저 / 손실함수 / 지표 지정 ------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 학습 — 순전파·역전파·가중치 갱신이 이 한 줄 안에서 전부 일어난다 --------
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 5. 평가 -------------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n[Keras] 테스트 정확도: {test_acc:.4f}")   # 약 0.977 예상
