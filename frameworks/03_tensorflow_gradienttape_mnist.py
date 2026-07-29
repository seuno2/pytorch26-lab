# -*- coding: utf-8 -*-
"""
[실습 3] TensorFlow 저수준 (GradientTape) — MNIST 손글씨 분류
==============================================================
동일 과제 3종 비교 실습 중 3번: 미분 과정까지 노출하는 방식
모델 구조: 784 → 128(ReLU) → 10  (※ from_logits=True로 softmax 대체)

핵심 관찰 포인트
  - PyTorch와 루프 구조는 같지만, 기울기를 tape.gradient()로
    "명시적으로 꺼내서" 옵티마이저에 전달한다.
  - 기록(Tape) → 미분(gradient) → 적용(apply_gradients) 3단계 분리
    → 미분 과정 자체를 가르치기에는 오히려 가장 노골적인 방식
  - 실행: pip install tensorflow  →  python 03_tensorflow_gradienttape_mnist.py
"""

import tensorflow as tf

# 1. 데이터 준비 ------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train / 255.0).astype('float32')
x_test  = (x_test  / 255.0).astype('float32')

dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
           .shuffle(60000).batch(32))

# 2. 모델 정의 (구조 선언은 Keras 층을 재사용) -------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)                       # logits 출력
])
loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# 3. 학습 루프 — GradientTape로 미분 과정을 직접 제어 ------------------------
for epoch in range(5):
    for x, y in dataset:
        with tf.GradientTape() as tape:              # ① 연산 기록 시작
            pred = model(x, training=True)           # ② 순전파 (기록됨)
            loss = loss_fn(y, pred)                  # ③ 손실 계산 (기록됨)
        grads = tape.gradient(loss, model.trainable_variables)   # ④ 미분 실행
        optimizer.apply_gradients(zip(grads, model.trainable_variables))  # ⑤ 적용
    print(f"Epoch {epoch+1}/5  loss: {float(loss):.4f}")

# 4. 평가 -------------------------------------------------------------------
pred = model(x_test, training=False)
acc = tf.reduce_mean(
    tf.cast(tf.argmax(pred, axis=1) == y_test, tf.float32))
print(f"\n[TF-GradientTape] 테스트 정확도: {float(acc):.4f}")  # 약 0.977 예상
