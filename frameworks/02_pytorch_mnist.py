# -*- coding: utf-8 -*-
"""
[실습 2] PyTorch — MNIST 손글씨 분류
=====================================
동일 과제 3종 비교 실습 중 2번: 저수준 프레임워크 (명령형)
모델 구조: 784 → 128(ReLU) → 10  (※ softmax는 손실함수에 내장)

핵심 관찰 포인트
  - 학습 루프 5단계가 코드에 그대로 드러난다:
    ① zero_grad → ② forward → ③ loss → ④ backward → ⑤ step
  - 이 5줄이 딥러닝 학습의 실체다. PyTorch는 이것을 숨기지 않는다.
  - 루프 안에 print()를 넣으면 그대로 디버깅이 된다.
  - 실행: pip install torch torchvision  →  python 02_pytorch_mnist.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 0. 디바이스 선택 (CUDA / Apple Silicon MPS / CPU 자동) --------------------
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"사용 디바이스: {device}")

# 1. 데이터 준비 ------------------------------------------------------------
transform = transforms.ToTensor()                   # 0~1 정규화 포함
train_ds = datasets.MNIST('.', train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST('.', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=256)

# 2. 모델 정의 --------------------------------------------------------------
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10)          # softmax 없음 → CrossEntropyLoss에 포함
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 3. 학습 루프 — 직접 작성 (여기가 Keras의 fit() 한 줄에 해당) ---------------
for epoch in range(5):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()        # ① 이전 배치의 기울기 초기화
        pred = model(x)              # ② 순전파
        loss = criterion(pred, y)    # ③ 손실 계산
        loss.backward()              # ④ 역전파 (기울기 자동 계산)
        optimizer.step()             # ⑤ 가중치 갱신
    print(f"Epoch {epoch+1}/5  loss: {loss.item():.4f}")

# 4. 평가 -------------------------------------------------------------------
model.eval()
correct = 0
with torch.no_grad():                # 평가 시 기울기 계산 끄기
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()

print(f"\n[PyTorch] 테스트 정확도: {correct / len(test_ds):.4f}")  # 약 0.977 예상
