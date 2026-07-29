# ============================================================
# 2일차 7교시 — A2C 구현
# 복사해서 그대로 실행하면 됩니다. 고칠 것 없습니다.
# ------------------------------------------------------------
# 이 교시 코드는 앞 교시의 변수·클래스를 이어 씁니다.
# 그래서 이 블록에는 **여기까지 필요한 코드가 전부** 들어 있습니다.
# (수업용 코드만 따로 복사하면 NameError 가 납니다 — 그건 정상입니다.)
# ============================================================

# ── 1교시에서 이어받음 — DQN 소개 ──
import random                            # 여러 개 중에서 아무거나 뽑을 때 쓰는 도구
from collections import deque             # 앞뒤로 넣고 빼기 쉬운 '줄서기 상자'
import numpy as np                        # 숫자 계산 도구 (파이썬의 계산기)
import torch                              # 파이토치 — 신경망을 다루는 도구


class ReplayBuffer:
    """
    경험 재현 버퍼 — 게임하면서 겪은 일을 적어 두는 '일기장'입니다.

    왜 필요할까요?
      방금 겪은 일만 보고 배우면 비슷한 것만 연달아 보게 됩니다.
      (왼쪽으로 갔다 -> 또 왼쪽 -> 또 왼쪽 ...)
      그러면 신경망이 방금 본 것에만 맞추고 예전 것을 잊어버립니다.
      그래서 일기장에 잔뜩 적어 두고, 배울 때는 여기저기서 섞어서 꺼냅니다.

    이 상자는 3일 내내 씁니다 — 오늘 DQN, 내일 DDPG와 SAC.
    """

    def __init__(self, capacity=100_000, action_dtype=torch.int64):
        # capacity = 일기장에 몇 줄까지 적어 둘지 (10만 줄)
        #            넘치면 가장 오래된 것부터 자동으로 지워집니다.
        #
        # action_dtype = 행동을 어떤 숫자로 적을지
        #   오늘 DQN  : 행동이 "0번? 1번?" 이라서 정수(int64)
        #   내일 DDPG : 행동이 "힘을 1.37만큼" 이라서 실수(float32)
        #   -> 내일은 ReplayBuffer(100_000, action_dtype=torch.float32) 로 씁니다.
        #      정수로 두면 1.37 이 1 로 잘려서 학습이 통째로 망가집니다.
        self.buffer = deque(maxlen=capacity)    # 실제 일기장 (꽉 차면 앞쪽부터 밀려남)
        self.action_dtype = action_dtype        # 행동을 적을 숫자 종류를 기억해 둔다

    def push(self, s, a, r, s_next, done):
        # 일기 한 줄 적기:
        #   "이 상황(s)에서 이 행동(a)을 했더니 점수(r)를 받고
        #    저 상황(s_next)이 됐다. 그리고 판이 끝났나?(done)"
        self.buffer.append((s, a, r, s_next, done))   # 다섯 개를 한 묶음으로 저장

    def sample(self, batch_size):
        # 일기장에서 batch_size 줄을 무작위로 뽑아 온다 (= 섞어서 꺼내기)
        batch = random.sample(self.buffer, batch_size)   # 예: 아무 데서나 64줄

        # 뽑아온 것은 (상황, 행동, 점수, 다음상황, 끝났나) 묶음들의 목록입니다.
        # zip(*batch) 는 이걸 세로로 갈라 줍니다 —
        #   상황은 상황끼리, 행동은 행동끼리 따로 모아 줍니다.
        s, a, r, s_next, done = zip(*batch)

        # 파이썬 목록 -> 넘파이 배열 -> 파이토치 텐서 순서로 바꿉니다.
        # 왜 np.array 를 한 번 거칠까요?
        #   배열들의 '목록'을 텐서로 바로 만들면 파이토치가 하나씩 복사하느라
        #   아주 느려집니다. 넘파이로 먼저 한 덩어리를 만들면 훨씬 빠릅니다.
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32),       # 상황들
            torch.as_tensor(np.array(a), dtype=self.action_dtype),   # 행동들
            torch.as_tensor(np.array(r), dtype=torch.float32),       # 점수들
            torch.as_tensor(np.array(s_next), dtype=torch.float32),  # 다음 상황들
            torch.as_tensor(np.array(done), dtype=torch.float32),    # 끝났나 (1이면 끝)
        )

    def __len__(self):
        # len(buffer) 라고 쓰면 이 함수가 불립니다 — 지금 몇 줄 적혀 있는지 알려 줍니다
        return len(self.buffer)

# ============================================================
# 잘 만들어졌는지 확인 (이 부분이 있어야 실행했을 때 결과가 보입니다)
# ============================================================
print('일기장이 잘 만들어졌는지 확인합니다.')
print()

buf = ReplayBuffer(capacity=1000)              # 1000줄짜리 일기장 하나 만들기

for i in range(200):                            # 가짜 경험 200줄 적어 보기
    s = np.random.randn(4).astype(np.float32)   # 상황 (숫자 4개)
    a = np.random.randint(2)                    # 행동 (0 또는 1)
    r = 1.0                                     # 점수
    s2 = np.random.randn(4).astype(np.float32)  # 다음 상황
    buf.push(s, a, r, s2, 0.0)                  # 일기장에 한 줄 적기

print(f'  일기장에 적힌 줄 수 : {len(buf)}')
print()

s, a, r, s2, done = buf.sample(32)              # 32줄 무작위로 꺼내기
print('  32줄 꺼냈을 때 모양')
print(f'    상황 s      {tuple(s.shape)}   {s.dtype}')
print(f'    행동 a      {tuple(a.shape)}      {a.dtype}   <- 정수여야 gather 가 됩니다')
print(f'    점수 r      {tuple(r.shape)}      {r.dtype}')
print(f'    다음상황 s2 {tuple(s2.shape)}   {s2.dtype}')
print(f'    끝났나 done {tuple(done.shape)}      {done.dtype}')
print()
print('  -> 모양이 위와 같이 나오면 정상입니다. 4교시에서 이 상자를 그대로 씁니다.')

# ── 2교시에서 이어받음 — Double DQN 소개 ──
import torch                                   # 파이토치

# ============================================================
# DQN 과 Double DQN 은 '정답(목표값)을 만드는 방법'만 다릅니다.
# 아래 두 함수를 나란히 두고 보면 차이가 딱 한 곳뿐인 게 보입니다.
# ============================================================


@torch.no_grad()                               # 이 함수 안에서는 미분 준비를 하지 않는다
def dqn_target(q_target, r, s_next, done, gamma=0.99):
    """
    그냥 DQN 의 목표값.
    '고르기'와 '점수 매기기'를 둘 다 타깃 네트워크가 합니다.

    왜 no_grad 인가요?
      목표값은 '과녁'입니다. 과녁은 가만히 있어야 맞출 수 있습니다.
      미분이 여기까지 흘러오면 과녁이 같이 도망갑니다.
    """
    # q_target(s_next) = 다음 상황에서 각 행동이 얼마나 좋은지 (여러 개)
    # .max(dim=1)      = 그중 가장 큰 것을 고른다 (dim=1 은 '행동 방향으로')
    # .values          = 값만 꺼낸다 (몇 번째인지는 안 쓴다)
    max_q = q_target(s_next).max(dim=1).values

    # 목표 = 지금 받은 점수 + 감마 × 다음 상황의 값
    # (1 - done) 은 "판이 끝났으면 뒤는 없다"는 뜻입니다.
    #   done=1(끝) -> 0 을 곱해 다음 값을 지운다
    #   done=0(계속) -> 1 을 곱해 그대로 둔다
    return r + gamma * max_q * (1 - done)


@torch.no_grad()
def double_dqn_target(q_online, q_target, r, s_next, done, gamma=0.99):
    """
    Double DQN 의 목표값.
    '고르는 사람'과 '점수 매기는 사람'을 나눕니다.

    왜 나누나요?
      한 사람이 고르고 그 사람이 점수까지 매기면,
      운 좋게 튄 값을 고른 뒤 그 튄 값을 그대로 믿게 됩니다.
      -> Q값이 실제보다 부풀려집니다 (최대화 편향)
    """
    # ① 고르기는 온라인 네트워크(지금 학습 중인 쪽)가 한다
    #    argmax = "가장 큰 것이 몇 번째냐" (값이 아니라 번호)
    #    keepdim=True = 모양을 (배치, 1) 로 유지 (뒤에서 gather 에 쓰려고)
    best_a = q_online(s_next).argmax(dim=1, keepdim=True)

    # ② 점수 매기기는 타깃 네트워크(잠시 고정된 쪽)가 한다
    #    gather(1, best_a) = 각 줄에서 best_a 번째 값만 골라 뽑기
    #    squeeze(1)        = (배치, 1) -> (배치,) 로 눌러서 모양 맞추기
    max_q = q_target(s_next).gather(1, best_a).squeeze(1)

    # 목표를 만드는 마지막 줄은 위와 똑같습니다.
    return r + gamma * max_q * (1 - done)


# ============================================================
# 정리 — 딱 이 차이입니다
#   DQN        : 타깃넷이 고르고, 타깃넷이 점수 매김   (혼자 다 함)
#   Double DQN : 온라인넷이 고르고, 타깃넷이 점수 매김 (역할 분담)
# ============================================================

# ============================================================
# 잘 만들어졌는지 확인 (이 부분이 있어야 실행했을 때 결과가 보입니다)
# ============================================================
import torch.nn as nn
torch.manual_seed(0)

print('두 방식이 실제로 다른 값을 내는지 확인합니다.')
print()

# 아주 작은 가짜 신경망 두 개 (온라인용, 타깃용)
q_online = nn.Linear(4, 3)                      # 상황 4개 -> 행동 3개의 Q값
q_target = nn.Linear(4, 3)

s_next = torch.randn(5, 4)                      # 다음 상황 5개
r = torch.ones(5)                               # 점수는 전부 1점
done = torch.zeros(5)                           # 아직 안 끝남

y1 = dqn_target(q_target, r, s_next, done)                    # 그냥 DQN
y2 = double_dqn_target(q_online, q_target, r, s_next, done)   # Double DQN

print('  그냥 DQN   :', [round(v, 4) for v in y1.tolist()])
print('  Double DQN :', [round(v, 4) for v in y2.tolist()])
print('  차이       :', [round(v, 4) for v in (y1 - y2).tolist()])
print()
print(f'  평균 차이  : {(y1 - y2).mean().item():+.4f}')
print('  -> 그냥 DQN 쪽이 대체로 큽니다. 이게 부풀려진 만큼입니다.')
print('     (한 번만 보면 우연일 수 있습니다. 아래 [실험] 에서 2만 번 반복해 확인합니다)')

# ── 3교시에서 이어받음 — PyTorch 소개 및 구현 ──
import torch                                    # 파이토치 본체
import torch.nn as nn                            # 신경망 부품 상자 (층, 손실함수 등)

# 그래픽카드(GPU)가 있으면 쓰고, 없으면 CPU 를 쓴다.
# 오늘 코드는 아주 작아서 CPU 로도 충분합니다 — 없다고 걱정하지 마세요.
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. 텐서와 자동미분 — 파이토치가 미분을 대신 해준다
# ============================================================

# requires_grad=True = "이 값에 대해 미분할 거야" 라고 표시하는 것
x = torch.tensor([2.0], requires_grad=True)

y = x ** 2 + 3 * x          # y = x제곱 + 3x  (x=2 이므로 y = 4 + 6 = 10)

y.backward()                # 여기서 미분이 일어난다 -> x.grad 에 답이 들어감

print(x.grad)               # dy/dx = 2x + 3 = 2*2 + 3 = 7
                            # 손으로 계산한 값과 똑같습니다. 파이토치가 대신 해준 것뿐입니다.


# ============================================================
# 2. nn.Module 로 Q-네트워크 만들기
#    상황을 받아서 "각 행동이 얼마나 좋은지"를 내놓는 신경망
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        # state_dim  = 상황을 나타내는 숫자가 몇 개인지 (CartPole 은 4개)
        # action_dim = 할 수 있는 행동이 몇 가지인지 (CartPole 은 왼/오 2가지)
        # hidden     = 가운데 층의 크기. 클수록 똑똑하지만 느립니다.
        super().__init__()                       # 부모(nn.Module) 준비 — 빠뜨리면 오류

        # nn.Sequential = 위에서 아래로 순서대로 통과시키는 통로
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),   # 4개 -> 128개, 그리고 구부리기
            nn.Linear(hidden, hidden), nn.ReLU(),      # 128개 -> 128개, 또 구부리기
            nn.Linear(hidden, action_dim),             # 128개 -> 2개 (행동별 Q값)
        )
        # 마지막에 활성화 함수를 안 붙이는 이유:
        #   Q값은 -100 일 수도 +500 일 수도 있습니다.
        #   Sigmoid 같은 걸 붙이면 0~1 로 눌려서 표현을 못 합니다.

    def forward(self, s):
        # 데이터가 지나가는 길. model(s) 라고 쓰면 이 함수가 불립니다.
        return self.net(s)                       # 각 행동의 Q값이 한 줄로 나온다


# ============================================================
# 3. 학습 루프 5단계 — 앞으로 모든 코드에 그대로 반복됩니다
# ============================================================

model = QNetwork(4, 2).to(device)                # 신경망을 만들고 CPU/GPU 로 보낸다
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#           ^^^^^^^^^^^^^^^^ 고칠 대상(신경망 속 숫자들)을 알려 준다
#                                                lr = 한 번에 얼마나 움직일지
criterion = nn.MSELoss()                         # 얼마나 틀렸는지 재는 자 (평균제곱오차)

# 진짜 게임 대신 아무 숫자나 만들어 연습합니다 (패턴만 익히는 게 목적)
states = torch.randn(64, 4, device=device)       # 상황 64개, 각각 숫자 4개
targets = torch.randn(64, 2, device=device)      # 정답 64개, 각각 숫자 2개

for step in range(200):                          # 200번 반복해서 배운다
    pred = model(states)                         # ① 예측한다
    loss = criterion(pred, targets)              # ② 얼마나 틀렸나 잰다
    optimizer.zero_grad()                        # ③ 지난 기울기를 지운다 (안 지우면 쌓임!)
    loss.backward()                              # ④ 어디를 고칠지 계산한다
    optimizer.step()                             # ⑤ 그 방향으로 한 걸음 간다

    if step % 50 == 0:                           # 50번마다 한 번씩만 출력
        print(f"step {step:3d}  loss = {loss.item():.4f}")
        # .item() = 텐서 안의 숫자 하나를 꺼내는 것 (그냥 출력하면 tensor(...) 로 나옴)

# 정답이 아무 숫자라서 손실이 0까지 내려가진 않습니다.
# 여기서 볼 것은 "숫자가 줄어드는가" 하나뿐입니다.

# ── 4교시에서 이어받음 — DQN, Double DQN 구현 ──
import gymnasium as gym                        # 게임(환경)을 만들어 주는 도구
import torch                                   # 파이토치
import torch.nn as nn                          # 신경망 부품 상자
import numpy as np                             # 숫자 계산 도구

# CartPole = 막대가 쓰러지지 않게 수레를 좌우로 미는 게임
# 1초 버틸 때마다 1점. 500점이 만점입니다.
env = gym.make("CartPole-v1")

obs_dim = env.observation_space.shape[0]       # 상황을 나타내는 숫자 개수 = 4
                                               # (수레 위치, 수레 속도, 막대 각도, 막대 회전속도)
n_actions = env.action_space.n                 # 할 수 있는 행동 = 2 (왼쪽 밀기, 오른쪽 밀기)

# ── 신경망을 두 개 만듭니다. 구조는 똑같고 역할만 다릅니다 ──
#   q_net    : 지금 학습하는 신경망      (화살)
#   q_target : 목표를 계산할 때만 쓰는 것 (과녁) — 한동안 고정해 둡니다
# 과녁이 계속 움직이면 맞출 수가 없습니다. 그래서 따로 둡니다.
q_net = QNetwork(obs_dim, n_actions)           # 3교시에서 만든 그 QNetwork 입니다
q_target = QNetwork(obs_dim, n_actions)
q_target.load_state_dict(q_net.state_dict())   # 처음엔 둘을 똑같이 맞춰 둔다
                                               # state_dict = 신경망 속 숫자들의 묶음

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)   # q_net 만 학습시킨다
                                               # q_target 은 학습하지 않습니다 (복사만 받음)
buffer = ReplayBuffer(50_000)                  # 1교시에서 만든 일기장. 5만 줄까지.

gamma, batch_size = 0.99, 64
# gamma      = 미래를 얼마나 챙길지 (0.99 = 거의 다 챙긴다)
# batch_size = 한 번 배울 때 일기장에서 몇 줄을 꺼낼지

# eps = 아무 행동이나 해볼 확률 (탐험)
#   1.0 에서 시작해 매 판 0.995배씩 줄어들어 0.05 에서 멈춥니다.
#   처음엔 마구 둘러보고, 나중엔 아는 길로 갑니다.
eps, eps_min, eps_decay = 1.0, 0.05, 0.995

DOUBLE = True                                  # <- Double DQN 스위치 (False 로 바꿔 비교해 보세요)


def train_step():
    """일기장에서 조금 꺼내 한 번 배우는 함수. 위의 5단계가 그대로 들어 있습니다."""

    # 일기장에서 과거 경험 64줄을 무작위로 꺼냅니다 (섞어서 꺼내기)
    s, a, r, s_next, done = buffer.sample(batch_size)

    # q_net(s) 는 모든 행동의 값을 냅니다. 예: [3.2, 5.1]
    # 그런데 우리는 "내가 실제로 한 행동"의 값만 필요합니다 -> gather 로 뽑습니다.
    #   unsqueeze(1) : 모양을 (64,) -> (64,1) 로. gather 가 이 모양을 요구합니다.
    #   gather(1, ...) : 각 줄에서 지정한 자리 하나씩 뽑기
    #   squeeze(1)   : 뽑고 나서 (64,1) -> (64,) 로 되돌리기
    q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():                      # 여기부터는 '정답 만들기' — 미분 금지 구역
        if DOUBLE:
            # Double DQN: 고르는 사람과 점수 매기는 사람을 나눈다
            best_a = q_net(s_next).argmax(1, keepdim=True)          # 고르기는 학습 중인 쪽
            q_next = q_target(s_next).gather(1, best_a).squeeze(1)  # 점수는 고정된 쪽
        else:
            # 그냥 DQN: 타깃넷이 고르기도 하고 점수도 매긴다
            q_next = q_target(s_next).max(1).values

        # (1 - done) 이 오늘 가장 중요한 부분입니다.
        #   판이 끝났으면 done=1 -> 뒤쪽이 0이 되어 "미래 점수 없음"이 됩니다.
        #   이걸 빼먹으면 끝난 뒤에도 점수가 계속 더해져 값이 무한히 커집니다.
        target = r + gamma * q_next * (1 - done)

    # smooth_l1_loss = 크게 틀렸을 때 벌점을 완만하게 주는 손실
    #   강화학습은 가끔 값이 크게 튀는데, 제곱(MSE)을 쓰면 그 하나에 학습이 휘둘립니다.
    loss = nn.functional.smooth_l1_loss(q, target)

    optimizer.zero_grad()                      # ③ 지난 기울기 지우기
    loss.backward()                            # ④ 어디를 고칠지 계산
    optimizer.step()                           # ⑤ 한 걸음 이동


returns = []                                   # 판마다 받은 총점을 기록 (= 학습곡선)

for episode in range(400):                     # 400판을 한다
    s, _ = env.reset()                         # 새 판 시작. s = 지금 상황
    total, done = 0, False                     # total = 이번 판 점수

    while not done:                            # 판이 끝날 때까지 반복
        # ── 행동 고르기 (엡실론 그리디) ──
        if np.random.rand() < eps:
            a = env.action_space.sample()      # eps 확률로 아무거나 (탐험)
        else:
            with torch.no_grad():              # 행동만 고를 땐 미분 준비 불필요
                # 상황을 텐서로 바꿔 신경망에 넣고, 가장 큰 Q값의 번호를 고른다
                a = q_net(torch.as_tensor(s, dtype=torch.float32)).argmax().item()

        # ── 그 행동을 실제로 해본다 ──
        s_next, r, term, trunc, _ = env.step(a)
        # term  = 진짜로 끝났다 (막대가 쓰러졌다)
        # trunc = 시간이 다 됐다 (500점을 채웠다)
        done = term or trunc

        # 일기장에 한 줄 적는다. float(term) 을 쓰는 이유:
        #   시간이 다 돼서 끝난 것(trunc)은 '실패'가 아닙니다.
        #   여기서 미래를 0으로 만들면 잘한 판을 나쁘게 배웁니다.
        buffer.push(s, a, r, s_next, float(term))

        s, total = s_next, total + r           # 다음 상황으로 넘어가고 점수 누적

        if len(buffer) >= 1000:                # 일기장이 1000줄 넘게 쌓인 뒤부터 배운다
            train_step()                       # (너무 적을 때 배우면 그 몇 개만 외웁니다)

    # ── 한 판이 끝난 뒤 ──
    eps = max(eps_min, eps * eps_decay)        # 탐험 확률을 조금 줄인다 (0.05 아래로는 안 감)

    # 20판마다 과녁을 최신 것으로 갈아 끼웁니다.
    #   너무 자주 하면 과녁이 흔들리고, 너무 안 하면 낡은 과녁을 보고 쏩니다.
    if episode % 20 == 0:
        q_target.load_state_dict(q_net.state_dict())

    returns.append(total)                      # 이번 판 점수 기록

    if episode % 20 == 0:                      # 20판마다 진행 상황 출력
        print(f"ep {episode:3d}  return {np.mean(returns[-20:]):6.1f}  eps {eps:.2f}")
        # 최근 20판 평균을 보는 이유: 한 판 점수는 너무 들쭉날쭉합니다.

# 최근 평균이 475를 넘으면 CartPole 을 풀었다고 봅니다.
# 400판으로는 거기까지 못 갈 수도 있습니다 — 그러면 range(400) 을 늘려 보세요.

# ── 5교시에서 이어받음 — Policy Gradient 소개 ──
import gymnasium as gym                    # 게임(환경) 만드는 도구
import torch                               # 파이토치
import torch.nn as nn                      # 신경망 부품 상자
import numpy as np                         # 숫자 계산 도구

env = gym.make("CartPole-v1")              # 막대 세우기 게임

# ── 지금까지와 완전히 다른 접근입니다 ──
# DQN 은 "각 행동이 얼마나 좋은지(Q값)"를 배우고, 그중 큰 걸 골랐습니다.
# 여기서는 "무엇을 할지"를 바로 배웁니다. 값을 안 거칩니다.
policy = nn.Sequential(
    nn.Linear(4, 128), nn.ReLU(),          # 상황 4개 -> 128개로 늘리고 구부린다
    nn.Linear(128, 2),                     # 128개 -> 2개 (행동별 '점수', logit 이라고 부름)
)
# 마지막에 softmax 를 안 붙이는 이유:
#   아래 Categorical 이 안에서 알아서 확률로 바꿔 줍니다. 두 번 하면 안 됩니다.

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)   # 정책망을 학습시킨다
gamma = 0.99                               # 미래를 얼마나 챙길지

print('=' * 52)
print('5교시 REINFORCE 학습 시작 (600판)')
print('=' * 52)
# ↑ 전체본으로 돌리면 앞 교시 출력과 섞이므로 어디서부터인지 표시합니다

score_log = []                             # 판별 점수를 모아 둘 곳 (평균을 내려고)

for episode in range(600):                 # 600판을 한다
    s, _ = env.reset()                     # 새 판 시작
    log_probs, rewards, done = [], [], False
    # log_probs = 내가 고른 행동의 '로그 확률'을 모아 둘 곳 (나중에 미분할 값)
    # rewards   = 매 걸음 받은 점수를 모아 둘 곳

    # ── 1) 한 판을 끝까지 해본다 (배우지 않고 모으기만) ──
    while not done:
        logits = policy(torch.as_tensor(s, dtype=torch.float32))   # 행동별 점수
        dist = torch.distributions.Categorical(logits=logits)      # 점수 -> 확률분포로
        a = dist.sample()                  # 확률대로 뽑는다 (항상 최선을 고르지 않음!)
                                           # 이게 곧 탐험입니다. eps 가 따로 필요 없습니다.

        log_probs.append(dist.log_prob(a)) # "그 행동을 뽑을 확률"의 로그를 저장
                                           # 왜 로그인가: 미분이 아주 간단해지기 때문 (로그미분 트릭)

        s, r, term, trunc, _ = env.step(a.item())   # 실제로 그 행동을 해본다
                                                    # a.item() = 텐서에서 숫자 하나 꺼내기
        done = term or trunc               # 쓰러졌거나 시간이 다 됐으면 끝
        rewards.append(r)                  # 받은 점수 저장

    # ── 2) 판이 끝난 뒤, 각 시점의 '앞으로 받은 총점'을 계산한다 ──
    G, returns = 0.0, []
    for r in reversed(rewards):            # 뒤에서부터 거꾸로 온다
        G = r + gamma * G                  # 지금 점수 + 감마 x 뒤에서 온 총점
        returns.insert(0, G)               # 앞쪽에 끼워 넣어 원래 순서로 되돌린다
    # 거꾸로 도는 이유: 앞에서부터 하면 매번 끝까지 다시 더해야 해서 느립니다.

    returns = torch.tensor(returns)        # 파이썬 목록 -> 텐서

    # ── 베이스라인 (오늘 배운 그것) ──
    #   평균을 빼고 표준편차로 나눕니다.
    #   빼도 되는 이유: 상태에만 의존하는 값을 빼면 평균은 그대로이고 흔들림만 줄어듭니다.
    #   안 빼면 CartPole 처럼 점수가 전부 양수인 문제에서 "다 잘했다"로 보입니다.
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    # + 1e-8 은 0으로 나누는 사고를 막는 안전장치입니다.

    # ── 3) 배우기 ──
    # 앞의 마이너스가 붙는 이유:
    #   옵티마이저는 손실을 "줄이는" 방향으로만 움직입니다.
    #   우리는 점수를 "키우고" 싶으니 부호를 뒤집어 줍니다.
    # 점수(returns)가 큰 행동일수록 그 행동의 확률을 크게 올립니다.
    loss = -(torch.stack(log_probs) * returns).sum()
    # torch.stack = 낱개 텐서들을 하나로 쌓기

    optimizer.zero_grad()                  # ③ 지난 기울기 지우기
    loss.backward()                        # ④ 어디를 고칠지 계산
    optimizer.step()                       # ⑤ 한 걸음 이동

    score_log.append(sum(rewards))         # 이번 판 점수 기록

    if episode % 50 == 0:                  # 50판마다 출력
        recent = np.mean(score_log[-50:])  # 최근 50판 평균
        print(f"ep {episode:3d}  최근 50판 평균 {recent:6.1f}   이번 판 {sum(rewards):.0f}")
        # 한 판 점수만 보면 11 → 28 → 17 처럼 들쭉날쭉해서
        # 학습이 되는지 안 되는지 알 수가 없습니다.
        # 그래서 평균을 함께 봅니다. 4교시 DQN 과 같은 방식입니다.

print()
print(f'  처음 50판 평균 {np.mean(score_log[:50]):6.1f}  →  마지막 50판 평균 {np.mean(score_log[-50:]):6.1f}')
print('  → 이 두 숫자를 비교하시면 학습이 됐는지 한눈에 보입니다.')

# DQN 과 비교해 보세요.
#   DQN     : 매 걸음마다 조금씩 배운다 (일기장에서 꺼내서)
#   REINFORCE : 한 판이 끝나야 배운다 (끝까지 가봐야 총점을 알 수 있으니까)
#   -> 그래서 REINFORCE 는 배우는 횟수가 적고, 결과가 더 들쭉날쭉합니다.
#      이 문제를 6교시 Actor-Critic 이 해결합니다.

# ── 6교시에서 이어받음 — Actor-Critic 소개 ──
import torch                                   # 파이토치
import torch.nn as nn                           # 신경망 부품 상자


class ActorCritic(nn.Module):
    """
    배우(Actor) 와 평론가(Critic) 를 한 몸에 넣은 신경망.

    배우   : 무엇을 할지 정한다
    평론가 : 지금 상황이 얼마나 좋은지 점수를 매긴다

    왜 한 몸인가요?
      둘 다 "지금 상황을 이해하는 일"은 똑같이 필요합니다.
      그 공통 부분(몸통)을 함께 쓰고, 마지막 판단만 나눕니다.
      사람으로 치면 눈은 하나인데 판단하는 머리만 둘인 셈입니다.
    """

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()                      # 부모 준비 — 빠뜨리면 오류

        # 몸통 — 상황을 이해하는 부분. 배우와 평론가가 함께 씁니다.
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),   # 4개 -> 128개, 구부리기
        )

        self.actor_head = nn.Linear(hidden, action_dim)  # 배우 머리: 행동별 점수(logit)
        self.critic_head = nn.Linear(hidden, 1)          # 평론가 머리: 값 하나 V(s)

    def forward(self, s):
        h = self.body(s)                        # 먼저 몸통을 통과 (상황 이해)
        return (
            self.actor_head(h),                 # 무엇을 할지 (여러 개)
            self.critic_head(h).squeeze(-1),    # 얼마나 좋은지 (하나)
        )
        # squeeze(-1) = (배치, 1) -> (배치,) 로 눌러서 모양 맞추기
        # 안 하면 나중에 계산할 때 모양이 안 맞아 조용히 잘못된 값이 나옵니다.


def a2c_loss(logits, value, action, td_target, entropy_coef=0.01):
    """
    A2C 의 손실. 세 조각을 더한 것입니다.
    (전체 학습 루프는 다음 7교시에 완성합니다)
    """

    dist = torch.distributions.Categorical(logits=logits)   # 점수 -> 확률분포로

    # ── 어드밴티지 = "예상보다 얼마나 좋았나" ──
    #   td_target : 실제로 겪어 보니 이 정도였다
    #   value     : 평론가가 예상했던 값
    #   차이가 +면 예상보다 좋았다는 뜻 -> 그 행동의 확률을 올린다
    advantage = (td_target - value).detach()
    # ★ .detach() 를 빠뜨리면 안 됩니다 ★
    #   이걸 안 붙이면 "배우를 고치려던 신호"가 평론가까지 흘러갑니다.
    #   평론가는 자기 손실(아래 critic_loss)로만 배워야 합니다.
    #   섞이면 평론가가 "값을 정확히 맞히기"가 아니라
    #   "배우가 좋아할 값 내놓기"를 배워 버립니다.

    # ── 조각 ① 배우 손실 ──
    # 어드밴티지가 큰 행동일수록 그 행동의 확률을 크게 올린다
    # 앞의 마이너스는 "키우고 싶다"를 "줄이고 싶다"로 뒤집은 것
    actor_loss = -(dist.log_prob(action) * advantage).mean()

    # ── 조각 ② 평론가 손실 ──
    # 평론가의 예상(value)이 실제(td_target)에 가까워지게
    critic_loss = nn.functional.mse_loss(value, td_target)

    # ── 조각 ③ 엔트로피 (탐험 유지) ──
    # 엔트로피 = 얼마나 헷갈려 하는가.
    #   (0.5, 0.5) 면 크고, (0.99, 0.01) 이면 작습니다.
    entropy = dist.entropy().mean()

    # 세 조각을 더합니다. 계수는 셋의 크기를 맞추는 저울입니다.
    #   0.5  : 평론가 손실은 값이 크게 나오기 쉬워 절반으로 눌러 줍니다.
    #          안 그러면 배우 손실이 묻혀 버립니다.
    #   -0.01 : 엔트로피는 빼 줍니다. 헷갈리는 쪽이 손실이 작아지므로
    #           너무 일찍 한 행동만 고집하지 않게 붙잡아 줍니다.
    return actor_loss + 0.5 * critic_loss - entropy_coef * entropy

# ============================================================
# 잘 만들어졌는지 확인 (이 부분이 있어야 실행했을 때 결과가 보입니다)
# ============================================================
torch.manual_seed(0)

print('배우와 평론가가 잘 만들어졌는지 확인합니다.')
print()

model = ActorCritic(state_dim=4, action_dim=2)   # 상황 4개, 행동 2개

s = torch.randn(8, 4)                            # 상황 8개를 한 번에 (배치)
logits, value = model(s)

print(f'  입력 모양            {tuple(s.shape)}')
print(f'  배우가 낸 것(logits) {tuple(logits.shape)}   <- 상황마다 행동 2개의 점수')
print(f'  평론가가 낸 것(V)    {tuple(value.shape)}      <- 상황마다 값 하나')
print()

# 손실이 계산되는지 확인
action = torch.randint(0, 2, (8,))               # 실제로 한 행동
td_target = torch.randn(8)                       # 목표값 (가짜)
loss = a2c_loss(logits, value, action, td_target)

print(f'  a2c_loss 계산 결과   {loss.item():.4f}')
print(f'  미분 연결 상태       {loss.requires_grad}  <- True 여야 학습이 됩니다')
print()
print('  -> 여기까지 나오면 정상입니다. 7교시에서 이 조각들로 학습 루프를 완성합니다.')

# ── 오늘 이 교시 — A2C 구현 ──
import gymnasium as gym                        # 게임(환경) 만드는 도구
import torch                                   # 파이토치
import torch.nn as nn                          # 신경망 부품 상자
import numpy as np                             # 숫자 계산 도구

env = gym.make("CartPole-v1")                  # 막대 세우기 게임
model = ActorCritic(4, 2)                      # 6교시에서 만든 배우+평론가 신경망
optimizer = torch.optim.Adam(model.parameters(), lr=7e-4)
# lr 이 DQN(1e-3)보다 조금 작습니다.
#   배우와 평론가가 옵티마이저 하나를 함께 쓰기 때문에 더 민감합니다.

gamma, n_steps = 0.99, 5
# gamma   = 미래를 얼마나 챙길지
# n_steps = 몇 걸음 걸어 보고 한 번 배울지
#   REINFORCE 는 한 판(수백 걸음)을 다 가야 배웠습니다 -> 느리고 들쭉날쭉
#   여기서는 5걸음마다 배웁니다 -> 자주, 안정적으로

s, _ = env.reset()                             # 첫 상황
ep_return, returns = 0, []
# ep_return = 지금 진행 중인 판의 누적 점수
# returns   = 끝난 판들의 점수 목록 (= 학습곡선)

print('=' * 52)
print('7교시 A2C 학습 시작 (20,000번)')
print('=' * 52)
# ↑ 전체본으로 돌리면 앞 교시 출력과 섞이므로 어디서부터인지 표시합니다

for update in range(20000):                    # 20,000번 배운다 (판 수가 아니라 '배운 횟수')
    # 왜 20,000번인가?
    #   20,000 x 5걸음 = 100,000걸음입니다.
    #   5교시 REINFORCE 는 600판을 돌았는데 판이 길어지면서
    #   실제로는 90,000걸음쯤 겪었습니다.
    #   3,000번(=15,000걸음)만 돌리면 A2C 가 6배 적은 경험으로 배우는 셈이라
    #   "A2C 가 더 좋다면서 왜 점수가 낮죠?" 가 됩니다. 비교가 불공정합니다.

    # ── 1) n걸음만 걸어 보며 재료를 모은다 ──
    log_probs, values, rewards, entropies, dones = [], [], [], [], []

    for _ in range(n_steps):                   # 5걸음
        logits, v = model(torch.as_tensor(s, dtype=torch.float32))
        # logits = 배우가 낸 행동별 점수,  v = 평론가가 매긴 이 상황의 값

        dist = torch.distributions.Categorical(logits=logits)   # 점수 -> 확률분포
        a = dist.sample()                      # 확률대로 행동을 뽑는다

        s_next, r, term, trunc, _ = env.step(a.item())   # 실제로 해본다
        done = term or trunc                   # 쓰러졌거나 시간이 다 됐으면 끝

        log_probs.append(dist.log_prob(a))     # 그 행동의 로그확률 (나중에 미분할 값)
        values.append(v)                       # 평론가의 예상값
        rewards.append(r)                      # 받은 점수
        dones.append(done)                     # 여기서 판이 끝났나
        entropies.append(dist.entropy())       # 얼마나 헷갈려 했나

        ep_return += r                         # 이번 판 점수 누적
        s = s_next                             # 다음 상황으로 이동

        if done:                               # 판이 끝났으면
            returns.append(ep_return)          # 점수를 기록하고
            ep_return = 0                      # 초기화한 뒤
            s, _ = env.reset()                 # 새 판을 시작한다
            # 5걸음을 다 못 채워도 괜찮습니다. 이어서 새 판을 걷습니다.

    # ── 2) 목표값(정답)을 만든다 ──
    # 5걸음까지만 봤으니, 그 뒤는 평론가의 예상으로 대신합니다.
    with torch.no_grad():                      # 목표는 상수여야 한다 -> 미분 금지
        _, v_last = model(torch.as_tensor(s, dtype=torch.float32))

    R, td_targets = v_last, []
    for r, d in zip(reversed(rewards), reversed(dones)):   # 뒤에서부터 거꾸로
        R = r + gamma * R * (1 - d)            # 지금 점수 + 감마 x 뒤에서 온 값
                                               # (1-d) : 판이 끝난 자리에서는 뒤를 끊는다
        td_targets.insert(0, R)                # 앞에 끼워 원래 순서로

    td_targets = torch.stack(td_targets).detach()   # 낱개들을 하나로 쌓고 미분 끊기
    values = torch.stack(values)                    # 평론가 예상들도 하나로

    # ── 3) 어드밴티지 = 실제 - 예상 ──
    advantages = td_targets - values           # +면 예상보다 좋았다

    # ── 4) 손실 세 조각 ──
    actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
    # ★ advantages.detach() ★ 배우 쪽 신호가 평론가로 흘러가지 않게 끊는다

    critic_loss = advantages.pow(2).mean()     # 예상과 실제의 차이를 제곱해 평균
                                               # (= MSE 와 같습니다)
    entropy = torch.stack(entropies).mean()    # 평균적으로 얼마나 헷갈려 했나

    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
    #                   ^^^ 평론가 손실은 커지기 쉬워 절반으로 누른다
    #                                      ^^^^ 엔트로피는 빼서 탐험을 조금 붙잡는다

    # ── 5) 배우기 ──
    optimizer.zero_grad()                      # 지난 기울기 지우기
    loss.backward()                            # 어디를 고칠지 계산

    # nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    # ↑ 기울기가 너무 크면 잘라 주는 안전장치입니다. 지금은 꺼 두었습니다.
    #
    # ★ 왜 껐는지가 오늘 배울 것 중 하나입니다 ★
    #   원래 A2C 는 이 줄을 넣는 것이 관례입니다. 학습이 갑자기 무너지는 것을 막으니까요.
    #   그런데 강사 맥북에서 직접 재봤더니 이 줄이 학습을 막고 있었습니다.
    #
    #   같은 3,000번 학습, seed 3개 평균
    #     0.5 로 자름 : 18.3점
    #     5.0 로 자름 : 26.2점
    #     자르지 않음 : 71.0점
    #
    #   0.5 는 너무 세게 자른 값이었습니다. 신호가 거의 남지 않습니다.
    #   → 관례값이라고 그냥 쓰면 안 됩니다. 직접 재보고 판단해야 합니다.
    #
    #   폭주가 걱정되면 큰 값(예: 10.0)으로 켜 두시면 됩니다.

    optimizer.step()                           # 한 걸음 이동

    if update % 1000 == 0:                     # 1000번 배울 때마다 출력
        if returns:
            print(f"update {update:5d}  최근 20판 평균 {np.mean(returns[-20:]):6.1f}"
                  f"   (끝난 판 {len(returns)}개)")
        else:
            print(f"update {update:5d}  아직 한 판도 안 끝났습니다 (곧 나옵니다)")
        # 최근 20판 평균을 보는 이유: 한 판 점수는 너무 들쭉날쭉합니다.
        # update 0 에서도 한 줄은 찍습니다 — 아무것도 안 나오면
        # "멈춘 건가?" 하고 불안해지기 때문입니다.

print()
print(f'  처음 20판 평균 {np.mean(returns[:20]):6.1f}  ->  마지막 20판 평균 {np.mean(returns[-20:]):6.1f}')
print(f'  전체 {len(returns)}판, 약 100,000걸음 학습했습니다.')
print('  -> 이 두 숫자를 비교하시면 학습이 됐는지 한눈에 보입니다.')

# ============================================================
# 왜 20,000번인가 (중요)
#   20,000 x 5걸음 = 100,000걸음.
#   5교시 REINFORCE 가 겪은 것과 비슷하게 맞춘 것입니다.
#   그래야 두 방법을 공정하게 비교할 수 있습니다.
#
# ★ 점수 편차가 아주 큽니다 ★
#   A2C 는 시드(무작위 출발점)에 따라 결과가 크게 달라집니다.
#   한 번 돌려 보고 "안 되네" 하지 마시고 두세 번 돌려 보세요.
#
# 시간이 없으면 range(20000) 을 5000 쯤으로 줄이셔도 됩니다.
#   점수는 낮게 나오지만 "오르고 있다"는 것은 보입니다.
# ============================================================
