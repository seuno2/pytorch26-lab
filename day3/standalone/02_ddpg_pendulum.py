# ============================================================
# 3일차 2교시 — DDPG 구현
# 복사해서 그대로 실행하면 됩니다. 고칠 것 없습니다.
# ------------------------------------------------------------
# 이 교시 코드는 앞 교시의 변수·클래스를 이어 씁니다.
# 그래서 이 블록에는 **여기까지 필요한 코드가 전부** 들어 있습니다.
# (수업용 코드만 따로 복사하면 NameError 가 납니다 — 그건 정상입니다.)
# ============================================================

# ── 2일차 1교시에서 이어받음 ──
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

# ── 1교시에서 이어받음 — DDPG 소개 ──
import torch                                   # 파이토치
import torch.nn as nn                           # 신경망 부품 상자


# ============================================================
# 오늘부터 행동이 달라집니다.
#   어제까지 : "왼쪽? 오른쪽?" — 고르는 문제 (이산)
#   오늘부터 : "힘을 1.37만큼" — 값을 정하는 문제 (연속)
#
# 왜 어제 방식이 안 통할까요?
#   어제는 모든 행동의 Q값을 내놓고 그중 max 를 골랐습니다.
#   그런데 힘의 크기는 -2.000 부터 2.000 까지 무한히 많습니다.
#   전부 계산해서 max 를 고르는 게 불가능합니다.
#   -> 그래서 "행동을 직접 내놓는 신경망"을 따로 둡니다. 그게 Actor 입니다.
# ============================================================


class Actor(nn.Module):
    """
    상황을 받아서 '할 행동'을 바로 내놓는 신경망.

    어제 정책망과 다른 점:
      어제는 확률을 내놓고 뽑았습니다 (0번 70%, 1번 30%)
      오늘은 값 하나를 딱 정합니다 (힘 1.37) — 이걸 '결정적'이라고 합니다.
    """

    def __init__(self, state_dim, action_dim, max_action):
        # state_dim  = 상황을 나타내는 숫자 개수 (Pendulum 은 3개)
        # action_dim = 행동 값이 몇 개인지 (Pendulum 은 1개 — 회전시킬 힘)
        # max_action = 행동의 최대 크기 (Pendulum 은 2.0)
        super().__init__()                      # 부모 준비 — 빠뜨리면 오류

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),      # 3개 -> 256개, 구부리기
            nn.Linear(256, 256), nn.ReLU(),            # 256 -> 256, 또 구부리기
            nn.Linear(256, action_dim), nn.Tanh(),     # 256 -> 1개, 그리고 Tanh
        )
        # 왜 마지막에 Tanh 를 붙일까요?
        #   Tanh 는 어떤 숫자가 들어와도 -1 ~ +1 사이로 눌러 줍니다.
        #   그래야 행동이 엉뚱하게 큰 값(예: 500)이 되는 것을 막습니다.
        #   어제 Q값에는 활성화를 안 붙였는데, 그건 값의 범위가 정해져
        #   있지 않았기 때문입니다. 행동은 범위가 정해져 있습니다.

        self.max_action = max_action            # 나중에 곱해 줄 배율을 기억해 둔다

    def forward(self, s):
        # Tanh 로 -1~1 이 된 값에 max_action 을 곱해 실제 범위로 늘린다
        #   예: 0.685 x 2.0 = 1.37
        return self.net(s) * self.max_action


class Critic(nn.Module):
    """
    (상황, 행동) 을 함께 받아서 "이 조합이 얼마나 좋은지" 점수를 매기는 신경망.

    ★ 어제 DQN 과 가장 크게 다른 점 ★
      어제 : 상황만 받고 -> 모든 행동의 Q값을 한꺼번에 내놓음
      오늘 : 상황 + 행동을 같이 받고 -> 그 하나에 대한 Q값만 내놓음
      행동이 무한히 많으니 "다 내놓기"가 불가능하기 때문입니다.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            #          ^^^^^^^^^^^^^^^^^^^^^ 상황과 행동을 붙여서 넣으므로 크기를 더한다
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),                  # 결과는 점수 하나
        )

    def forward(self, s, a):
        # torch.cat = 두 텐서를 옆으로 이어 붙이기
        #   상황 3개 + 행동 1개 = 4개짜리 한 줄이 됩니다.
        # dim=-1 = "맨 마지막 축 방향으로" 붙인다는 뜻
        # squeeze(-1) = (배치, 1) -> (배치,) 로 눌러 모양 맞추기
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


def soft_update(target, source, tau=0.005):
    """
    과녁(타깃 네트워크)을 조금씩만 따라오게 하는 함수. DDPG 와 SAC 가 함께 씁니다.

    어제와 무엇이 다른가요?
      어제 : 20판마다 과녁을 통째로 갈아 끼웠습니다 (계단처럼 툭 바뀜)
      오늘 : 매번 0.5% 씩만 섞습니다 (미끄러지듯 천천히 따라옴)

    왜 바꾸나요?
      연속 행동은 값이 아주 예민합니다. 과녁이 계단처럼 툭툭 튀면
      학습이 그때마다 흔들리다 무너집니다. 그래서 천천히 섞습니다.
    """
    # zip = 두 신경망의 숫자 뭉치를 짝지어 하나씩 꺼낸다
    for tp, sp in zip(target.parameters(), source.parameters()):
        # 새 과녁 = 0.5% 는 최신 것 + 99.5% 는 원래 과녁
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)
        # .data 를 쓰는 이유: 이건 '학습'이 아니라 '복사'입니다.
        #   미분 기록을 남기지 않고 값만 바꿔치기합니다.

# ============================================================
# 잘 만들어졌는지 확인 (이 부분이 있어야 실행했을 때 결과가 보입니다)
# ============================================================
import copy
torch.manual_seed(0)

print('배우와 평론가가 잘 만들어졌는지 확인합니다.')
print()

actor = Actor(state_dim=3, action_dim=1, max_action=2.0)   # Pendulum 규격
critic = Critic(state_dim=3, action_dim=1)

s = torch.randn(6, 3)                            # 상황 6개
a = actor(s)                                     # 배우가 행동을 내놓는다
q = critic(s, a)                                 # 평론가가 점수를 매긴다

print(f'  입력 상황        {tuple(s.shape)}')
print(f'  배우가 낸 행동   {tuple(a.shape)}   범위 [{a.min().item():+.3f}, {a.max().item():+.3f}]')
print('                    <- Tanh x 2.0 이라 -2 ~ +2 안에 있어야 합니다')
print(f'  평론가가 낸 값   {tuple(q.shape)}')
print()

# soft_update 가 실제로 조금씩만 옮기는지 확인
target = copy.deepcopy(actor)
with torch.no_grad():
    for p in actor.parameters():
        p.add_(1.0)                              # 본체를 확 바꿔 본다

before = target.net[0].weight[0, 0].item()
soft_update(target, actor, tau=0.005)
after = target.net[0].weight[0, 0].item()

print('  soft_update 확인')
print(f'    과녁 값  {before:.5f} -> {after:.5f}   (움직인 폭 {after - before:+.5f})')
print('    본체는 1.0 만큼 바뀌었는데 과녁은 0.005 만 따라왔습니다.')
print()
print('  -> 여기까지 나오면 정상입니다. 2교시에서 이 셋으로 학습을 돌립니다.')

# ── 오늘 이 교시 — DDPG 구현 ──
import gymnasium as gym                        # 게임(환경) 만드는 도구
import torch                                   # 파이토치
import torch.nn as nn                          # 신경망 부품 상자
import numpy as np                             # 숫자 계산 도구
import copy                                    # 통째로 복사할 때 쓰는 도구

# Pendulum = 막대를 흔들어서 거꾸로 세우는 게임
#   점수가 항상 음수입니다. 0에 가까울수록 잘한 것입니다.
#   아무렇게나 하면 -1200 근처가 나옵니다.
env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]     # 상황을 나타내는 숫자 = 3개
                                               # (막대 각도의 cos, sin, 회전 속도)
action_dim = env.action_space.shape[0]         # 행동 값 = 1개 (돌리는 힘)
max_action = float(env.action_space.high[0])   # 힘의 최대 크기 = 2.0

# ── 신경망 4개를 만듭니다 ──
#   actor    : 무엇을 할지 정하는 쪽 (배우)
#   critic   : 그게 얼마나 좋은지 점수 매기는 쪽 (평론가)
#   actor_t  : 배우의 과녁 (천천히 따라오는 복사본)
#   critic_t : 평론가의 과녁
actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)

actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
# deepcopy = 완전히 똑같은 별개의 복사본을 만든다
#   그냥 actor_t = actor 라고 하면 같은 것을 가리켜서 과녁 역할을 못 합니다.

actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)    # 배우용 옵티마이저
critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)  # 평론가용 옵티마이저
# 배우의 학습률이 10배 작습니다.
#   평론가가 아직 엉터리인데 배우가 빨리 따라가면 엉뚱한 걸 배웁니다.
#   "평론가가 먼저 실력을 갖추고, 배우는 천천히" 라는 뜻입니다.

buffer = ReplayBuffer(100_000, action_dtype=torch.float32)
# ★ action_dtype=torch.float32 를 꼭 넣어야 합니다 ★
#   행동이 1.37 같은 실수인데 정수로 두면 1 로 잘려 학습이 통째로 망가집니다.

gamma, batch_size, noise_std = 0.99, 128, 0.1
# gamma      = 미래를 얼마나 챙길지
# batch_size = 한 번에 128개씩 꺼내 배운다
# noise_std  = 행동에 섞을 흔들림의 크기 (아래에서 설명)


def train_step():
    """일기장에서 꺼내 한 번 배우는 함수. 평론가 먼저, 배우 나중."""

    s, a, r, s_next, done = buffer.sample(batch_size)   # 128개 꺼내기

    # ── ① 평론가 배우기 ──
    with torch.no_grad():                      # 정답 만들기 — 미분 금지 구역
        # 다음 상황에서 '과녁 배우'가 할 행동을 구하고,
        # 그 행동의 점수를 '과녁 평론가'에게 물어본다.
        target_q = critic_t(s_next, actor_t(s_next))

        # 정답 = 지금 받은 점수 + 감마 x 다음 상황의 값
        # (1 - done) : 판이 끝났으면 뒤는 없다
        y = r + gamma * target_q * (1 - done)

    # 평론가의 예상이 정답에 가까워지게 한다
    critic_loss = nn.functional.mse_loss(critic(s, a), y)
    critic_opt.zero_grad()                     # 지난 기울기 지우기
    critic_loss.backward()                     # 어디를 고칠지 계산
    critic_opt.step()                          # 한 걸음 이동

    # ── ② 배우 배우기 ──
    # 배우의 목표는 "평론가에게 좋은 점수를 받는 행동을 내놓는 것" 입니다.
    #   critic(s, actor(s)) = 배우가 낸 행동을 평론가가 채점한 점수
    #   그 점수를 '키우고' 싶으니 마이너스를 붙여 '줄이는' 문제로 바꿉니다.
    actor_loss = -critic(s, actor(s)).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    # 여기서 평론가도 같이 바뀌지 않나요?
    #   critic_opt 는 평론가 것만 고치고 actor_opt 는 배우 것만 고칩니다.
    #   옵티마이저가 나뉘어 있어서 서로 건드리지 않습니다.

    # ── ③ 과녁을 아주 조금 따라오게 한다 ──
    soft_update(actor_t, actor)                # 0.5% 씩만 섞는다
    soft_update(critic_t, critic)


returns = []                                   # 판마다 점수 기록 (= 학습곡선)

for episode in range(200):                     # 200판을 한다
    s, _ = env.reset()                         # 새 판 시작
    total, done = 0.0, False

    while not done:
        # ── 행동 고르기 ──
        with torch.no_grad():                  # 행동만 고를 땐 미분 준비 불필요
            a = actor(torch.as_tensor(s, dtype=torch.float32)).numpy()

        # 배우는 같은 상황에서 항상 같은 답을 냅니다 (결정적).
        # 그대로 두면 새로운 것을 전혀 안 해봅니다 = 탐험이 없습니다.
        # 그래서 답에 살짝 흔들림을 더합니다. "32도" 대신 "32.4도" 처럼.
        #   어제 eps-greedy 가 "가끔 아예 딴 행동"이었다면,
        #   오늘은 "늘 조금씩 다르게" 입니다. 연속 행동이라 이 방식이 맞습니다.
        a += np.random.normal(0, noise_std * max_action, action_dim)

        # 흔들림을 더하다 범위를 벗어날 수 있으니 잘라 준다
        a = a.clip(-max_action, max_action)

        s_next, r, term, trunc, _ = env.step(a)    # 실제로 해본다
        done = term or trunc

        buffer.push(s, a, r, s_next, float(term))  # 일기장에 적는다
        s, total = s_next, total + r               # 다음 상황으로, 점수 누적

        if len(buffer) >= 1000:                # 1000줄 이상 쌓인 뒤부터 배운다
            train_step()

    returns.append(total)                      # 이번 판 점수 기록

    if episode % 10 == 0:                      # 10판마다 출력
        print(f"ep {episode:3d}  최근 10ep 평균 리턴 {np.mean(returns[-10:]):7.1f}")

# -1500 근처에서 시작해 -200 근처까지 올라오면 성공입니다.
# 학습에 6분쯤 걸립니다 — 돌려 놓고 설명을 들으시면 됩니다.
