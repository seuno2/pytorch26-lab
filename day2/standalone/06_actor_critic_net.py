# ============================================================
# 2일차 6교시 — Actor-Critic 소개
# 복사해서 그대로 실행하면 됩니다. 고칠 것 없습니다.
# ------------------------------------------------------------
# 이 교시 코드는 앞 교시의 변수·클래스를 이어 씁니다.
# 그래서 이 블록에는 **여기까지 필요한 코드가 전부** 들어 있습니다.
# (수업용 코드만 따로 복사하면 NameError 가 납니다 — 그건 정상입니다.)
# ============================================================

# ── 1교시에서 이어받음 — DQN 소개 ──
import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """경험 재현 버퍼 — DQN, DDPG, SAC 3일 내내 재사용합니다"""
    def __init__(self, capacity=100_000, action_dtype=torch.int64):
        # action_dtype: 오늘 DQN은 행동이 "몇 번 행동"인 정수라 int64입니다.
        # 3일차 DDPG·SAC는 행동이 연속값(실수 벡터)이므로 float32로 바꿔 씁니다
        #   buffer = ReplayBuffer(100_000, action_dtype=torch.float32)
        # int64로 두면 실수 행동이 정수로 잘려 학습이 통째로 망가집니다.
        self.buffer = deque(maxlen=capacity)
        self.action_dtype = action_dtype

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        # np.array로 한 번 묶고 텐서로 바꿉니다.
        # 배열 리스트를 텐서로 바로 만들면 파이토치가 하나씩 복사해 매우 느립니다.
        return (torch.as_tensor(np.array(s), dtype=torch.float32),
                torch.as_tensor(np.array(a), dtype=self.action_dtype),
                torch.as_tensor(np.array(r), dtype=torch.float32),
                torch.as_tensor(np.array(s_next), dtype=torch.float32),
                torch.as_tensor(np.array(done), dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

# ── 2교시에서 이어받음 — Double DQN 소개 ──
import torch

# DQN vs Double DQN — 목표(target) 계산의 차이
@torch.no_grad()
def dqn_target(q_target, r, s_next, done, gamma=0.99):
    max_q = q_target(s_next).max(dim=1).values          # 타깃넷이 선택+평가
    return r + gamma * max_q * (1 - done)

@torch.no_grad()
def double_dqn_target(q_online, q_target, r, s_next, done, gamma=0.99):
    best_a = q_online(s_next).argmax(dim=1, keepdim=True)   # 선택: 온라인넷
    max_q = q_target(s_next).gather(1, best_a).squeeze(1)   # 평가: 타깃넷
    return r + gamma * max_q * (1 - done)

# ── 3교시에서 이어받음 — PyTorch 소개 및 구현 ──
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Tensor & Autograd ──
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x          # y = x² + 3x
y.backward()
print(x.grad)               # dy/dx = 2x + 3 = 7

# ── 2. nn.Module로 Q-네트워크 정의 ──
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
    def forward(self, s):
        return self.net(s)      # 각 행동의 Q값 벡터

# ── 3. 학습 루프 5단계 (회귀 예제로 패턴 익히기) ──
model = QNetwork(4, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

states = torch.randn(64, 4, device=device)       # 가짜 배치
targets = torch.randn(64, 2, device=device)

for step in range(200):
    pred = model(states)             # ① 순전파
    loss = criterion(pred, targets)  # ② 손실
    optimizer.zero_grad()            # ③ 기울기 초기화
    loss.backward()                  # ④ 역전파
    optimizer.step()                 # ⑤ 갱신
    if step % 50 == 0:
        print(f"step {step:3d}  loss = {loss.item():.4f}")

# ── 4교시에서 이어받음 — DQN, Double DQN 구현 ──
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]     # 4
n_actions = env.action_space.n               # 2

# 신경망을 두 개 만듭니다. 구조는 똑같고 역할만 다릅니다.
#  q_net    : 지금 학습하는 신경망 (화살)
#  q_target : 목표 계산 전용, 한동안 고정 (과녁)
# 처음에는 둘을 똑같이 맞춰 둡니다.
q_net = QNetwork(obs_dim, n_actions)
q_target = QNetwork(obs_dim, n_actions)
q_target.load_state_dict(q_net.state_dict())
optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer(50_000)

gamma, batch_size = 0.99, 64
# eps = 아무 행동이나 해볼 확률.
#   1.0에서 시작해 매 판 0.995배씩 줄어들어 0.05에서 멈춥니다.
#   처음엔 마구 둘러보고, 나중엔 아는 길로 가는 것입니다.
eps, eps_min, eps_decay = 1.0, 0.05, 0.995
DOUBLE = True                                # ← Double DQN 스위치

def train_step():
    # 상자에서 과거 경험 64개를 무작위로 꺼냅니다.
    s, a, r, s_next, done = buffer.sample(batch_size)

    # q_net(s) 는 모든 행동의 값을 냅니다. 예: [3.2, 5.1]
    # 그중 "내가 실제로 한 행동"의 값만 뽑아야 합니다 → gather
    #   unsqueeze(1) : 모양을 (64,) → (64,1) 로. gather 가 요구하는 형태입니다.
    #   squeeze(1)   : 뽑고 나서 (64,1) → (64,) 로 되돌립니다.
    q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if DOUBLE:   # 선택은 온라인넷, 평가는 타깃넷
            best_a = q_net(s_next).argmax(1, keepdim=True)
            q_next = q_target(s_next).gather(1, best_a).squeeze(1)
        else:        # 타깃넷이 선택+평가 (바닐라 DQN)
            q_next = q_target(s_next).max(1).values
        # (1 - done) 이 오늘 가장 중요한 부분입니다.
        # 게임이 끝났다면 done=1 → 뒤쪽이 0이 되어 "미래 점수 없음"이 됩니다.
        # 이걸 빼먹으면 끝난 뒤에도 점수가 계속 더해져 값이 무한히 커집니다.
        target = r + gamma * q_next * (1 - done)

    # smooth_l1_loss : 오차가 클 때 덜 민감한 손실.
    #   강화학습은 가끔 튀는 값이 있어서 이 손실이 더 안정적입니다.
    loss = nn.functional.smooth_l1_loss(q, target)
    optimizer.zero_grad()   # 이전 기울기 지우기
    loss.backward()         # 기울기 계산
    optimizer.step()        # 가중치 갱신

returns = []
for episode in range(400):
    s, _ = env.reset()
    total, done = 0, False
    while not done:
        if np.random.rand() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = q_net(torch.as_tensor(s, dtype=torch.float32)).argmax().item()
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buffer.push(s, a, r, s_next, float(term))
        s, total = s_next, total + r
        if len(buffer) >= 1000:
            train_step()
    eps = max(eps_min, eps * eps_decay)
    # 20판마다 과녁을 최신 것으로 갈아 끼웁니다.
    # 너무 자주 하면 과녁이 흔들리고, 너무 안 하면 낡은 과녁을 보고 쏩니다.
    if episode % 20 == 0:
        q_target.load_state_dict(q_net.state_dict())
    returns.append(total)
    if episode % 20 == 0:
        print(f"ep {episode:3d}  return {np.mean(returns[-20:]):6.1f}  eps {eps:.2f}")
# 평균 리턴이 475를 넘으면 CartPole 해결!

# ── 5교시에서 이어받음 — Policy Gradient 소개 ──
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

env = gym.make("CartPole-v1")

policy = nn.Sequential(
    nn.Linear(4, 128), nn.ReLU(),
    nn.Linear(128, 2),          # 행동별 로짓
)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
gamma = 0.99

for episode in range(600):
    s, _ = env.reset()
    log_probs, rewards, done = [], [], False
    while not done:                              # 1) 에피소드 수집
        logits = policy(torch.as_tensor(s, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        log_probs.append(dist.log_prob(a))
        s, r, term, trunc, _ = env.step(a.item())
        done = term or trunc
        rewards.append(r)

    G, returns = 0.0, []                         # 2) 리턴 계산 (뒤에서부터)
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 정규화(간이 베이스라인)

    # 앞의 마이너스: 파이토치는 "줄이는" 방향으로만 움직입니다.
    # 우리는 점수를 "키우고" 싶으니 부호를 뒤집습니다.
    # 점수(returns)가 큰 행동일수록 그 행동의 확률을 크게 올립니다.
    loss = -(torch.stack(log_probs) * returns).sum()
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if episode % 50 == 0:
        print(f"ep {episode:3d}  return {sum(rewards):.0f}")

# ── 오늘 이 교시 — Actor-Critic 소개 ──
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    """몸통을 공유하고 머리만 둘 — A2C의 표준 구조"""
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, action_dim)   # 정책 로짓
        self.critic_head = nn.Linear(hidden, 1)           # V(s)

    def forward(self, s):
        h = self.body(s)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

# 손실 구성 미리보기 (다음 교시에 전체 루프 완성)
def a2c_loss(logits, value, action, td_target, entropy_coef=0.01):
    dist = torch.distributions.Categorical(logits=logits)
    advantage = (td_target - value).detach()      # Critic 신호는 Actor로 역전파 금지
    actor_loss = -(dist.log_prob(action) * advantage).mean()
    critic_loss = nn.functional.mse_loss(value, td_target)
    entropy = dist.entropy().mean()               # 탐험 유지 보너스
    return actor_loss + 0.5 * critic_loss - entropy_coef * entropy
