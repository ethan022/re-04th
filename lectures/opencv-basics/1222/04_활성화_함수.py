# 활성화 함수란?

# 뉴런이 "켜질지 말지" 결정하는 스위치!

# 비유: 시험 합격/불합격 기준

# 시험 점수가 나왔을 때:
# - 60점 미만: 불합격 (뉴런 꺼짐)
# - 60점 이상: 합격 (뉴런 켜짐)

# 활성화 함수 = 이런 기준을 정하는 것!


# 왜 필요할까?

# 만약 활성화 함수가 없다면?

# 예) 3개 층을 쌓아도:
# 층1: y = 2x
# 층2: y = 3y = 3(2x) = 6x
# 층3: y = 4y = 4(6x) = 24x
# 결국: y = 24x (그냥 곱셈!)

# 아무리 층을 많이 쌓아도 간단한 계산밖에 못함!

print("=== 활성화 함수 없으면? ===")
print("입력: 2")
print("층1: 2 × 2 = 4")
print("층2: 4 × 3 = 12")
print("층3: 12 × 4 = 48")
print()
print("결과: 2 × 2 × 3 × 4 = 48")
print("→ 그냥 곱셈만 됩니다!")
print("→ 복잡한 패턴을 배울 수 없어요")
print()


# 활성화 함수를 쓰면?

# 비선형성을 추가!
# = 복잡한 곡선을 표현할 수 있음
# = 복잡한 패턴을 학습할 수 있음

print("=== 활성화 함수 있으면? ===")
print("입력: 2")
print("층1: 2 × 2 = 4 → ReLU → 4")
print("층2: 4 × 3 = 12 → ReLU → 12")
print("층3: 12 × 4 = 48 → ReLU → 48")
print()
print("→ 각 층마다 복잡한 변환 가능!")
print("→ 고양이/강아지 같은 복잡한 것도 구분!")
print()


# 주요 활성화 함수들 (쉽게!)


# 1. ReLU (렐루)
# = 음수는 0으로, 양수는 그대로

# 비유: 빚은 탕감!
# - 마이너스 통장 (-100원) → 0원으로
# - 플러스 통장 (100원) → 그대로 100원

# 수식: y = max(0, x)
# -5 → 0
#  0 → 0
#  5 → 5

print("=== ReLU 활성화 함수 ===")
print("입력 -5 → 출력 0")
print("입력  0 → 출력 0")
print("입력  5 → 출력 5")
print()
print("장점: 계산이 빠름! (단순 비교)")
print("단점: 음수는 전부 죽어버림")
print()


# 2. Sigmoid (시그모이드)
# = 모든 값을 0~1 사이로 압축

# 비유: 확률로 변환
# - 큰 양수 → 1에 가까움 (거의 확실)
# - 큰 음수 → 0에 가까움 (거의 아님)
# - 0 → 0.5 (반반)

print("=== Sigmoid 활성화 함수 ===")
print("입력 -10 → 출력 0.0000... (거의 0)")
print("입력   0 → 출력 0.5")
print("입력  10 → 출력 0.9999... (거의 1)")
print()
print("장점: 확률로 해석 가능!")
print("단점: 학습이 느려질 수 있음")
print("사용: 주로 마지막 층(출력층)에서")
print()


# 3. Tanh (탄젠트 하이퍼볼릭)
# = 모든 값을 -1~1 사이로 압축

# Sigmoid와 비슷하지만:
# - Sigmoid: 0~1
# - Tanh: -1~1 (중심이 0)

print("=== Tanh 활성화 함수 ===")
print("입력 -10 → 출력 -1")
print("입력   0 → 출력  0")
print("입력  10 → 출력  1")
print()
print("장점: 중심이 0이라 학습이 조금 더 좋음")
print("사용: 요즘은 ReLU를 더 많이 씀")
print()


# 어떤 활성화 함수를 쓸까?

print("=== 활성화 함수 선택 가이드 ===")
print()
print("중간 층(은닉층): ReLU 사용!")
print("→ 가장 많이 쓰임, 빠르고 효과적")
print()
print("출력층 (문제 유형별):")
print("- 이진 분류 (개/고양이): Sigmoid")
print("  → 0~1 사이 확률로 표현")
print()
print("- 다중 분류 (개/고양이/새): Softmax")
print("  → 여러 클래스의 확률 합=1")
print()
print("- 회귀 (숫자 예측): 활성화 함수 없음")
print("  → 그냥 숫자를 바로 출력")
print()


# 실제로 사용해보기

import torch
import torch.nn as nn

print("=== PyTorch로 활성화 함수 사용 ===")
print()

# ReLU 테스트
relu = nn.ReLU()
test_input = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

print("입력:", test_input.tolist())
print("ReLU 출력:", relu(test_input).tolist())
print("→ 음수는 0, 양수는 그대로!")
print()

# Sigmoid 테스트
sigmoid = nn.Sigmoid()
print("Sigmoid 출력:", [f"{x:.3f}" for x in sigmoid(test_input).tolist()])
print("→ 모두 0~1 사이로!")
print()

# Tanh 테스트
tanh = nn.Tanh()
print("Tanh 출력:", [f"{x:.3f}" for x in tanh(test_input).tolist()])
print("→ 모두 -1~1 사이로!")
print()


# 신경망에서 사용하기

print("=== 간단한 신경망 만들기 ===")
print()

model = nn.Sequential(
    nn.Linear(10, 20),  # 입력층 → 은닉층
    nn.ReLU(),          # 활성화! (ReLU)
    nn.Linear(20, 10),  # 은닉층 → 은닉층
    nn.ReLU(),          # 활성화! (ReLU)
    nn.Linear(10, 1),   # 은닉층 → 출력층
    nn.Sigmoid()        # 활성화! (Sigmoid, 0~1 출력)
)

print(model)
print()
print("설명:")
print("- 중간층: ReLU 사용 (일반적)")
print("- 마지막층: Sigmoid 사용 (이진 분류)")
print()


# 시각화 (옵션)

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 활성화 함수 그래프 그리기 ===")

x = np.linspace(-5, 5, 100)

# ReLU
relu_y = np.maximum(0, x)

# Sigmoid
sigmoid_y = 1 / (1 + np.exp(-x))

# Tanh
tanh_y = np.tanh(x)

plt.figure(figsize=(12, 4))

# ReLU 그래프
plt.subplot(1, 3, 1)
plt.plot(x, relu_y, 'b-', linewidth=2)
plt.title('ReLU: 음수는 0')
plt.xlabel('입력')
plt.ylabel('출력')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Sigmoid 그래프
plt.subplot(1, 3, 2)
plt.plot(x, sigmoid_y, 'g-', linewidth=2)
plt.title('Sigmoid: 0~1로 압축')
plt.xlabel('입력')
plt.ylabel('출력')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

# Tanh 그래프
plt.subplot(1, 3, 3)
plt.plot(x, tanh_y, 'r-', linewidth=2)
plt.title('Tanh: -1~1로 압축')
plt.xlabel('입력')
plt.ylabel('출력')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()


# 핵심 정리

print("\n" + "="*60)
print("핵심 정리")
print("="*60)
print()
print("1. 활성화 함수 = 뉴런이 켜질지 결정하는 스위치")
print()
print("2. 왜 필요? → 복잡한 패턴 학습을 위해!")
print()
print("3. ReLU (가장 많이 씀)")
print("   - 음수 → 0, 양수 → 그대로")
print("   - 중간층에서 사용")
print()
print("4. Sigmoid")
print("   - 0~1로 압축")
print("   - 출력층(이진 분류)에서 사용")
print()
print("5. 기본 선택: 중간층은 ReLU, 출력층은 문제에 맞게!")
print()
print("="*60)
print()
print("비유로 기억하기:")
print("- ReLU = 빚은 탕감 (음수는 0)")
print("- Sigmoid = 확률 변환 (0~1)")
print("- 활성화 함수 = 합격/불합격 기준")
print("="*60)
