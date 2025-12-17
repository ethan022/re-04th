"""
============================================================
02. 경사 하강법 (Gradient Descent)
============================================================

이 파일에서 배울 내용:
1. 경사 하강법이 무엇인지
2. 컴퓨터가 어떻게 최적의 직선을 찾는지
3. 학습률의 중요성
4. 실제 코드로 구현하기
5. 시각화로 이해하기

소요 시간: 약 25분
난이도: ★★☆☆☆ (초급)
"""

# ============================================================
# STEP 1: 경사 하강법이란?
# ============================================================

# 상상해보세요: 안개 낀 산에서 길을 잃었습니다!
#
# 상황:
# - 당신은 산 어딘가에 있습니다
# - 마을은 가장 낮은 곳에 있습니다
# - 앞이 보이지 않습니다!
#
# 해결 방법:
# - 발밑의 경사만 느끼면서
# - "지금 서있는 곳에서 가장 가파르게 내려가는 방향" 찾기
# - 그 방향으로 한 걸음 내딛기
# - 반복!

print("=" * 60)
print("STEP 1: 경사 하강법의 개념")
print("=" * 60)

print("\n[비유] 안개 낀 산에서 내려오기")
print("  산 정상 = 오차가 큰 상태 (나쁜 모델)")
print("  마을 (골짜기) = 오차가 작은 상태 (좋은 모델)")
print("  발밑의 경사 = 기울기 (gradient)")
print("  한 걸음의 크기 = 학습률 (learning rate)")

print("\n[핵심]")
print("  경사 하강법 = 오차를 조금씩 줄여가며 최적의 모델을 찾는 방법")
print()

# ============================================================
# STEP 2: 경사 하강법의 수식 (쉽게 설명)
# ============================================================

print("=" * 60)
print("STEP 2: 작동 원리 (수식)")
print("=" * 60)

# 수식: 파라미터(새) = 파라미터(현재) - 학습률 × 기울기
#
# 말로 풀면:
# "현재 위치에서, 기울기 방향의 반대로, 조금씩 이동"
#
# 왜 반대 방향?
# - 기울기 = 함수가 증가하는 방향
# - 우리는 감소하는 방향으로 가야 함 (오차를 줄여야 하니까!)
# - 그래서 마이너스(-)를 붙임

print("\n[수식] 파라미터(새) = 파라미터(현재) - 학습률 × 기울기")
print("\n각 용어 설명:")
print("  - 파라미터: w(기울기), b(절편) - 우리가 찾고자 하는 값")
print("  - 학습률: 한 번에 얼마나 크게 이동할지 (보폭)")
print("  - 기울기: 현재 위치에서 오차가 증가하는 방향")
print("\n왜 '마이너스(-)'?")
print("  → 기울기는 '증가' 방향이므로")
print("  → 반대로 가야 '감소' (오차 줄이기)")
print()

# ============================================================
# STEP 3: 학습률의 중요성
# ============================================================

print("=" * 60)
print("STEP 3: 학습률(Learning Rate)의 중요성")
print("=" * 60)

# 학습률 = 한 걸음의 크기

print("\n[학습률이 너무 크면]")
print("  문제: 최적점을 지나쳐서 왔다갔다")
print("  비유: 산을 내려오는데 너무 큰 보폭으로 뛰면")
print("       반대편 산으로 넘어감!")
print("  결과: 학습이 발산 (오차가 줄어들지 않음)")

print("\n[학습률이 너무 작으면]")
print("  문제: 목표에 도달하는데 시간이 너무 오래 걸림")
print("  비유: 아주 작은 걸음으로 조금씩 이동")
print("  결과: 학습이 너무 느림")

print("\n[적절한 학습률]")
print("  → 보통 0.001 ~ 0.1 사이 값 사용")
print("  → 실험을 통해 최적값 찾기")
print()

# ============================================================
# STEP 4: 코드로 구현하기
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("STEP 4: 경사 하강법 직접 구현하기")
print("=" * 60)

# 데이터 생성
# 목표: y = 2x + 3 이라는 관계를 데이터에서 찾아내기
np.random.seed(42)  # 결과를 일정하게 하기 위한 시드
x = np.random.randn(100)  # 랜덤 x값 100개
y = 2 * x + 3 + np.random.randn(100) * 0.5  # y = 2x + 3 + 노이즈

print("\n[데이터 생성]")
print(f"  데이터 개수: {len(x)}개")
print(f"  실제 관계: y = 2x + 3 (+ 약간의 노이즈)")
print(f"  목표: 컴퓨터가 이 관계를 스스로 찾아내기!")

# 파라미터 초기화 (랜덤하게 시작)
w = 0.0  # 기울기를 0으로 시작
b = 0.0  # 절편을 0으로 시작

# 하이퍼파라미터 설정
lr = 0.1  # 학습률 (learning rate)
epochs = 100  # 반복 횟수

print(f"\n[초기 설정]")
print(f"  w (기울기) 초기값: {w}")
print(f"  b (절편) 초기값: {b}")
print(f"  학습률: {lr}")
print(f"  반복 횟수: {epochs}")

# 학습 과정 기록용
loss_history = []  # 오차 기록
w_history = []  # w 변화 기록
b_history = []  # b 변화 기록

print("\n[학습 시작]")
print("-" * 60)

# 경사 하강법 실행!
for epoch in range(epochs):
    # STEP 1: 현재 파라미터로 예측하기
    y_pred = w * x + b

    # STEP 2: 오차 계산 (MSE)
    loss = np.mean((y - y_pred) ** 2)  # 평균 제곱 오차
    loss_history.append(loss)

    # STEP 3: 기울기 계산 (미분)
    # dw = "w를 조금 바꾸면 오차가 어떻게 변하는가?"
    # db = "b를 조금 바꾸면 오차가 어떻게 변하는가?"
    dw = np.mean(2 * (y_pred - y) * x)  # w에 대한 기울기
    db = np.mean(2 * (y_pred - y))  # b에 대한 기울기

    # STEP 4: 파라미터 업데이트
    # 기울기 반대 방향으로 조금씩 이동
    w = w - lr * dw  # w를 기울기 반대 방향으로 조정
    b = b - lr * db  # b를 기울기 반대 방향으로 조정

    # 기록
    w_history.append(w)
    b_history.append(b)

    # 20번마다 진행상황 출력
    if epoch % 20 == 0:
        print(f'Epoch {epoch:3d}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}')

print("-" * 60)
print(f'\n[최종 결과]')
print(f'  w = {w:.4f} (목표: 2.0)')
print(f'  b = {b:.4f} (목표: 3.0)')
print(f'  최종 오차: {loss_history[-1]:.4f}')

# 얼마나 잘 찾았는지 평가
w_error = abs(w - 2.0)
b_error = abs(b - 3.0)
print(f'\n[정확도]')
print(f'  w 오차: {w_error:.4f}')
print(f'  b 오차: {b_error:.4f}')
if w_error < 0.1 and b_error < 0.1:
    print(f'  평가: 매우 정확하게 찾았습니다! ✓')
else:
    print(f'  평가: 더 많은 학습이 필요합니다.')
print()

# ============================================================
# STEP 5: 시각화로 이해하기
# ============================================================

print("=" * 60)
print("STEP 5: 학습 과정 시각화")
print("=" * 60)
print("\n그래프가 열립니다. 창을 닫으면 다음으로 진행됩니다.\n")

# 그래프 3개 그리기
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. 오차 감소 그래프
axes[0].plot(loss_history, color='red', linewidth=2)
axes[0].set_xlabel('Epoch (반복 횟수)', fontsize=10)
axes[0].set_ylabel('Loss (오차)', fontsize=10)
axes[0].set_title('오차 변화: 점점 줄어듭니다!', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 2. w 변화 그래프
axes[1].plot(w_history, color='blue', linewidth=2, label='w 변화')
axes[1].axhline(y=2.0, color='green', linestyle='--', linewidth=2, label='목표값 (2.0)')
axes[1].set_xlabel('Epoch (반복 횟수)', fontsize=10)
axes[1].set_ylabel('w 값', fontsize=10)
axes[1].set_title('w(기울기)가 목표값으로 수렴', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. b 변화 그래프
axes[2].plot(b_history, color='purple', linewidth=2, label='b 변화')
axes[2].axhline(y=3.0, color='green', linestyle='--', linewidth=2, label='목표값 (3.0)')
axes[2].set_xlabel('Epoch (반복 횟수)', fontsize=10)
axes[2].set_ylabel('b 값', fontsize=10)
axes[2].set_title('b(절편)가 목표값으로 수렴', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# STEP 6: 개선된 경사 하강법들
# ============================================================

print("=" * 60)
print("STEP 6: 경사 하강법의 한계와 개선 방법")
print("=" * 60)

print("\n[한계 1] 지역 최솟값 (Local Minimum) 문제")
print("  문제: 전체에서 가장 낮은 곳이 아니라")
print("        근처의 움푹 파인 곳에 갇힐 수 있음")
print("  비유: 작은 웅덩이에 빠져서 더 큰 골짜기를 못 찾음")

print("\n[개선 1] SGD (Stochastic Gradient Descent)")
print("  방법: 전체 데이터 대신 일부만 사용")
print("  장점: 빠르고, 무작위성이 지역 최솟값 탈출에 도움")
print("  단점: 경로가 지그재그")

print("\n[개선 2] Momentum (모멘텀)")
print("  방법: 이전 이동 방향의 관성을 반영")
print("  비유: 공이 굴러가듯 탄력을 받아 움푹 파인 곳을 넘어감")
print("  장점: 지역 최솟값 탈출에 유리")

print("\n[개선 3] Adam (가장 많이 사용!)")
print("  방법: 학습률을 파라미터별로 적응적으로 조절")
print("  장점: 대부분의 경우 잘 작동")
print("  특징: 현재 가장 널리 사용되는 최적화 알고리즘")
print()

# ============================================================
# 요약 및 다음 단계
# ============================================================

print("=" * 60)
print("🎉 축하합니다! 경사 하강법을 이해했습니다!")
print("=" * 60)

print("\n[오늘 배운 내용]")
print("  ✓ 경사 하강법 = 산을 내려가듯 오차를 줄이기")
print("  ✓ 기울기 = 오차가 증가하는 방향")
print("  ✓ 학습률 = 한 걸음의 크기 (너무 크거나 작으면 안됨)")
print("  ✓ 반복 = 조금씩 파라미터를 조정")
print("  ✓ 개선 방법 = SGD, Momentum, Adam")

print("\n[핵심 코드 4단계]")
print("  1. 예측: y_pred = w * x + b")
print("  2. 오차: loss = mean((y - y_pred)²)")
print("  3. 기울기: dw, db 계산 (미분)")
print("  4. 업데이트: w = w - lr * dw")

print("\n[실전 팁]")
print("  1. scikit-learn은 자동으로 경사 하강법 사용")
print("  2. 딥러닝에서는 Adam이 가장 많이 사용됨")
print("  3. 학습률은 0.001부터 시작해보기")

print("\n[다음 단계]")
print("  → 03_다중선형회귀.py: 여러 변수를 동시에 고려하기")

print("\n[연습 문제]")
print("  1. 학습률을 0.01, 0.5로 바꿔보기 (어떻게 달라질까?)")
print("  2. epochs를 10, 1000으로 바꿔보기")
print("  3. 초기값 w=10, b=-5로 시작하면? (수렴할까?)")

print("\n" + "=" * 60)
print("Happy Learning! 🚀")
print("=" * 60)
