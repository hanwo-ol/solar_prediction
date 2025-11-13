베이지안 딥러닝이라 너무 램 많이 먹네








### 모델 및 학습 프로세스의 수학적 정의

#### 1. 전체 목표 (Overall Objective)

우리의 목표는 신경망 가중치에 대한 **메타-사전 분포(meta-prior distribution)** $p(\phi|\theta)$를 학습하는 것입니다. 이 분포는 학습 가능한 **메타-파라미터(meta-parameter)** $\theta$에 의해 결정됩니다. 잘 학습된 메타-사전 분포는 다음과 같은 능력을 갖습니다:
1.  새로운 태스크 $\mathcal{T} _i$가 주어졌을 때, 해당 태스크의 소량 데이터(서포트 셋 $D _i^{(S)}$)를 사용하여 베이즈 정리를 통해 효율적으로 **태스크-특화 사후 분포(task-specific posterior distribution)** $q(\phi|\lambda _i)$로 적응(adapt)할 수 있습니다.
2.  이 사후 분포를 사용하여 불확실성을 포함한 정확한 예측을 수행할 수 있습니다.

이 목표는 태스크 분포 $p(\mathcal{T})$에 대한 증거 하한(Evidence Lower Bound, ELBO)을 최대화하는 메타-파라미터 $\theta^*$를 찾는 것으로 공식화됩니다.

$$
\theta^* = \arg\max _{\theta} \mathbb{E} _{\mathcal{T} _i \sim p(\mathcal{T})} \left[ \mathbb{E} _{q(\phi|\lambda _i)}[\log p(D _i^{(Q)}|\phi)] - \text{KL}(q(\phi|\lambda _i)  \Vert  p(\phi|\theta)) \right]
$$

#### 2. 모델 및 확률 분포 정의

*   **기저 모델 (Base Model)**: 베이지안 U-Net 모델을 $f _\phi(\cdot)$로 표기하며, $\phi \in \mathbb{R}^D$는 모델의 모든 가중치 및 편향 파라미터를 포함하는 벡터입니다.

*   **메타-사전 분포 $p(\phi|\theta)$ (Meta-Prior)**:
    메타-러너(`MetaLearner`)의 `prior _net`에 해당하며, 모든 태스크에 대한 일반적인 지식을 담고 있습니다. 우리는 이 분포를 대각 공분산을 갖는 가우시안 분포로 가정합니다.

$$
p(\phi|\theta) = \prod _{j=1}^{D} \mathcal{N}(\phi _j | \mu _{\theta,j}, \sigma^2 _{\theta,j})
$$

여기서 메타-파라미터 $\theta = \{\mu _{\theta,j}, \rho _{\theta,j}\} _{j=1}^D$는 `prior _net`의 모든 `mu`와 `rho` 파라미터에 해당하며, $\sigma _{\theta,j} = \text{softplus}(\rho _{\theta,j})$ 입니다.

*   **근사 사후 분포 $q(\phi|\lambda _i)$ (Approximate Posterior)**:
    태스크 $\mathcal{T} _i$의 서포트 셋 $D _i^{(S)}$에 적응한 후의 가중치 분포입니다. `higher` 라이브러리의 `fmodel`에 해당하며, 변분 파라미터 $\lambda _i$에 의해 결정됩니다. 이 또한 대각 가우시안 분포로 가정합니다.

$$
q(\phi|\lambda _i) = \prod _{j=1}^{D} \mathcal{N}(\phi _j | \mu _{\lambda _i,j}, \sigma^2 _{\lambda _i,j})
$$

여기서 변분 파라미터 $\lambda _i = \{\mu _{\lambda _i,j}, \rho _{\lambda _i,j}\} _{j=1}^D$는 내부 루프 최적화를 통해 찾아집니다.

*   **가능도 함수 $p(Y|X, \phi)$ (Likelihood)**:
    주어진 가중치 $\phi$를 갖는 모델이 입력 $X$에 대해 출력 $Y$를 생성할 확률입니다. 우리는 MSE 손실 함수를 사용하므로, 이는 출력에 등방성 가우시안 노이즈가 있다고 가정하는 것과 같습니다.

$$
p(Y|X, \phi) = \mathcal{N}(Y | f _\phi(X), \sigma _{noise}^2 \mathbf{I})
$$

로그 가능도는 $\log p(Y|X, \phi) \propto - \Vert Y - f _\phi(X) \Vert ^2 _F$ 이므로, MSE 손실을 최소화하는 것은 로그 가능도를 최대화하는 것과 같습니다.

#### 3. 메타-훈련 프로세스 (Meta-Training Process)

`meta _train _one _epoch` 함수는 다음의 이중 루프 최적화를 수행합니다.

**For** each training epoch:
1.  **태스크 샘플링**: 데이터셋에서 태스크 $\mathcal{T} _i = (D _i^{(S)}, D _i^{(Q)})$를 샘플링합니다.
2.  **내부 루프 (Inner Loop): 태스크 적응**
    *   사후 분포의 파라미터를 사전 분포의 파라미터로 초기화합니다: $\lambda _i^{(0)} \leftarrow \theta$.
    *   $M$번의 경사 상승 단계(`INNER _STEPS`)를 통해 서포트 셋 $D _i^{(S)}$에 대한 ELBO를 최대화하여 $\lambda _i$를 최적화합니다. $m=0, \dots, M-1$에 대해:

$$
\mathcal{L} _{inner}(\lambda _i^{(m)}) = \mathbb{E} _{q(\phi|\lambda _i^{(m)})} [\log p(D _i^{(S)}|\phi)] - w _{KL} \cdot \text{KL}(q(\phi|\lambda _i^{(m)})  \Vert  p(\phi|\theta))
$$

$$
\lambda _i^{(m+1)} \leftarrow \lambda _i^{(m)} + \alpha \nabla _{\lambda _i^{(m)}} \mathcal{L} _{inner}(\lambda _i^{(m)})
$$

여기서 $\alpha$는 `INNER _LR`, $w _{KL}$은 `KL _WEIGHT`에 해당합니다. 이 과정은 `higher` 라이브러리의 `diffopt.step()`을 통해 수행됩니다. 최종적으로 적응된 파라미터를 $\lambda _i^* = \lambda _i^{(M)}$로 둡니다.

3.  **외부 루프 (Outer Loop): 메타-업데이트**
    *   적응된 사후 분포 $q(\phi|\lambda _i^*)$를 사용하여 쿼리 셋 $D _i^{(Q)}$에 대한 재구성 손실을 계산합니다.

$$
\mathcal{L} _{outer}(\theta, \lambda _{i}^{\ast}) = -\mathbb{E} _{q(\phi \vert \lambda _{i}^{\ast})} [\log p(D _i^{(Q)} \vert \phi)]
$$

*   이 손실을 메타-파라미터 $\theta$에 대해 미분하여 메타-그래디언트를 계산하고, AdamW 옵티마이저를 사용하여 $\theta$를 업데이트합니다.

$$
\theta \leftarrow \text{AdamW}(\theta, \nabla _\theta \mathcal{L} _{outer}(\theta, \lambda _i^*), \eta _{meta})
$$

여기서 $\eta _{meta}$는 `META _LR`입니다. $\nabla _\theta$는 내부 루프의 전체 최적화 과정을 통과하는 고차 미분(higher-order gradient)입니다.

#### 4. 메타-테스팅 프로세스 (Meta-Testing Process)

`meta _evaluate` 함수는 학습된 메타-사전 분포 $p(\phi|\theta^*)$를 사용하여 새로운 태스크 $\mathcal{T} _{new}$에 대한 예측을 수행합니다.

1.  **태스크 적응**: 새로운 태스크의 서포트 셋 $D _{new}^{(S)}$를 사용하여 훈련 시와 동일한 내부 루프 과정을 $M _{eval}$번(`NUM _ADAPTATION _STEPS`) 수행하여 최종 사후 분포 파라미터 $\lambda _{new}^*$를 얻습니다.
2.  **예측 및 불확실성 추정**: 쿼리 셋 입력 $X _{new}^{(Q)}$에 대해, 적응된 사후 분포 $q(\phi|\lambda _{new}^*)$에서 $S$개(`NUM _EVAL _SAMPLES`)의 가중치 샘플을 추출합니다.
    *   $\phi^{(s)} \sim q(\phi|\lambda _{new}^*)$ for $s=1, \dots, S$.
    *   각 샘플에 대해 예측을 수행합니다: $\hat{Y}^{(s)} = f _{\phi^{(s)}}(X _{new}^{(Q)})$.
3.  **결과 종합**:
    *   **최종 예측 (평균)**: 몬테카를로 추정으로 예측의 기댓값을 계산합니다.

$$
\bar{Y} _{pred} \approx \frac{1}{S} \sum _{s=1}^{S} \hat{Y}^{(s)}
$$

*   **예측 불확실성 (표준편차)**: 예측의 표준편차를 계산합니다.

$$
U _{pred} \approx \sqrt{\frac{1}{S-1} \sum _{s=1}^{S} (\hat{Y}^{(s)} - \bar{Y} _{pred})^2}
$$

---
### 요약: 표기법과 코드의 연결

| 수학적 표기 | 설명 | 관련 코드 |
| :--- | :--- | :--- |
| $f _\phi(\cdot)$ | 베이지안 U-Net 모델 | `BayesianUNet` 클래스 |
| $\theta$ | 메타-파라미터 (사전 분포의 파라미터) | `meta _learner.prior _net.parameters()` |
| $p(\phi \vert \theta)$ | 메타-사전 분포 | `meta _learner.prior _net` |
| $\lambda _i$ | 태스크-특화 변분 파라미터 (사후 분포의 파라미터) | `higher`의 `fmodel.parameters()` |
| $q(\phi \vert \lambda _i)$ | 근사 사후 분포 | `higher`의 `fmodel` |
| $\mathcal{L} _{inner}$ | 내부 루프 손실 (ELBO) | `meta _learner.inner _loop _loss()` |
| $\mathcal{L} _{outer}$ | 외부 루프 손실 (재구성 손실) | `meta _learner.outer _loop _loss()` |
| $\alpha$ | 내부 루프 학습률 | `CONFIG['INNER _LR']` |
| $\eta _{meta}$ | 외부 루프 학습률 | `CONFIG['META _LR']` |
| $\bar{Y} _{pred}, U _{pred}$ | 최종 예측과 불확실성 | `meta _evaluate` 함수의 반환값 `mean _prediction`, `std _prediction` |
