### 1. 로그 수치들의 의미와 수식적 해석

`Epoch 8/50 | KL Weight: 2.87e-07 | Meta-Train Loss: 1.050415 | Val Loss: 40.109211`

#### 가. `Epoch 8/50`
*   **의미**: 총 50번의 메타-훈련 에포크 중 8번째 에포크가 완료되었음을 의미합니다.
*   **수식적 해석**: 메타-파라미터 $\theta$가 8번의 대규모 업데이트를 거쳤음을 의미합니다. 8번째 에포크가 끝난 시점의 메타-파라미터를 $\theta^{(8)}$이라고 할 수 있습니다.

$$
\theta^{(e)} = \text{Update}(\theta^{(e-1)}, \{\mathcal{T} _i\} _{i=1}^{N _{tasks}})
$$

여기서 $e=8$이고, $N _{tasks}$는 `TASKS _PER _EPOCH` (100) 입니다.

#### 나. `KL Weight: 2.87e-07`
*   **의미**: 현재 8번째 에포크의 내부 루프 손실 함수에서 KL Divergence 항에 곱해지는 가중치($\beta _{KL}$)의 값입니다. KL Annealing 기법에 의해 에포크가 진행됨에 따라 이 값이 점차 증가합니다.
*   **수식적 해석**: 내부 루프 손실 함수 $\mathcal{L} _{inner}$에서 사용되는 가중치입니다.

$$
\mathcal{L} _{inner}(\lambda _i) = \text{MSE}(D _i^{(S)}) + \beta _{KL} \cdot \text{KL}(q(\phi|\lambda _i) || p(\phi|\theta^{(7)}))
$$

여기서 $\beta _{KL} = 2.87 \times 10^{-7}$ 입니다. 이 값이 작다는 것은 현재 에포크에서는 모델이 KL 정규화보다는 서포트 셋의 MSE를 줄이는 데 더 집중하고 있음을 의미합니다.

#### 다. `Meta-Train Loss: 1.050415`
*   **의미**: 8번째 에포크 동안 처리한 100개 훈련 태스크들의 **평균 외부 루프 손실(average outer loop loss)**입니다. 이 값은 현재 메타-파라미터 $\theta^{(7)}$를 기반으로 새로운 태스크에 적응했을 때, 얼마나 좋은 일반화 성능을 보이는지에 대한 훈련 데이터 기준의 척도입니다.
*   **수식적 해석**: 8번째 에포크에 사용된 100개의 훈련 태스크 집합을 $\{\mathcal{T} _{i}^{train}\} _{i=1}^{100}$ 라고 할 때, 이 값은 다음과 같이 계산됩니다.

$$
\text{Meta-Train Loss} = \frac{1}{100} \sum _{i=1}^{100} \mathcal{L} _{outer}(\theta^{(7)}, \mathcal{T} _i^{train})
$$

여기서 $\mathcal{L} _{outer}(\theta, \mathcal{T}) = \mathbb{E} _{q(\phi|\lambda)} \left[ \text{MSE}(f _\phi(X^{(Q)}), Y^{(Q)}) \right]$ 이며, $\lambda$는 $\theta$와 $\mathcal{T}$의 서포트 셋 $D^{(S)}$로부터 계산됩니다. 이 값이 낮고 안정적이라는 것은 메타-학습이 훈련 태스크 분포 내에서는 잘 진행되고 있음을 시사합니다.

#### 라. `Val Loss: 40.109211`
*   **의미**: 8번째 에포크가 끝난 후, **훈련 중에 보지 못했던 월(month)의 데이터로 구성된 고정된 검증 태스크(validation task)** 하나에 대해 측정한 외부 루프 손실입니다. 이 값은 모델의 **진정한 일반화 성능**을 나타내는 가장 중요한 지표입니다.
*   **수식적 해석**: 검증 태스크를 $\mathcal{T}  _{val}$이라고 할 때, 이 값은 다음과 같습니다.

$$
\text{Val Loss} = \mathcal{L}  _{outer}(\theta^{(8)}, \mathcal{T}  _{val})
$$

여기서 $\theta^{(8)}$은 8번째 에포크의 훈련이 모두 끝난 후 업데이트된 최종 메타-파라미터입니다. 이 값이 이전 에포크의 `Val Loss`보다 낮아졌기 때문에, 현재 모델의 상태(`meta _learner.state _dict()`)가 `best _bayesian _meta _model.pth` 파일로 저장된 것입니다.
