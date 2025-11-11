`Meta-Training: 4%|███▊ | 4/100 [00:21<08:16, 5.17s/it]`

이 로그는 한 메타-훈련 에포크가 **100개의 태스크(Task)로 구성**되어 있으며(`TASKS_PER_EPOCH=100`), 현재 그중 **4번째 태스크를 처리**하고 있음을 의미합니다.

한 에포크의 목표는, 100개의 서로 다른 미니 예측 문제(태스크)를 풀어보는 경험을 통해, 모델의 **보편적인 사전 지식(meta-prior, $p(\phi|\theta)$)**을 아주 조금 개선하는 것입니다.

아래에서는 **단일 태스크(e.g., 4번째 태스크)를 처리하는 과정**을 수식과 함께 단계별로 설명하겠습니다. 이 과정이 `meta_train_one_epoch` 함수의 `for` 루프 내부에서 한 번 반복될 때 일어나는 일입니다.

---

### 단일 태스크 $\mathcal{T}_i$ 처리 과정 (1 Iteration)

#### **Step 1: 태스크 샘플링 (Task Sampling)**

`MetaSolarPredictionDataset`이 데이터셋에서 무작위로 데이터를 샘플링하여 하나의 태스크 $\mathcal{T}_i$를 생성합니다.

*   **수식:**
    *   태스크 샘플링: $\mathcal{T}_i \sim p(\mathcal{T})$
    *   데이터 분할: 태스크 $\mathcal{T}_i$는 서포트 셋 $D_i^{(S)}$와 쿼리 셋 $D_i^{(Q)}$으로 구성됩니다.
        *   $D_i^{(S)} = \{(X_j^{(S)}, Y_j^{(S)})\}_{j=1}^{N_S}$ (여기서 $N_S$는 `K_SHOT`)
        *   $D_i^{(Q)} = \{(X_j^{(Q)}, Y_j^{(Q)})\}_{j=1}^{N_Q}$ (여기서 $N_Q$는 `K_QUERY`)

*   **코드:** `meta_train_loader`가 이 태스크 하나를 `task_batch`로 반환합니다.

---

#### **Step 2: 내부 루프 - 태스크 적응 (Inner Loop - Task Adaptation)**

현재의 메타-사전 분포 $p(\phi|\theta)$를 기반으로, 이 태스크 $\mathcal{T}_i$에 특화된 **근사 사후 분포 $q(\phi|\lambda_i)$를 찾습니다.** 이 과정은 `higher` 라이브러리를 통해 미분 가능한 형태로 진행됩니다.

1.  **모델 복제**: 현재 메타-러너의 `prior_net`($\theta$를 파라미터로 가짐)을 미분 가능하게 복제하여 임시 모델 `fmodel`을 만듭니다. 이 `fmodel`의 파라미터가 바로 태스크-특화 사후 분포의 파라미터 $\lambda_i$가 됩니다.
    *   **초기화**: $\lambda_i^{(0)} = \theta$

2.  **경사 하강**: `INNER_STEPS` 만큼 경사 하강을 반복하며 $\lambda_i$를 업데이트합니다.
    *   **수식 (m번째 스텝):**

$$
\lambda_i^{(m+1)} = \lambda_i^{(m)} - \alpha \nabla_{\lambda_i^{(m)}} \mathcal{L}_{inner}(\lambda_i^{(m)})
$$

*   $\alpha$: 내부 학습률 `INNER_LR`
*   $\mathcal{L}_{inner}$: 내부 루프 손실 함수. 서포트 셋 $D_i^{(S)}$을 사용하여 계산됩니다.

$$
\mathcal{L}_{inner}(\lambda_i) = \underbrace{\mathbb{E}_{q(\phi|\lambda_i)} \left[ \text{MSE}(f_\phi(X^{(S)}), Y^{(S)}) \right]}_{\text{재구성 손실 (MSE)}} + \beta_{KL} \cdot \underbrace{\text{KL} \left( q(\phi|\lambda_i) || p(\phi|\theta) \right)}_{\text{KL Divergence 정규화}}
$$

*   $\beta_{KL}$: `KL_WEIGHT`

*   **코드:** `with higher.innerloop_ctx(...)` 블록 내부의 `for` 루프가 이 과정을 수행합니다. `diffopt.step(inner_loss)`가 $\lambda_i$를 업데이트합니다.

이 과정이 끝나면, `fmodel`은 서포트 셋 $D_i^{(S)}$의 정보에 적응하여 태스크 $\mathcal{T}_i$를 잘 풀도록 특화된 모델이 됩니다.

---

#### **Step 3: 외부 루프 - 메타 손실 계산 (Outer Loop - Meta Loss Calculation)**

내부 루프에서 적응된 `fmodel`이 얼마나 **일반화**되었는지를 **쿼리 셋 $D_i^{(Q)}$**을 사용하여 평가합니다. 이 평가 결과가 바로 메타-파라미터 $\theta$를 업데이트하기 위한 손실이 됩니다.

*   **수식:**

$$
\mathcal{L}_{outer}(\theta, \mathcal{T}_i) = \mathbb{E}_{q(\phi|\lambda_i^{(M)})} \left[ \text{MSE}(f_\phi(X^{(Q)}), Y^{(Q)}) \right]
$$

*   $\lambda_i^{(M)}$: 내부 루프를 $M$ (`INNER_STEPS`)번 반복한 후의 최종 파라미터.
*   **핵심**: $\lambda_i^{(M)}$은 초기값 $\theta$에서부터 시작하여 계산되었으므로, $\mathcal{L}_{outer}$는 **원래의 메타-파라미터 $\theta$에 대한 함수**입니다.

*   **코드:** `outer_loss = meta_learner.outer_loop_loss(fmodel, query_x, query_y)`가 이 손실을 계산합니다.

---

#### **Step 4: 메타-업데이트 (Meta-Update)**

외부 루프에서 계산된 메타 손실 $\mathcal{L}_{outer}$를 사용하여 **메타-파라미터 $\theta$를 업데이트**합니다.

1.  **메타-그래디언트 계산**: $\mathcal{L}_{outer}$를 $\theta$에 대해 미분합니다. 이 그래디언트는 Step 3의 손실 계산뿐만 아니라, Step 2의 내부 루프 적응 과정 전체를 거슬러 올라가며 계산됩니다.
    *   **수식:**

$$
g_\theta = \nabla_\theta \mathcal{L}_{outer}(\theta, \mathcal{T}_i)
$$

2.  **메타-파라미터 업데이트**: 계산된 그래디언트를 사용하여 `meta_optimizer`가 $\theta$를 업데이트합니다.
    *   **수식:**

$$
\theta \leftarrow \theta - \eta \cdot g_\theta
$$

*   $\eta$: 메타 학습률 `META_LR`

*   **코드:** `outer_loss.backward()`가 $g_\theta$를 계산하고, `meta_optimizer.step()`이 $\theta$를 업데이트합니다. `torch.nn.utils.clip_grad_norm_`는 $g_\theta$가 너무 커지는 것을 방지합니다.

---

### 한 에포크의 전체 과정 요약

1.  **초기화**: 메타-파라미터 $\theta$를 가진 `meta_learner`가 있습니다.
2.  **반복 (100번)**: `TASKS_PER_EPOCH` 만큼 다음을 반복합니다.
    *   **a. 태스크 샘플링**: `meta_train_loader`에서 새로운 태스크 $\mathcal{T}_i$를 가져옵니다.
    *   **b. 가상 업데이트 (내부 루프)**: 현재 $\theta$를 이용해 $\mathcal{T}_i$의 서포트 셋에 빠르게 적응하는 가상의 모델(`fmodel`)을 만듭니다.
    *   **c. 일반화 성능 평가 (외부 루프)**: 이 `fmodel`을 $\mathcal{T}_i$의 쿼리 셋으로 평가하여 메타 손실 $\mathcal{L}_{outer}$를 계산합니다.
    *   **d. 메타-업데이트**: 메타 손실을 기반으로 **원본 `meta_learner`의 파라미터 $\theta$를 직접 업데이트**합니다.
3.  **에포크 종료**: 100개의 태스크에 대한 경험을 통해 $\theta$가 조금 더 개선되었습니다.

이 과정을 비유하자면, 숙련된 교사($\theta$)가 100명의 다른 학생에게 각각 다른 연습 문제(서포트 셋)를 주고 스스로 공부하게 한 뒤(**내부 루프**), 실전 시험(쿼리 셋)을 보게 합니다. 그리고 그 시험 결과를 바탕으로 자신의 **근본적인 교수법($\theta$)**을 조금씩 수정하여, 다음 에포크에서는 학생들이 더 빠르고 효과적으로 학습할 수 있도록 만드는 과정과 같습니다.
