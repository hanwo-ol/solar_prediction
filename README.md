### 1. 수학적 정의 (Mathematical Formulation)

#### 가. 기본 표기법 (Notations)

*   **모델**: 직접 다중 스텝 출력이 가능한 U-Net 모델을 $f_\phi(\cdot)$로 표기합니다. $\phi \in \mathbb{R}^D$는 모델의 모든 학습 가능한 가중치(weights)입니다.
*   **입력 및 출력**:
    *   입력 시퀀스: $X_t = (I_{t-k+1}, \dots, I_t)$는 시간 $t$까지의 과거 $k$개(`INPUT_LEN`)의 위성 이미지 시퀀스입니다.
    *   타겟 시퀀스: $Y_t = (I_{t+1}, \dots, I_{t+L})$는 시간 $t$ 이후의 미래 $L$개(`TARGET_LEN`)의 위성 이미지 시퀀스입니다.
*   **태스크 (Task)**: 태스크 $\mathcal{T}_i$는 특정 시공간적 맥락(예: 2022년 봄, 한반도 남해안 지역)에서의 소규모 예측 문제를 의미합니다. 각 태스크는 고유한 데이터 분포 $p_i(X, Y)$를 따릅니다.
*   **태스크 데이터**: 각 태스크 $\mathcal{T}_i$는 훈련(적응)을 위한 **서포트 셋(Support Set)** $D_i^{(S)}$과 메타-학습 평가를 위한 **쿼리 셋(Query Set)** $D_i^{(Q)}$으로 구성됩니다.
    *   $D_i^{(S)} = \{(X_j^{(S)}, Y_j^{(S)})\}_{j=1}^{N_S}$
    *   $D_i^{(Q)} = \{(X_j^{(Q)}, Y_j^{(Q)})\}_{j=1}^{N_Q}$

#### 나. 계층적 베이지안 모델 (Hierarchical Bayesian Model)

우리는 모델 가중치 $\phi$를 단일 값(point estimate)이 아닌 확률 분포로 다룹니다.

*   **메타-사전 분포 (Meta-Prior)**: 모든 태스크에 걸쳐 공유되는 가중치에 대한 사전 지식입니다. 이 분포는 메타-파라미터 $\theta$에 의해 결정되며, 우리는 이 $\theta$를 학습하는 것이 목표입니다. 완전 인수분해 가우시안(fully factorized Gaussian)을 가정합니다:

$$
p(\phi|\theta) = \mathcal{N}(\phi | \mu_\theta, \text{diag}(\sigma^2_\theta))
$$

여기서 $\theta = (\mu_\theta, \sigma_\theta)$는 모든 가중치의 평균과 표준편차 벡터입니다.

*   **태스크-특화 사후 분포 (Task-Specific Posterior)**: 태스크 $\mathcal{T}_i$의 서포트 셋 $D_i^{(S)}$를 관찰한 후, 베이즈 정리를 통해 얻는 가중치의 사후 분포입니다. 계산의 편의를 위해 변분 추론(Variational Inference)을 사용하여 근사 사후 분포 $q(\phi|D_i^{(S)})$를 구합니다.

$$
q(\phi|D_i^{(S)}) \approx p(\phi|D_i^{(S)}, \theta) \propto p(D_i^{(S)}|\phi) p(\phi|\theta)
$$

이 근사 사후 분포 역시 가우시안 $q(\phi|\lambda_i) = \mathcal{N}(\phi | \mu_{\lambda_i}, \text{diag}(\sigma^2_{\lambda_i}))$로 가정하며, $\lambda_i = (\mu_{\lambda_i}, \sigma_{\lambda_i})$는 태스크-특화 변분 파라미터입니다.

#### 다. 최적화 목표 (Optimization Objective)

메타-학습의 목표는 모든 태스크의 쿼리 셋에 대한 로그 가능도(log-likelihood)의 기댓값을 최대화하는 메타-파라미터 $\theta$를 찾는 것입니다. 이는 증거 하한(Evidence Lower Bound, ELBO)을 최대화하는 것과 같습니다.

$$
\theta^* = \arg\max_{\theta} \mathbb{E}_{\mathcal{T}_i \sim p(\mathcal{T})} \left[ \mathcal{L}_i(\theta) \right]
$$

여기서 단일 태스크 $\mathcal{T}_i$에 대한 손실(ELBO) $\mathcal{L}_i(\theta)$는 다음과 같이 정의됩니다:

$$
\mathcal{L}_i(\theta) = \mathbb{E}_{q(\phi|\lambda_i)} \left[ \log p(D_i^{(Q)}|\phi) \right] - \text{KL} \left( q(\phi|\lambda_i) || p(\phi|\theta) \right)
$$

*   **첫 번째 항 (재구성 손실)**: 서포트 셋으로 적응된 모델($q(\phi|\lambda_i)$)이 쿼리 셋을 얼마나 잘 예측하는지를 측정합니다. $\log p(D_i^{(Q)}|\phi)$는 MSE 손실을 사용할 경우, $-\frac{1}{2\sigma_{noise}^2} \sum_{(X,Y) \in D_i^{(Q)}} ||Y - f_\phi(X)||^2_F$ 에 비례합니다. (Frobenius norm)
*   **두 번째 항 (KL 발산)**: 태스크-특화 사후 분포($q$)가 메타-사전 분포($p$)에서 너무 멀어지지 않도록 규제하여 과적합을 방지하고 일반화를 촉진합니다.

#### 라. 상각 변분 추론 (Amortized Variational Inference)

각 태스크마다 $\lambda_i$를 최적화하는 것은 비용이 크므로, **상각(amortization)** 기법을 사용합니다. 즉, $\lambda_i$를 메타-파라미터 $\theta$에서 시작하여 서포트 셋 $D_i^{(S)}$에 대한 손실 함수를 몇 스텝 경사 하강하여 근사적으로 구합니다.

1.  **초기화**: 변분 파라미터를 메타-파라미터로 초기화합니다. $\lambda_i^{(0)} = \theta$.
2.  **내부 루프 (Inner Loop Adaptation)**: $m=0, \dots, M-1$에 대해 $M$번 업데이트합니다.

$$
\lambda_i^{(m+1)} = \lambda_i^{(m)} - \alpha \nabla_{\lambda_i^{(m)}} \left[ \text{KL}(q(\phi|\lambda_i^{(m)}) || p(\phi|\theta)) - \mathbb{E}_{q(\phi|\lambda_i^{(m)})} [\log p(D_i^{(S)}|\phi)] \right]
$$

최종적으로 얻은 $\lambda_i = \lambda_i^{(M)}$가 태스크-특화 사후 분포의 파라미터가 됩니다.
3.  **외부 루프 (Outer Loop Meta-Update)**: 배치 내의 모든 태스크에 대해 계산된 $\lambda_i$를 사용하여 전체 손실 $\mathcal{L}_{meta} = -\sum_i \mathcal{L}_i(\theta)$를 계산하고, 이를 통해 메타-파라미터 $\theta$를 업데이트합니다.

$$
\theta \leftarrow \theta - \beta \nabla_\theta \mathcal{L}_{meta}
$$

이 과정은 $\nabla_\theta$가 내부 루프의 경사 하강 과정 전체를 통과해야 하므로, 고차 미분(higher-order differentiation)이 필요합니다.


---

참조

@inproceedings{ravi2019amortized,
  title={Amortized bayesian meta-learning},
  author={Ravi, Sachin and Beatson, Alex},
  booktitle={International Conference on Learning Representations}
}
