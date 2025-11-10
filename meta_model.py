# meta_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# -----------------------------------------------------------------------------
# 1. 베이지안 신경망을 위한 기본 레이어 정의
# -----------------------------------------------------------------------------

class BayesianLayer(nn.Module):
    """
    모든 베이지안 레이어의 부모 클래스.
    가중치의 평균(mu)과 분산(rho)을 파라미터로 가지며,
    Reparameterization Trick을 이용한 샘플링과 KL Divergence 계산을 담당.
    """
    def __init__(self):
        super().__init__()
        self.mu = None
        self.rho = None
    
    def sample(self):
        """Reparameterization Trick을 사용하여 가중치를 샘플링합니다."""
        # sigma = log(1 + exp(rho)) -> 항상 양수 보장
        sigma = F.softplus(self.rho)
        epsilon = torch.randn_like(sigma)
        return self.mu + sigma * epsilon

    def kl_divergence(self, prior_mu, prior_sigma):
        """
        현재 레이어의 사후분포 q(w|lambda)와 사전분포 p(w|theta) 사이의
        KL Divergence를 계산합니다. (두 분포 모두 대각 가우시안 가정)
        KL(q || p) = log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q-mu_p)^2)/(2*sigma_p^2) - 0.5
        """
        posterior_sigma = F.softplus(self.rho)
        
        kl = (torch.log(prior_sigma / posterior_sigma) +
              (posterior_sigma.pow(2) + (self.mu - prior_mu).pow(2)) / (2 * prior_sigma.pow(2)) - 0.5)
        
        return kl.sum()

class BayesianConv2d(BayesianLayer):
    """베이지안 2D Convolutional 레이어"""
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.kwargs = kwargs

        # 가중치(weight)와 편향(bias)의 평균(mu)과 분산 제어(rho) 파라미터
        self.mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.rho = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mu, a=0.01)
        nn.init.normal_(self.rho, mean=-3, std=0.1) # 초기 분산을 작게 설정
        nn.init.zeros_(self.bias_mu)
        nn.init.normal_(self.bias_rho, mean=-3, std=0.1)

    def forward(self, x, sample=True):
        if sample:
            weight = self.sample()
            bias_sigma = F.softplus(self.bias_rho)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else: # 샘플링 없이 평균 가중치만 사용 (평가 시)
            weight = self.mu
            bias = self.bias_mu
            
        return F.conv2d(x, weight, bias, **self.kwargs)

    def kl_divergence(self, prior_layer):
        """가중치와 편향의 KL Divergence를 합산하여 반환합니다."""
        weight_kl = super().kl_divergence(prior_layer.mu, F.softplus(prior_layer.rho))
        
        bias_posterior_sigma = F.softplus(self.bias_rho)
        bias_prior_sigma = F.softplus(prior_layer.bias_rho)
        bias_kl = (torch.log(bias_prior_sigma / bias_posterior_sigma) +
                   (bias_posterior_sigma.pow(2) + (self.bias_mu - prior_layer.bias_mu).pow(2)) / (2 * bias_prior_sigma.pow(2)) - 0.5)
        
        return weight_kl + bias_kl.sum()

class BayesianConvTranspose2d(BayesianLayer):
    """베이지안 2D Transposed Convolutional 레이어"""
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.kwargs = kwargs

        self.mu = nn.Parameter(torch.Tensor(in_channels, out_channels, *self.kernel_size))
        self.rho = nn.Parameter(torch.Tensor(in_channels, out_channels, *self.kernel_size))
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mu, a=0.01)
        nn.init.normal_(self.rho, mean=-3, std=0.1)
        nn.init.zeros_(self.bias_mu)
        nn.init.normal_(self.bias_rho, mean=-3, std=0.1)

    def forward(self, x, sample=True):
        if sample:
            weight = self.sample()
            bias_sigma = F.softplus(self.bias_rho)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            weight = self.mu
            bias = self.bias_mu
            
        return F.conv_transpose2d(x, weight, bias, **self.kwargs)

    def kl_divergence(self, prior_layer):
        weight_kl = super().kl_divergence(prior_layer.mu, F.softplus(prior_layer.rho))
        
        bias_posterior_sigma = F.softplus(self.bias_rho)
        bias_prior_sigma = F.softplus(prior_layer.bias_rho)
        bias_kl = (torch.log(bias_prior_sigma / bias_posterior_sigma) +
                   (bias_posterior_sigma.pow(2) + (self.bias_mu - prior_layer.bias_mu).pow(2)) / (2 * bias_prior_sigma.pow(2)) - 0.5)
        
        return weight_kl + bias_kl.sum()

# -----------------------------------------------------------------------------
# 2. U-Net 구성 요소를 베이지안 버전으로 재정의
# -----------------------------------------------------------------------------

class BayesianDoubleConv(nn.Module):
    """(Bayesian convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = BayesianConv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = BayesianConv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, sample=True):
        x = self.conv1(x, sample)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x, sample)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class BayesianDown(nn.Module):
    """Downscaling with maxpool then Bayesian double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.bdouble_conv = BayesianDoubleConv(in_channels, out_channels)

    def forward(self, x, sample=True):
        x = self.maxpool(x)
        x = self.bdouble_conv(x, sample)
        return x

class BayesianUp(nn.Module):
    """Upscaling then Bayesian double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.bdouble_conv = BayesianDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = BayesianConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.bdouble_conv = BayesianDoubleConv(in_channels, out_channels)
        self.bilinear = bilinear

    def forward(self, x1, x2, sample=True):
        if self.bilinear:
            x1 = self.up(x1)
        else:
            x1 = self.up(x1, sample)
            
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.bdouble_conv(x, sample)

class BayesianOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = BayesianConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, sample=True):
        return self.conv(x, sample)

# -----------------------------------------------------------------------------
# 3. 완전한 베이지안 U-Net 모델 정의
# -----------------------------------------------------------------------------

class BayesianUNet(nn.Module):
    def __init__(self, n_channels, num_future_steps, bilinear=True):
        super(BayesianUNet, self).__init__()
        self.inc = BayesianDoubleConv(n_channels, 64)
        self.down1 = BayesianDown(64, 128)
        self.down2 = BayesianDown(128, 256)
        self.down3 = BayesianDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = BayesianDown(512, 1024 // factor)
        self.up1 = BayesianUp(1024, 512 // factor, bilinear)
        self.up2 = BayesianUp(512, 256 // factor, bilinear)
        self.up3 = BayesianUp(256, 128 // factor, bilinear)
        self.up4 = BayesianUp(128, 64, bilinear)
        self.outc = BayesianOutConv(64, num_future_steps)

    def forward(self, x, sample=True):
        x1 = self.inc(x, sample)
        x2 = self.down1(x1, sample)
        x3 = self.down2(x2, sample)
        x4 = self.down3(x3, sample)
        x5 = self.down4(x4, sample)
        x = self.up1(x5, x4, sample)
        x = self.up2(x, x3, sample)
        x = self.up3(x, x2, sample)
        x = self.up4(x, x1, sample)
        return self.outc(x, sample)

    def kl_divergence(self, prior_model):
        """모델 전체의 모든 베이지안 레이어에 대한 KL Divergence의 총합을 계산합니다."""
        kl_total = 0.0
        # self.modules()는 모델의 모든 레이어를 순회합니다.
        for module, prior_module in zip(self.modules(), prior_model.modules()):
            if isinstance(module, BayesianLayer):
                kl_total += module.kl_divergence(prior_module)
        return kl_total

# -----------------------------------------------------------------------------
# 4. 메타-러너 클래스 정의
# -----------------------------------------------------------------------------

class MetaLearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 이 모델의 파라미터가 메타-학습의 대상인 '사전 분포(prior)'의 파라미터(theta)가 됩니다.
        self.prior_net = BayesianUNet(config['INPUT_LEN'], config['TARGET_LEN'])
        
    def forward(self, x, sample=True, num_samples=1):
        """
        평가 시 불확실성 측정을 위해 여러 번 샘플링하여 예측합니다.
        """
        if not sample or num_samples == 1:
            return self.prior_net(x, sample)
        
        # 여러 개의 예측 샘플을 생성
        predictions = [self.prior_net(x, sample=True) for _ in range(num_samples)]
        predictions = torch.stack(predictions) # (num_samples, batch, C, H, W)
        
        # 예측의 평균과 표준편차(불확실성) 반환
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        return mean_pred, std_pred

    def inner_loop_loss(self, fmodel, support_x, support_y):
        """
        내부 루프에서 사용될 손실 함수를 정의합니다.
        ELBO를 최대화하기 위해 (재구성 손실 + KL Divergence)를 최소화합니다.
        """
        # 1. 재구성 손실 (MSE)
        outputs = fmodel(support_x, sample=True)
        mse_loss = F.mse_loss(outputs, support_y)
        
        # 2. KL Divergence (정규화)
        # fmodel은 적응된 사후분포(q), self.prior_net은 사전분포(p)
        kl_loss = fmodel.kl_divergence(self.prior_net)
        
        # 3. 최종 손실
        # KL 가중치는 하이퍼파라미터로, 두 손실 간의 균형을 맞춥니다.
        total_loss = mse_loss + self.config['KL_WEIGHT'] * kl_loss
        return total_loss

    def outer_loop_loss(self, fmodel, query_x, query_y):
        """
        외부 루프에서 사용될 손실 함수를 정의합니다.
        적응된 모델의 일반화 성능을 쿼리 셋으로 측정합니다.
        """
        # 외부 루프에서는 재구성 손실(MSE)만 사용하여 성능을 평가합니다.
        # KL Divergence의 효과는 내부 루프를 통해 그래디언트에 이미 반영됩니다.
        outputs = fmodel(query_x, sample=True)
        outer_loss = F.mse_loss(outputs, query_y)
        return outer_loss