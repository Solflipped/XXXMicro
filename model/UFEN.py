import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetBinaryClassifier
from model.FT_transformer import NumericalFeatureTokenizer

class residual_block(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=1, padding=kernel_size//2, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class attention_gate(nn.Module):
    """注意力门"""
    def __init__(self, F_g, F_l, F_int):
        super(attention_gate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(8, F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(8, F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, g, x):
        # g: 解码器深层特征 (门控信号)
        # x: 编码器浅层特征 (跳连信号)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 确保尺寸匹配
        if g1.shape[2] != x1.shape[2]:
            g1 = F.interpolate(g1, size=x1.shape[2], mode='nearest')
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # 扩展注意力权重到与x相同的通道数
        psi = psi.expand_as(x)
        
        return x * psi   
  

class Bridge(nn.Module):
    """
    在编码器输出端预测均值(mu)和标准差(sigma)
    通过随机采样增强模型对微生物数据噪声的稳健性 
    """
    def __init__(self, in_channels, latent_dim):
        super(Bridge, self).__init__()
        self.fc_mu = nn.Conv1d(in_channels, latent_dim, 1)
        self.fc_var = nn.Conv1d(in_channels, latent_dim, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0) 
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu  # 推理时用确定性输出
        return z, mu, logvar


class UnimodalPredictor(nn.Module):
    def __init__(self, d_token, n_features, num_scales):
        super(UnimodalPredictor, self).__init__()
        # 1. 通道压缩：将 d_token 个维度的残差信息压缩，提取核心“异常扰动”
        # [batch_size, d_token, n_num_features] -> [batch_size, 1, n_num_features]
        self.feature_reduction = nn.Sequential(
            nn.Conv1d(d_token, 8, kernel_size=1),
            nn.GroupNorm(4, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(8, 1, kernel_size=1)
        )
        # 2. 重构评分融合层
        self.score_fusion = nn.Linear(num_scales, 16)
        # 3. 全局分类层
        self.global_fusion = nn.Sequential(
            nn.Linear(n_features + 16, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, final_residual, scale_errors):
        # 提取关键物种的扰动特征
        x_res = self.feature_reduction(final_residual).flatten(1)
        # 整合层级重构评分
        x_score = F.leaky_relu(self.score_fusion(scale_errors))
        return self.global_fusion(torch.cat([x_res, x_score], dim=1))
    


class UFEN(nn.Module):
    """
    单模态特征提取网络 (Unimodal Feature Extraction Network)，用于数据的特征增强
    """
    def __init__(self, 
                 n_num_features: int,      # 原始特征维度
                 d_token: int = 32,        # 输入通道数
                 base_channels: int = 32,  # 基础通道数
                 num_layers=2,             # 编码器/解码器层数
                 latent_dim=128):          # 潜在表示维度
        super(UFEN, self).__init__()

        self.num_layers = num_layers
        self.tokenizer = NumericalFeatureTokenizer(n_features=n_num_features, d_token=d_token, bias=True,initialization='uniform')
        self.channels = [base_channels * (2**i) for i in range(num_layers)]


        # 编码器
        self.encoders = nn.ModuleList()
        in_ch = d_token
        for ch in self.channels:
            downsample = nn.Sequential(nn.Conv1d(in_ch, ch, 1), nn.GroupNorm(8, ch))
            self.encoders.append(residual_block(in_ch, ch, downsample=downsample))
            in_ch = ch
            
        # 桥接
        self.bridge = Bridge(self.channels[-1], latent_dim)
        
        # 解码器与重构评分器
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.side_reconstructors = nn.ModuleList() 
        
        curr_ch = latent_dim
        for i in range(num_layers-1, -1, -1):
            self.attention_gates.append(attention_gate(curr_ch, self.channels[i], self.channels[i]//2))
            dec_in = curr_ch + self.channels[i]
            downsample = nn.Sequential(nn.Conv1d(dec_in, self.channels[i], 1), nn.GroupNorm(8, self.channels[i]))
            self.decoders.append(residual_block(dec_in, self.channels[i], downsample=downsample))
            # 每一层级重构器 (用于生成 EnsDeepDP 的疾病评分)
            self.side_reconstructors.append(nn.Conv1d(self.channels[i], d_token, 1))
            curr_ch = self.channels[i]

        self.final_op = nn.Conv1d(self.channels[0], d_token, 1)
        self.predictor = UnimodalPredictor(d_token, n_num_features, num_layers)

    def forward(self, raw_x):
        # 1. Tokenize 并转置适配 Conv1d
        x_tokens = self.tokenizer(raw_x).transpose(1, 2) # [B, d_token, N]
        
        # 2. 编码器路径
        en_outs = []
        curr = x_tokens
        for enc in self.encoders:
            curr = enc(curr)
            en_outs.append(curr)
            
        # 3. 潜在空间重采样 (桥接逻辑)
        latent, mu, logvar = self.bridge(curr)
        
        # 4. 解码器路径与多尺度误差计算
        scale_errors = []
        curr = latent
        for i in range(self.num_layers):
            en_idx = self.num_layers - 1 - i
            if curr.shape[2] != en_outs[en_idx].shape[2]:
                curr = F.interpolate(curr, size=en_outs[en_idx].shape[2], mode='nearest')
            
            # 注意力融合
            attended = self.attention_gates[i](curr, en_outs[en_idx])
            curr = self.decoders[i](torch.cat([curr, attended], dim=1))
            
            # 提取该层重构误差 
            side_out = self.side_reconstructors[i](curr)
            if side_out.shape[2] != x_tokens.shape[2]:
                x_target = F.interpolate(x_tokens, size=side_out.shape[2], mode='nearest')
            else:
                x_target = x_tokens
            err = F.mse_loss(side_out, x_target, reduction='none').mean(dim=[1, 2])
            scale_errors.append(err)
            
        # 5. 最终残差计算
        feat_rec = self.final_op(curr)
        if feat_rec.shape[2] != x_tokens.shape[2]:
            feat_rec = F.interpolate(feat_rec, size=x_tokens.shape[2], mode='nearest')
        final_residual = x_tokens - feat_rec
        
        # 6. 集成预测
        scale_errors = torch.stack(scale_errors, dim=1)
        logits = self.predictor(final_residual, scale_errors)
        # 形状转置为：[batch_size, n_num_features, d_token]  后续模块使用
        out_seq = final_residual.transpose(1, 2)
        
        return logits, mu, logvar, out_seq

    @classmethod
    def make_default(
        cls, # 类本身（UFEN）
        n_num_features: int,      # 原始特征维度
        d_token: int = 64,        # 输入通道数
        base_channels: int = 64,  # 基础通道数
        num_layers=2,             # 编码器/解码器层数
        latent_dim=256,           # 潜在表示维度
        **kwargs
    ) -> 'UFEN':
        """
        参数:
        - n_num_features: 输入特征维度
        - d_token: token 输入通道数
        - base_channels: 编码器/解码器的基础通道数，实际通道数会随着层数成倍增加
        - num_layers: 编码器/解码器的层数
        - latent_dim: 潜在空间的维度大小
        返回:
        - UFEN 模型实例
        """
        return cls(
            n_num_features=n_num_features,
            d_token=d_token,
            base_channels=base_channels,
            num_layers=num_layers,
            latent_dim=latent_dim,
            **kwargs
        )


class UFENNet(NeuralNetBinaryClassifier):
    def __init__(self, *args, beta=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    # 1. 重写 get_loss 方法，计算分类损失 + KLD 损失  (损失计算器)
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        # 解包 UFEN 的返回结果
        logits, mu, logvar, _ = y_pred
        y_true = y_true.float().view(-1)
        
        # 1. 分类损失 (BCE)
        loss_bce = super().get_loss(logits, y_true, *args, **kwargs)
        
        # 2. KLD 损失 (VAE 正则化)
        # 强制潜在分布趋向标准正态分布 
        logvar_for_kld = torch.clamp(logvar, min=-10.0)
        kld = -0.5 * torch.sum(1 + logvar_for_kld - mu.pow(2) - logvar_for_kld.exp(), dim=-1)
        loss_kld = kld.mean()

        # 3. 训练初期更关注分类损失，逐渐增加 KLD 的权重
        # epoch = len(self.history) if hasattr(self, 'history') and self.history else 0
        # current_beta = min(1.0, (epoch + 1) / 50) * self.beta
        total_loss = loss_bce +  self.beta * loss_kld
    
        return loss_bce if torch.isnan(total_loss) else total_loss

    # 2. 重写 predict_proba 方法，确保返回 [N, 2] 的概率分布 (预测概率)， 对外接口
    def predict_proba(self, X):
        non_probas = []
        for yp in self.forward_iter(X, training=False):
            # 动态兼容不同阶段的返回类型，彻底消灭 KeyError 和 TypeError
            logits = yp[0] if isinstance(yp, tuple) else yp.get('y_pred', list(yp.values())[0])
            
            # 确保 logits 只有一维，并计算类别 1 的概率 (sigmoid)
            p1 = torch.sigmoid(logits).view(-1, 1)
            non_probas.append(p1)
        
        # 合并所有的 batch 预测结果, 得到一个 (N, 1) 的概率向量
        p1_all = torch.cat(non_probas, dim=0).cpu().numpy()
        # 构造 Skorch 期望的 [N, 2] 结构
        # 第一列是类别 0 的概率 (1 - p1)，第二列是类别 1 的概率 (p1)
        p0_all = 1 - p1_all
        return np.hstack([p0_all, p1_all])
    
    # 3. 重写 evaluation_step 方法，确保评估时也能正确处理三个返回值
    def evaluation_step(self, batch, training=False):
        # 确保评估时也能正确处理三个返回值
        X, y = batch
        with torch.set_grad_enabled(training):
            yp = self.infer(X) # 这里 yp 是 (logits, mu, logvar)
            loss = self.get_loss(yp, y)
            return {'loss': loss, 'y_pred': yp[0]}