import torch
from torch import nn, einsum
import torch.nn.functional as F
from collections import OrderedDict
from model.FT_transformer import NumericalFeatureTokenizer, CLSToken
from einops import rearrange, repeat

class residual_block(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
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
            nn.BatchNorm1d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
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
            g1 = F.interpolate(g1, size=x1.shape[2], mode='linear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # 扩展注意力权重到与x相同的通道数
        psi = psi.expand_as(x)
        
        return x * psi   
  
class UnimodalPredictor(nn.Module):
    def __init__(self, d_token, n_features, num_scales):
        super(UnimodalPredictor, self).__init__()
        
        # 1. 通道压缩：将 d_token 个维度的残差信息压缩，提取核心“异常扰动”
        # [batch_size, d_token, n_num_features] -> [batch_size, 16, n_num_features]
        self.feature_reduction = nn.Sequential(
            nn.Conv1d(d_token, 8, kernel_size=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. 特征映射：将压缩后的残差映射到单一的重要性分数图
        # [batch_size, 16, n_num_features] -> [batch_size, 1, n_num_features]
        self.importance_map = nn.Sequential(
            nn.Conv1d(8, 1, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 3. 全局关联融合：捕捉物种/基因之间的长程依赖
        self.global_fusion = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, residual):
        # 输入 residual 形状为 [B, d_token, N]
        x = self.feature_reduction(residual)
        x = self.importance_map(x)
        logits = self.global_fusion(x)
        return logits

class UFEN(nn.Module):
    """
    单模态特征提取网络 (Unimodal Feature Extraction Network)，用于数据的特征增强
    """
    def __init__(self, 
                 n_num_features: int,      # 原始特征维度
                 d_token: int = 64,        # 输入通道数
                 base_channels: int = 64,  # 基础通道数
                 expansion_factor=2,       # 每次扩展的倍数
                 num_layers=3,             # 编码器/解码器层数
                 latent_dim=256):          # 潜在表示维度
        super(UFEN, self).__init__()

        self.d_token = d_token
        self.n_num_features = n_num_features
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
        self.latent_dim = latent_dim

        # 使用FT-Transformer的NumericalFeatureTokenizer将数值特征转换为token序列
        self.tokenizer = NumericalFeatureTokenizer(
            n_features=n_num_features,
            d_token=d_token,
            bias=True,
            initialization='uniform'
        )
        
        # 计算各层通道数  
        self.channels = [base_channels * (expansion_factor ** i) 
                        for i in range(num_layers)]
        
        # ========== 编码器 ==========
        self.encoder_layers = nn.ModuleList()
        # 第一层：
        downsample_first = None
        if d_token != self.channels[0]:
            downsample_first = nn.Sequential(
                nn.Conv1d(d_token, self.channels[0], kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(self.channels[0])
            )
        self.encoder_layers.append( 
            residual_block(d_token, self.channels[0], downsample=downsample_first)
        )
    
        # 中间层
        for i in range(1, num_layers):
            downsample = nn.Sequential(
                nn.Conv1d(self.channels[i-1], self.channels[i], kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(self.channels[i])
            )
            block = residual_block(self.channels[i-1], self.channels[i], downsample=downsample)
            self.encoder_layers.append(block)
        
        # ========== 桥接层 ==========
        self.bridge = nn.Sequential(
            nn.Conv1d(self.channels[-1], self.latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # ========== 注意力门 ==========
        self.attention_gates = nn.ModuleList()
        for i in range(num_layers-1, -1, -1):
            self.attention_gates.append(
                attention_gate(
                    F_g=self.latent_dim if i == num_layers - 1 else self.channels[i+1],
                    F_l=self.channels[i],
                    F_int=self.channels[i] // 2
                )
            )
        
        # ========== 解码器 ==========
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers-1, -1, -1):
            # 计算输入通道数
            if i == num_layers-1:
                in_channels = self.latent_dim + self.channels[i]
            else:
                in_channels = self.channels[i+1] + self.channels[i]
            
            
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, self.channels[i], kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(self.channels[i])
            )
            block = residual_block(in_channels, self.channels[i], downsample=downsample)
     
            self.decoder_layers.append(block)
            
        # ========== 输出层 ==========
        self.output_layer = nn.Sequential(
            nn.Conv1d(self.channels[0], d_token, kernel_size=1),
            # nn.Sigmoid()  
        )

        # ========== 单模态预测(Unimodal Predictor) ==========
        self.unimodal_predictor = UnimodalPredictor(d_token, n_num_features)
        
        
    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            raw_x: 张量，形状为 [batch, n_features]
            
        Returns:
            logits: [batch, 1] 的二分类 logits
        """


        # 使用FT-Transformer的NumericalFeatureTokenizer将数值特征转换为token序列
        x_tokens = self.tokenizer(raw_x)  # (batch_size, n_num_features, d_token)
        # 维度转换以适配 Conv1d: [batch_size, n_num_features, d_token] -> [Batch, d_token, n_num_features]
        x = x_tokens.transpose(1, 2)
        
        # ========== 编码 ==========
        encoder_outputs = []
        current = x
        
        for layer in self.encoder_layers:
            current = layer(current)
            encoder_outputs.append(current)
        
        # ========== 桥接层 ==========
        latent = self.bridge(current)
        
        # ========== 解码 ==========
        current = latent
        for i, decoder_layer in enumerate(self.decoder_layers):
            # 获取对应的编码器输出
            encoder_idx = self.num_layers - 1 - i
            encoder_out = encoder_outputs[encoder_idx]
            
            # 调整尺寸以对齐
            if current.shape[2] != encoder_out.shape[2]:
                current = F.interpolate(
                    current, 
                    size=encoder_out.shape[2], 
                    mode='linear', 
                    align_corners=False
                )
            
            # 注意力融合
            attended = self.attention_gates[i](current, encoder_out)
            # 拼接
            current = torch.cat([current, attended], dim=1)
            # 解码层
            current = decoder_layer(current)

        # ========== 最终特征输出 [batch_size, d_token, n_num_features] ==========
        feat_reconstructed = self.output_layer(current)
        residual = x - feat_reconstructed
        # residual = feat_reconstructed
        
        # 1. 计算单模态预测结果 y_i: [batch_size, 1]
        y_i = self.unimodal_predictor(residual)
        
        # 2. 将增强后的特征转置回 [batch_size, n_num_features, d_token] 供下游多模态融合
        output_features = residual.transpose(1, 2)

        return y_i
    
    @classmethod
    def make_default(
        cls, # 类本身（UFEN）
        n_num_features: int,     # 原始特征维度
        d_token: int = 64,      # 输入通道数
        **kwargs
    ) -> 'UFEN':
        """
        参数:
        - n_num_features: 输入特征维度
        - d_token: token 输入通道数
        返回:
        - UFEN 模型实例
        """
        return cls(
            n_num_features=n_num_features,
            d_token=d_token,
            **kwargs
        )
