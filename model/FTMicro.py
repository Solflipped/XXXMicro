import torch
import torch.nn as nn
import torch.nn.functional as F

class residual_block(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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
    """ 注意力门"""
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
        
        self.relu = nn.ReLU(inplace=True)
        
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
  


class UFEN(nn.Module):
    """
    单模态特征提取网络 (Unimodal Feature Extraction Network)，用于数据的特征增强
    """
    def __init__(self, 
                 d_token: int ,            # 输入通道数
                 n_num_features: int,      # 特征长度
                 base_channels=64,         # 基础通道数
                 expansion_factor=2,       # 每次扩展的倍数
                 num_layers=4,             # 编码器/解码器层数
                 latent_dim=256):          # 潜在表示维度
        super(UFEN, self).__init__()

        self.d_token = d_token
        self.n_num_features = n_num_features
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # 计算各层通道数
        self.channels = [base_channels * (expansion_factor ** i) 
                        for i in range(num_layers)]
        
        # ========== 编码器 ==========
        self.encoder_layers = nn.ModuleList()
        # self.pool_layers = nn.ModuleList()  # 可选的下采样
        
        # 第一层
        self.encoder_layers.append(
            residual_block(d_token, self.channels[0])
        )
    
        # 中间层
        for i in range(1, num_layers):
            block = residual_block(self.channels[i-1], self.channels[i])
            self.encoder_layers.append(block)
        
        # ========== 桥接层（潜在空间） ==========
        self.bridge = nn.Sequential(
            nn.Conv1d(self.channels[-1], self.latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # ========== 注意力门（如果需要） ==========
        self.attention_gates = nn.ModuleList()
        for i in range(num_layers-1, -1, -1):
            self.attention_gates.append(
                attention_gate(
                    F_g=self.channels[-1] if i == num_layers-1 else self.channels[i+1],
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
            

            block = residual_block(in_channels, self.channels[i])
     
            self.decoder_layers.append(block)
            
        # ========== 输出层 ==========
        self.output_layer = nn.Sequential(
            nn.Conv1d(self.channels[0], d_token, kernel_size=1),
            nn.Sigmoid()  
        )
        
        # ========== 特征提取层（用于下游任务） ==========
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.BatchNorm1d(self.latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.latent_dim // 2, self.latent_dim // 4)  # 最终的增强特征
        )
        
    def forward(self, x, return_features=True):
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, 1, feature_length]
            return_features: 是否返回增强的特征表示
            
        返回:
            output: 重建的输出 [batch_size, 1, feature_length]
            features: 增强的特征表示 [batch_size, latent_dim//4] (如果return_features=True)
        """
        batch_size = x.shape[0]
        
        # ========== 编码 ==========
        encoder_outputs = []
        current = x
        
        for i, encoder_layer in enumerate(self.encoder_layers):
            current = encoder_layer(current)
            encoder_outputs.append(current)
            
            # 可选的下采样
            # if i < len(self.pool_layers):
            #     current = self.pool_layers[i](current)
        
        # ========== 桥接层 ==========
        latent = self.bridge(current)
        
        # ========== 解码 ==========
        current = latent
        attention_idx = 0
        
        for i, decoder_layer in enumerate(self.decoder_layers):
            # 获取对应的编码器输出
            encoder_idx = self.num_layers - 1 - i
            encoder_out = encoder_outputs[encoder_idx]
            
            # 调整尺寸（如果需要）
            if current.shape[2] != encoder_out.shape[2]:
                current = F.interpolate(
                    current, 
                    size=encoder_out.shape[2], 
                    mode='linear', 
                    align_corners=False
                )
            
            # 应用注意力
            attention_gate = self.attention_gates[attention_idx]
            attended = attention_gate(current, encoder_out)
            # 拼接
            current = torch.cat([current, attended], dim=1)
            attention_idx += 1
   
            # 解码层
            current = decoder_layer(current)
            
            # 可选的上采样
            # if i < len(self.upsample_layers):
            #     current = self.upsample_layers[i](current)
        
        # ========== 输出 ==========
        output = self.output_layer(current)
        
        if return_features:
            # 提取增强特征
            features = self.feature_extractor(latent)
            return output, features
        else:
            return output