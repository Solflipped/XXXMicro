import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from model.FT_transformer import FeatureTokenizer, CLSToken

class UFEN(nn.Module):
    """
    单模态特征提取网络 (Unimodal Feature Extraction Network)
    基于FT-Transformer的tokenization + 多层CNN + 自注意力机制
    """
    def __init__(
        self,
        n_num_features: int,
        d_token: int = 128,
        num_conv_layers: int = 3,
        num_filters_list: Optional[list] = None,
        kernel_sizes: Optional[list] = None,
        dropout: float = 0.15,
        num_heads: int = 2
    ):
        """
        参数:
        - n_num_features: 输入特征维度（表格的列数）
        - d_token: token维度（每个特征映射到的向量维度）
        - num_conv_layers: 卷积层数量
        - num_filters_list: 每层卷积的滤波器数量列表
        - kernel_sizes: 每层卷积核大小列表
        - dropout: dropout率
        - num_heads: 注意力头数
        """
        super(UFEN, self).__init__()
        
        # 设置默认参数
        if num_filters_list is None:
            num_filters_list = [96, 128]  # 默认每层滤波器数量
        if kernel_sizes is None:
            kernel_sizes = [3, 5]  # 默认卷积核大小， 使用不同感受野构建多尺度信息
        
        # 确保参数数量一致
        assert len(num_filters_list) == num_conv_layers
        assert len(kernel_sizes) == num_conv_layers
        
        self.n_num_features = n_num_features
        self.d_token = d_token
        self.num_conv_layers = num_conv_layers
        
        # 使用FT-Transformer的FeatureTokenizer将数值特征转换为token序列
        self.feature_tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=[],  # 纯数值特征
            d_token=d_token
        )
        
        # 2. 多层1D卷积层
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第一层卷积输入是token embedding
        in_channels = d_token   
        
        for i in range(num_conv_layers):
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filters_list[i],
                kernel_size=kernel_sizes[i],
                padding=kernel_sizes[i] // 2  # 保持序列长度不变
            )
            self.conv_layers.append(conv_layer)
            
            # 批归一化
            self.batch_norms.append(nn.BatchNorm1d(num_filters_list[i]))
            
            # 下一层输入维度为当前层输出维度
            in_channels = num_filters_list[i]
        
        # 3. 每层卷积对应的自注意力机制
        self.layer_attentions = nn.ModuleList()
        for filters in num_filters_list:
            # 自注意力
            attention_layer = nn.MultiheadAttention(
                embed_dim=filters,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.layer_attentions.append(attention_layer)
        
        # 4. 上采样/调整维度层 - 将每层特征调整到统一维度
        self.unify_layers = nn.ModuleList()
        for filters in num_filters_list:
            # 使用1x1卷积调整通道数到统一维度
            unify_layer = nn.Conv1d(
                in_channels=filters,
                out_channels=d_token,
                kernel_size=1
            )
            self.unify_layers.append(unify_layer)
        
        # 5. 线性输出层 (用于单模态预测)
        self.output_layer = nn.Linear(d_token, 1) 
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 可选的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.constant_(m.bias, 0)


    @classmethod
    def make_default(
        cls, # 类本身（UFEN）
        n_num_features: int,
        d_token: int = 192,
        num_conv_layers: int = 3,
        d_out: int = 1, 
        **kwargs
    ) -> 'UFEN':
        """
        创建默认配置的 UFEN 模型（工厂方法，兼容 train.py）
        
        参数:
        - n_num_features: 输入特征维度
        - d_token: token 嵌入维度
        - num_conv_layers: 卷积层数量
        - d_out: 输出维度（默认 1，用于二分类）
        - **kwargs: 其他可选参数（num_filters_list, kernel_sizes, dropout, num_heads 等）
        
        返回:
        - UFEN 模型实例
        """
        return cls(
            n_num_features=n_num_features,
            d_token=d_token,
            num_conv_layers=num_conv_layers,
            **kwargs
        )
        
    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        输入:  x_num - 形状为 (batch_size, n_num_features) 的表格数据
        输出: 
        - global_features: 全局特征表示 (batch_size, d_token)
        - y_i: 单模态预测 (batch_size, 1)
        """
        # batch_size = x_num.shape[0]

        # 1. Ft-Transformer的FeatureTokenizer、CLSToken将数值特征转换为token序列
        token_embeddings = self.feature_tokenizer(x_num, None)  # (batch_size, n_num_features, d_token)

        # 2. 多层卷积特征提取 + 自注意力处理
        # 调整维度以适应Conv1d: (batch, n_num_features, d_token) -> (batch, d_token, n_num_features)
        conv_input = token_embeddings.transpose(1, 2)
        
        # 存储各层处理后的特征
        unified_features_list = []
        
        for i in range(self.num_conv_layers):
            # a) 卷积层
            conv_out = self.conv_layers[i](conv_input)
            conv_out = self.batch_norms[i](conv_out)
            conv_out = F.relu(conv_out)
            conv_out = self.dropout(conv_out)
            
            # 调整维度以适应注意力机制: (batch, d_token, n_num_features) -> (batch, n_num_features, d_token)
            conv_out_att = conv_out.transpose(1, 2)
            
            # b) 自注意力机制
            attn_output, _ = self.layer_attentions[i](
                conv_out_att, conv_out_att, conv_out_att
            )
            
            # c) 元素相乘
            self_att_out = conv_out_att + attn_output  # 元素相加
            
            # d) 调整维度以适应1x1卷积: (batch, n_num_features, d_token) -> (batch, d_token, n_num_features)
            self_att_out = self_att_out.transpose(1, 2)
            
            # e) 上采样/统一维度, 实际使用1x1卷积调整维度
            unified_feature = self.unify_layers[i](self_att_out)  # (batch, unified_dim, seq_len)
            
            # 保存统一维度后的特征
            unified_features_list.append(unified_feature)
            
            # 更新输入为当前层输出，用于下一层卷积
            conv_input = conv_out
        
        # 3. 多层特征融合
        fusion_features = torch.zeros_like(unified_features_list[0])
        for feature in unified_features_list:
            fusion_features = fusion_features + feature  # 元素相加
        
        # 4. 调整到原始维度: (batch, d_token, n_num_features+1) -> (batch, n_num_features+1, d_token)
        fusion_features = fusion_features.transpose(1, 2)
        
        # 5. 全局特征聚合 - 平均池化
        global_features = torch.mean(fusion_features, dim=1)  # (batch, d_token)
        
        # 6. 单模态预测 - 输出 logits（不使用 sigmoid，由 BCEWithLogitsLoss 处理）
        logits = self.output_layer(global_features)  # (batch_size, 1)
    
        return logits
    
