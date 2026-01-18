"""PyTorch implementation of MBT Encoder components.

MBT (Multimodal Bottleneck Transformers) 

主要组件：
1. EncoderBlock: 标准 Transformer 编码器块 + Stochastic Depth
2. Encoder: 多模态融合编码器，支持三种融合策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math
from model.FT_transformer import FeatureTokenizer, CLSToken, _TokenInitialization


class MLP(nn.Module):
    """
    MLP Block（多层感知机块）
    
    结构：Linear → GELU → Dropout → Linear → Dropout
    用于 Transformer 中的前馈网络（FFN）
    
    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层维度（通常是 in_features 的 4 倍）
        dropout_rate: Dropout 概率
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        # 第一层：升维
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数：GELU（Gaussian Error Linear Unit）
        self.act = nn.GELU()
        # 第二层：降维回原始维度
        self.fc2 = nn.Linear(hidden_features, in_features)
        # Dropout 正则化
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # x: [batch, seq_len, in_features]
        x = self.fc1(x)           # [batch, seq_len, hidden_features]
        x = self.act(x)           # GELU 激活
        x = self.dropout(x)       # Dropout
        x = self.fc2(x)           # [batch, seq_len, in_features]
        x = self.dropout(x)       # Dropout
        return x



class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        return output



class EncoderBlock(nn.Module):
    """
    Transformer 编码器块（EncoderBlock）
    
    这是标准的 Transformer 编码器层，采用 Pre-Norm 架构并加入 Stochastic Depth。
    
    结构流程：
    1. LayerNorm → Multi-Head Self-Attention → Dropout → Stochastic Depth + 残差
    2. LayerNorm → MLP → Stochastic Depth + 残差
    
    Stochastic Depth（随机深度）：
    - 训练时：以 droplayer_p 的概率随机"跳过"整个层（直接返回输入）
    - 推理时：正常执行所有层
    - 目的：类似 Dropout，但作用在层级而非神经元级别，提高泛化能力
    
    Args:
        dim: 嵌入维度（特征维度）
        mlp_dim: MLP 隐藏层维度（通常是 dim 的 4 倍）
        num_heads: 多头注意力的头数
        dropout_rate: 标准 Dropout 概率
        attention_dropout_rate: 注意力层的 Dropout 概率
        droplayer_p: Stochastic Depth 概率（层丢弃概率）
    """
    
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        droplayer_p: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.droplayer_p = droplayer_p  # Stochastic Depth 概率
        
        # === 注意力块 ===
        self.norm1 = nn.LayerNorm(dim)  # Pre-Norm: 在注意力前做层归一化
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout_rate,
            batch_first=True  # 输入格式为 (batch, seq, feature)
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # === MLP 块 ===
        self.norm2 = nn.LayerNorm(dim)  # Pre-Norm: 在 MLP 前做层归一化
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_dim,
            dropout_rate=dropout_rate
        )
        
    def get_drop_pattern(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """
        生成 Stochastic Depth 的 drop pattern（丢弃模式）
        
        工作原理：
        - 对每个样本独立决定是否丢弃该层（不是丢弃某些神经元）
        - shape = (batch_size, 1, 1, ...) 确保同一样本的所有 token 统一处理
        - 返回 0 表示不丢弃，返回 1 表示丢弃
        
        Args:
            x: 输入张量
            training: 是否在训练模式
            
        Returns:
            drop_pattern: 0 或 1 的张量，shape = (batch_size, 1, ...)
        """
        if training and self.droplayer_p > 0:
            # 保留概率
            keep_prob = 1 - self.droplayer_p
            # 创建 shape: (batch_size, 1, 1, ...) 用于广播
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            # 生成随机数
            random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
            # 转换为二值：random < keep_prob 则保留（=1），否则丢弃（=0）
            binary_tensor = (random_tensor < keep_prob).float()
            # 返回 drop pattern: 1 表示丢弃，0 表示保留
            return 1.0 - binary_tensor
        else:
            # 推理时不丢弃
            return 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        处理流程：
        1. 注意力分支: LayerNorm → Attention → Dropout → Stochastic Depth → 残差
        2. MLP 分支: LayerNorm → MLP → Stochastic Depth → 残差
        
        Args:
            x: 输入张量，shape = (batch, seq_len, dim)
            
        Returns:
            输出张量，shape = (batch, seq_len, dim)
        """
        # === 第一部分：Self-Attention 块 ===
        # Step 1: Pre-LayerNorm
        x_norm = self.norm1(x)  # [batch, seq_len, dim]
        
        # Step 2: Multi-Head Self-Attention
        # 注意：PyTorch 的 MultiheadAttention 需要 (query, key, value) 三个输入
        # Self-Attention 中这三者都是 x_norm
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # [batch, seq_len, dim]
        
        # Step 3: Dropout
        attn_out = self.dropout1(attn_out)
        
        # Step 4: Stochastic Depth + 残差连接
        # 公式: x = attn_out * (1 - drop_pattern) + x
        # - drop_pattern = 0: x = attn_out + x (正常残差)
        # - drop_pattern = 1: x = x (跳过该层)
        drop_pattern = self.get_drop_pattern(x, self.training)
        x = attn_out * (1.0 - drop_pattern) + x
        
        # === 第二部分：MLP 块 ===
        # Step 1: Pre-LayerNorm
        x_norm = self.norm2(x)  # [batch, seq_len, dim]
        
        # Step 2: MLP (前馈网络)
        mlp_out = self.mlp(x_norm)  # [batch, seq_len, dim]
        
        # Step 3: Stochastic Depth + 残差连接
        drop_pattern = self.get_drop_pattern(x, self.training)
        x = mlp_out * (1.0 - drop_pattern) + x
        
        return x  # [batch, seq_len, dim]


class Encoder(nn.Module):
    """
    多模态融合 Transformer 编码器（Encoder）- Bottleneck 融合版本
    
    这是 MBT 的核心组件，负责处理多个模态（species + ko）的融合。
    
    ============================================================================
    核心概念：Bottleneck 多模态融合
    ============================================================================
    
    方案：分阶段融合 + 瓶颈 token
    - 前 N 层：各模态独立编码（学习模态内部特征）
    - 后续层：通过瓶颈 token 进行模态间信息交换
    - fusion_layer 参数控制在哪一层开始融合
    
    原理：使用少量可学习的"瓶颈 token"作为信息交换中介
    
    工作流程：
    ┌──────────────────┐     ┌──────────────┐     ┌──────────────────┐
    │  Species tokens  │ ──→ │  Bottleneck  │ ←── │    KO tokens     │
    │   [B, N1, D]     │     │   [B, K, D]  │     │   [B, N2, D]     │
    └──────────────────┘     └──────────────┘     └──────────────────┘
             ↓                      ↓                       ↓
       拼接在一起            共享瓶颈信息             拼接在一起
       [B, N1+K, D]           (取平均)              [B, N2+K, D]
             ↓                      ↓                       ↓
      Encoder_Species          更新瓶颈             Encoder_KO
    
    ============================================================================
    ============================================================================
    Args:
        dim: 嵌入维度
        mlp_dim: MLP 隐藏层维度
        num_layers: Transformer 层数
        num_heads: 多头注意力头数
        dropout_rate: Dropout 概率
        attention_dropout_rate: 注意力 Dropout 概率
        stochastic_droplayer_rate: Stochastic Depth 最大概率（线性递增）
        modality_fusion: 要融合的模态元组，如 ('species', 'ko')
        fusion_layer: 从哪一层开始融合（0 = 早期融合，num_layers = 晚期融合）
        test_with_bottlenecks: 测试时是否使用瓶颈（训练时总是使用）
    """
    
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        stochastic_droplayer_rate: float = 0.0,
        modality_fusion: Tuple[str, ...] = ('species', 'ko'),
        fusion_layer: int = 0,
        test_with_bottlenecks: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.stochastic_droplayer_rate = stochastic_droplayer_rate
        self.modality_fusion = modality_fusion
        self.fusion_layer = fusion_layer
        self.test_with_bottlenecks = test_with_bottlenecks
        
        # === 为每一层和每个模态创建编码器块 ===
        # 使用 ModuleDict 存储，便于动态访问
        self.encoder_blocks = nn.ModuleDict()
        
        for lyr in range(num_layers):
            # Stochastic Depth 概率随层数线性递增
            # 浅层概率低，深层概率高（逐渐增加正则化强度）
            droplayer_p = (lyr / max(num_layers - 1, 1)) * stochastic_droplayer_rate
            
            # 为每个模态创建独立的编码器
            for modality in modality_fusion:
                self.encoder_blocks[f'{modality}_layer_{lyr}'] = EncoderBlock(
                    dim=dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    droplayer_p=droplayer_p
                )
        
        # === 最终的 LayerNorm ===
        self.encoder_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        bottleneck: torch.Tensor
    ) -> torch.Tensor:
        """
        Bottleneck 融合编码器的前向传播
        
        处理流程：
        1. 逐层处理：
           - 融合前（lyr < fusion_layer）：各模态独立编码
           - 融合后（lyr >= fusion_layer）：通过瓶颈 token 进行模态融合
        2. 拼接所有模态输出，应用最终的 LayerNorm
        
        Args:
            x: 模态字典，如 {'species': [B, N1, D], 'ko_genes': [B, N2, D]}
               - B: batch size
               - N1, N2: 各模态的特征数量（物种数、基因数）
               - D: 嵌入维度
            bottleneck: 瓶颈 token，shape = (batch, n_bottlenecks, dim)
                       通常 n_bottlenecks = 4-8
            
        Returns:
            编码后的输出，shape = (batch, total_seq_len, dim)
            其中 total_seq_len = sum(各模态的 seq_len)
        """
        # 决定是否使用瓶颈（训练时或测试时根据配置）
        use_bottlenecks = self.training or self.test_with_bottlenecks
        
        # ============================================================
        # 逐层处理
        # ============================================================
        for lyr in range(self.num_layers):
            # 获取当前层各模态的编码器
            encoders = {}
            for modality in self.modality_fusion:
                key = f'{modality}_layer_{lyr}'
                encoders[modality] = self.encoder_blocks[key]
            
            # ------------------------------------------------------------
            # 判断：当前层是否需要融合？
            # ------------------------------------------------------------
            need_fusion = (
                lyr >= self.fusion_layer and  # 已到达融合层
                len(self.modality_fusion) > 1 and  # 有多个模态
                use_bottlenecks  # 使用瓶颈
            )
            
            if not need_fusion:
                # ========== 分支 A: 各模态独立处理 ==========
                for modality in self.modality_fusion:
                    x[modality] = encoders[modality](x[modality])
            
            else:
                # ========== 分支 B: Bottleneck 融合 ==========
                bottle = []  # 存储各模态处理后的瓶颈 token
                
                for modality in self.modality_fusion:
                    # Step 1: 拼接模态 token 和瓶颈 token
                    t_mod = x[modality].shape[1]  # 模态的 token 数量
                    in_mod = torch.cat([x[modality], bottleneck], dim=1)
                    # 现在 in_mod.shape = [B, t_mod + n_bottlenecks, D]
                    
                    # Step 2: 通过编码器处理
                    # 注意力机制会让模态 token 和瓶颈 token 相互交互
                    out_mod = encoders[modality](in_mod)
                    
                    # Step 3: 分离模态部分和瓶颈部分
                    x[modality] = out_mod[:, :t_mod]  # 前 t_mod 个是模态输出
                    bottle.append(out_mod[:, t_mod:])  # 后面的是瓶颈输出
                
                # Step 4: 更新瓶颈 = 所有模态瓶颈的平均
                # 这样瓶颈就聚合了所有模态的信息
                bottleneck = torch.stack(bottle, dim=-1).mean(dim=-1)
        
        # ============================================================
        # 拼接所有模态输出
        # ============================================================
        x_out = []
        for modality in self.modality_fusion:
            x_out.append(x[modality])
        x_out = torch.cat(x_out, dim=1)
        
        # ============================================================
        # 最终的 LayerNorm
        # ============================================================
        encoded = self.encoder_norm(x_out)
        
        return encoded  # [batch, total_seq_len, dim]



class MBT(nn.Module):
    """
    Multimodal Bottleneck Transformer (MBT) with FTTransformer tokenization.

    功能：
    - 使用 FTTransformer 的 FeatureTokenizer 和 CLSToken 处理 species_abundance.csv 和 ko_abundance.csv
    - 支持瓶颈 token 融合多模态数据（species 和 ko）
    - 提供 make_default 类方法，仿照 FTTransformer 优化超参数
    - 固定 modality_fusion=("species", "ko")，mlp_dim=dim*4
    - 确保 fusion_layer <= num_layers

    假设：
    - 输入为两个 CSV 文件（数值型表格，行是样本，列是特征，值为相对丰度）
    - 输出为二分类 logits [batch, 1]
    """
    def __init__(
        self,
        feature_tokenizers: Dict[str, FeatureTokenizer],
        cls_tokens: Dict[str, CLSToken],
        dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        modality_fusion: Tuple[str, ...] = ("species", "ko"),
        fusion_layer: int = 0,
        n_bottlenecks: int = 4,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        stochastic_droplayer_rate: float = 0.0,
        representation_size: Optional[int] = None,
        classifier: str = "token",
        test_with_bottlenecks: bool = True,
        use_cross_atn: bool = True,
    ) -> None:
        super().__init__()
        assert classifier == "token", "当前实现仅支持基于 CLS 的 'token' 分类方式"
        assert fusion_layer <= num_layers, f"fusion_layer ({fusion_layer}) 必须小于或等于 num_layers ({num_layers})"

        self.dim = dim
        self.representation_size = representation_size
        self.classifier = classifier
        self.modality_fusion = modality_fusion
        self.use_cross_atn = use_cross_atn

        # FeatureTokenizer 和 CLSToken
        self.feature_tokenizers = nn.ModuleDict(feature_tokenizers)
        self.cls_tokens = nn.ModuleDict(cls_tokens)

        # Encoder
        self.encoder = Encoder(
            dim=dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            stochastic_droplayer_rate=stochastic_droplayer_rate,
            modality_fusion=modality_fusion,
            fusion_layer=fusion_layer,
            test_with_bottlenecks=test_with_bottlenecks,
        )

        # 瓶颈 token
        n_bottlenecks_actual = n_bottlenecks + (1 if classifier == "token" else 0)
        self.bottleneck = nn.Parameter(torch.randn(1, n_bottlenecks_actual, dim) * 0.02)

        # 交叉注意力（可选）：在编码器之前对齐单模态嵌入与其“原始”值
        if self.use_cross_atn:
            assert dim % num_heads == 0, "当启用交叉注意力时，要求 dim 能被 num_heads 整除"
            k_dim = v_dim = dim // num_heads
            self.cross_attns = nn.ModuleDict({
                m: CrossAttention(in_dim1=dim, in_dim2=dim, k_dim=k_dim, v_dim=v_dim, num_heads=num_heads)
                for m in modality_fusion
            })

        # 表示层和分类头
        out_dim = representation_size if representation_size is not None else dim
        self.pre_logits = nn.Linear(dim, representation_size) if representation_size else None
        self.output_head = nn.Linear(out_dim, 1)
        nn.init.zeros_(self.output_head.weight)
        if self.output_head.bias is not None:
            nn.init.zeros_(self.output_head.bias)

    @classmethod
    def get_default_config(cls, num_layers: int = 4) -> Dict[str, Any]:
        """
        获取默认配置，仿照 FTTransformer.get_default_transformer_config。
        
        Args:
            num_layers: Transformer 层数（1~6）
            
        Returns:
            配置字典，包含 d_token, dropout_rate, attention_dropout_rate 等
        """
        assert 1 <= num_layers <= 6, "num_layers 必须在 1~6 之间"
        grid = {
            'd_token': [96, 128, 192, 256, 320, 384],
            'attention_dropout': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
            'dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
        }
        config = {
            'num_layers': num_layers,
            'd_token': grid['d_token'][num_layers - 1],
            'mlp_dim': grid['d_token'][num_layers - 1] * 4,  # mlp_dim = dim * 4
            'dropout_rate': grid['dropout'][num_layers - 1],
            'attention_dropout_rate': grid['attention_dropout'][num_layers - 1],
            'stochastic_droplayer_rate': 0.1,  # 默认值，参考 MBT 论文
            'modality_fusion': ('species', 'ko'),
            'fusion_layer': min(num_layers, 8),  # 确保不超过 num_layers
            'n_bottlenecks': 4,
            'num_heads': 8,  # 默认值，参考 FTTransformer
            'representation_size': None,
            'test_with_bottlenecks': True,
        }
        return config

    @classmethod
    def make_default(
        cls,
        *,
        n_species_features: int,
        n_ko_features: int,
        num_layers: int = 3,
        num_heads: int = 8,
        fusion_layer: Optional[int] = None,
        n_bottlenecks: int = 4,
        representation_size: Optional[int] = None,
        test_with_bottlenecks: bool = True,
        use_cross_atn: bool = True,
    ) -> 'MBT':
        """
        创建默认 MBT 实例，仿照 FTTransformer.make_default。

        Args:
            n_species_features: species_abundance.csv 的特征数
            n_ko_features: ko_abundance.csv 的特征数
            num_layers: Transformer 层数
            num_heads: 多头注意力头数
            fusion_layer: 从哪一层开始融合（默认 min(num_layers, 8)）
            n_bottlenecks: 瓶颈 token 数量
            representation_size: 表示层维度（可选）
            test_with_bottlenecks: 测试时是否使用瓶颈

        Returns:
            MBT 实例
        """
        config = cls.get_default_config(num_layers)
        config.update({
            'num_heads': num_heads,
            'n_bottlenecks': n_bottlenecks,
            'representation_size': representation_size,
            'test_with_bottlenecks': test_with_bottlenecks,
        })
        if fusion_layer is not None:
            assert fusion_layer <= num_layers, f"fusion_layer ({fusion_layer}) 必须小于或等于 num_layers ({num_layers})"
            config['fusion_layer'] = fusion_layer
        elif config['fusion_layer'] > num_layers:
            print(f"Warning: fusion_layer ({config['fusion_layer']}) 超过 num_layers ({num_layers}), 设置为 {num_layers}")
            config['fusion_layer'] = num_layers

        # 创建 FeatureTokenizer 和 CLSToken
        feature_tokenizers = {
            'species': FeatureTokenizer(
                n_num_features=n_species_features,
                cat_cardinalities=[],
                d_token=config['d_token'],
            ),
            'ko': FeatureTokenizer(
                n_num_features=n_ko_features,
                cat_cardinalities=[],
                d_token=config['d_token'],
            ),
        }
        cls_tokens = {
            'species': CLSToken(d_token=config['d_token'], initialization='uniform'),
            'ko': CLSToken(d_token=config['d_token'], initialization='uniform'),
        }

        return cls(
            feature_tokenizers=feature_tokenizers,
            cls_tokens=cls_tokens,
            dim=config['d_token'],
            mlp_dim=config['mlp_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            modality_fusion=config['modality_fusion'],
            fusion_layer=config['fusion_layer'],
            n_bottlenecks=config['n_bottlenecks'],
            dropout_rate=config['dropout_rate'],
            attention_dropout_rate=config['attention_dropout_rate'],
            stochastic_droplayer_rate=config['stochastic_droplayer_rate'],
            representation_size=config['representation_size'],
            test_with_bottlenecks=config['test_with_bottlenecks'],
            use_cross_atn=use_cross_atn,
        )

    def tokenize_inputs(self, raw_x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        将原始表格数据转换为 token 序列。

        Args:
            raw_x: 字典，包含 'species' 和 'ko' 的张量，形状为 [batch, n_features]

        Returns:
            x_tokens: 字典，包含每个模态的 token 序列，形状为 [batch, n_tokens + 1, d_token]
        """
        x_tokens = {}
        for modality in self.modality_fusion:
            assert modality in raw_x, f"缺少模态 {modality} 的输入数据"
            x = self.feature_tokenizers[modality](raw_x[modality], None)  # [batch, n_tokens, d_token]
            x = self.cls_tokens[modality](x)  # [batch, n_tokens + 1, d_token]
            x_tokens[modality] = x
        return x_tokens

    def forward(self, raw_x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向推理。

        Args:
            raw_x: 字典，包含 'species' 和 'ko' 的原始数据，形状为 [batch, n_features]

        Returns:
            logits: [batch, 1] 的二分类 logits
        """
        # 转换为 token 序列
        x_tokens = self.tokenize_inputs(raw_x)

        # 可选：在进入 Encoder 之前进行一次按模态的交叉注意力对齐
        if self.use_cross_atn:
            init_embed = x_tokens
            x_tokens = {m: self.cross_attns[m](x_tokens[m], init_embed[m]) for m in self.modality_fusion}

        # 维度校验
        B = next(iter(x_tokens.values())).size(0)
        for m in self.modality_fusion:
            assert x_tokens[m].dim() == 3, f"{m} 的输入应为 [B, L, D]"
            assert x_tokens[m].size(-1) == self.dim, f"{m} 的 token 维度与 dim 不一致"

        # 扩展瓶颈
        bottleneck = self.bottleneck.expand(B, -1, -1)

        # 编码器
        encoded = self.encoder(x_tokens, bottleneck)

        # 提取 CLS token
        cls_list = []
        offset = 0
        for m in self.modality_fusion:
            Lm = x_tokens[m].size(1)
            # 注意：我们的 CLSToken 会把 CLS 追加到序列“末尾”，
            # 因此每个模态片段中的最后一个位置是 CLS，对应全局下标 offset + (Lm - 1)
            cls_list.append(encoded[:, offset + (Lm - 1), :])
            offset += Lm

        # 表示层和分类头
        if self.pre_logits is not None:
            cls_list = [torch.tanh(self.pre_logits(h)) for h in cls_list]
        logit_list = [self.output_head(h) for h in cls_list]
        logits = torch.stack(logit_list, dim=-1).mean(dim=-1)
        return logits

    def make_default_optimizer(self) -> torch.optim.AdamW:
        """
        创建默认 AdamW 优化器，仿照 FTTransformer。
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
        )

