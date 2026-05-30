"""
XXXMicro: 多模态微生物组疾病预测框架 (FTMicro v2)

架构设计依据:
  - Stage 1 (去噪增强): 借鉴 Disrobiom 的核心发现 ——
    重构残差包含判别性生物标志物信息。用 Self-Attention 替代 Conv1d 以获得置换不变性,
    适配无序的微生物丰度数据 (物种/KO)。
  - Stage 2 (特征压缩): 采用可学习 query token 通过交叉注意力聚合高维 token 序列,
    类似 Perceiver (Jaegle et al., 2021) 的思路, 将 N 个特征 token 压缩为 M 个。
  - Stage 3 (多模态融合): 借鉴 MSFT 的 Bottleneck 机制, 共享瓶颈 token 在两个模态间
    传递信息, 实现对称的多模态交互。
  - Stage 4 (分类): 融合各模态 CLS 特征与瓶颈特征, 通过 MLP 输出预测。

训练策略:
  - 多任务损失: BCE(分类) + lambda * MSE(去噪重构), 半监督思路
  - 去噪重构提供不依赖标签的正则化信号, 缓解小样本过拟合
"""

import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, cast
from collections import OrderedDict
from skorch import NeuralNetBinaryClassifier

from model.FT_transformer import (
    NumericalFeatureTokenizer, CLSToken, MultiheadAttention,
    FTTransformer, Transformer, _make_nn_module
)

ModuleType = Union[str, callable]


# ============================================================
# Stage 1: 单模态去噪增强模块
# ============================================================
class DenoisingEncoder(nn.Module):
    """
    基于 Self-Attention 的单模态去噪增强器。

    训练时: 对 token 序列施加随机掩码 → Self-Attention 编码 → 重构干净 token
    推理时: 直接编码, 无掩码

    设计依据: Disrobiom 证明自编码器的重构残差含判别信息。
    我们以掩码去噪的方式实现类似效果, 同时 Self-Attention 保证置换不变性。
    """
    def __init__(self, d_token: int, n_heads: int = 4,
                 n_blocks: int = 1, dropout: float = 0.1,
                 mask_ratio: float = 0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.d_token = d_token

        # 使用 FT-Transformer 的默认配置构建 Transformer blocks
        # 但只取 block 部分, 不要 Head
        self.blocks = nn.ModuleList()
        for layer_idx in range(n_blocks):
            layer = nn.ModuleDict({
                'attention': MultiheadAttention(
                    d_token=d_token, n_heads=n_heads,
                    dropout=dropout, bias=True, initialization='kaiming'
                ),
                'ffn': Transformer.FFN(
                    d_token=d_token,
                    d_hidden=int(d_token * 4 / 3),  # ReGLU style
                    bias_first=True, bias_second=True,
                    dropout=dropout, activation='ReGLU'
                ),
                'attention_residual_dropout': nn.Dropout(dropout),
                'ffn_residual_dropout': nn.Dropout(dropout),
            })
            # pre-normalization (FT-Transformer 默认)
            if layer_idx > 0:
                layer['attention_normalization'] = nn.LayerNorm(d_token)
            layer['ffn_normalization'] = nn.LayerNorm(d_token)
            self.blocks.append(layer)

        self.prenormalization = True

    def _start_residual(self, layer, stage, x):
        x_residual = x
        if self.prenormalization:
            norm_key = f'{stage}_normalization'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer, stage, x, x_residual):
        x_residual = layer[f'{stage}_residual_dropout'](x_residual)
        x = x + x_residual
        return x

    def forward(self, tokens: Tensor):
        """
        Args:
            tokens: [B, N, d_token] — Tokenizer 输出的 token 序列
        Returns:
            enhanced: [B, N, d_token] — 增强后的 token 序列
            loss_denoise: scalar — 去噪重构损失 (仅训练时有效, 推理时为 0)
        """
        clean_tokens = tokens.detach()  # 保存干净版本用于计算去噪损失 (detach 防止梯度回传到目标)

        mask = None
        if self.training and self.mask_ratio > 0:
            # 随机掩码: 将部分 token 置零, 模拟特征缺失/噪声
            B, N, D = tokens.shape
            mask = torch.bernoulli(
                torch.full((B, N, 1), 1.0 - self.mask_ratio, device=tokens.device)
            )
            tokens = tokens * mask

        # Self-Attention 编码
        x = tokens
        for layer in self.blocks:
            layer = cast(nn.ModuleDict, layer)
            x_residual = self._start_residual(layer, 'attention', x)
            x_residual, _ = layer['attention'](x_residual, x_residual, None, None)
            x = self._end_residual(layer, 'attention', x, x_residual)

            x_residual = self._start_residual(layer, 'ffn', x)
            x_residual = layer['ffn'](x_residual)
            x = self._end_residual(layer, 'ffn', x, x_residual)

        enhanced = x

        # 计算去噪损失
        if self.training and self.mask_ratio > 0 and mask is not None:
            masked_positions = (1.0 - mask)
            squared_error = (enhanced - clean_tokens).pow(2) * masked_positions
            denom = masked_positions.sum() * enhanced.shape[-1]
            loss_denoise = squared_error.sum() / denom.clamp_min(1.0)
        else:
            loss_denoise = torch.tensor(0.0, device=tokens.device)

        return enhanced, loss_denoise


# ============================================================
# Stage 2: 特征压缩模块
# ============================================================
class FeatureCompressor(nn.Module):
    """
    使用可学习的 query token 通过交叉注意力从长序列中聚合关键信息。

    将 N 个特征 token 压缩为 M 个 (M << N), 类似 Perceiver 的 cross-attention。
    同时追加一个 CLS token 用于后续分类。
    """
    def __init__(self, d_token: int, n_query: int = 8,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_query = n_query

        # 可学习的 query tokens
        self.query_tokens = nn.Parameter(torch.empty(n_query, d_token))
        nn.init.xavier_uniform_(self.query_tokens)

        # CLS token
        self.cls_token = CLSToken(d_token, 'uniform')

        # Cross-Attention: query attend to feature tokens
        self.cross_attn = MultiheadAttention(
            d_token=d_token, n_heads=n_heads,
            dropout=dropout, bias=True, initialization='kaiming'
        )
        self.norm_q = nn.LayerNorm(d_token)
        self.norm_kv = nn.LayerNorm(d_token)

        # FFN after cross-attention
        self.ffn = Transformer.FFN(
            d_token=d_token,
            d_hidden=int(d_token * 4 / 3),
            bias_first=True, bias_second=True,
            dropout=dropout, activation='ReGLU'
        )
        self.norm_ffn = nn.LayerNorm(d_token)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: [B, N, d_token]
        Returns:
            compressed: [B, M+1, d_token]  (M个query + 1个CLS)
        """
        B = tokens.shape[0]

        # 扩展 query tokens 到 batch 维度
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, M, d]

        # Cross-Attention: queries attend to all feature tokens
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(tokens)
        q_attn, _ = self.cross_attn(q_norm, kv_norm, None, None)
        q = q + self.drop(q_attn)

        # FFN
        q_ffn = self.norm_ffn(q)
        q_ffn = self.ffn(q_ffn)
        q = q + self.drop(q_ffn)

        # 追加 CLS token
        compressed = self.cls_token(q)  # [B, M+1, d]

        return compressed


# ============================================================
# Stage 3: 多模态瓶颈融合模块 (借鉴 MSFT)
# ============================================================
class BottleneckFusionLayer(nn.Module):
    """
    单层瓶颈融合: 共享 bottleneck tokens 与各模态交互。

    设计借鉴 MSFT 的 Bottleneck 类:
    - 将 bottleneck tokens 拼接到各模态序列末尾
    - 通过 Transformer block 做自注意力 (模态内 + bottleneck 交互)
    - 分离模态 token 和 bottleneck, 对 bottleneck 求平均
    """
    def __init__(self, d_token: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token

        # 每个模态一个 Transformer block (与 MSFT 一致)
        self.attn_blocks = nn.ModuleDict()
        for name in ['f1', 'f2']:
            self.attn_blocks[name] = nn.ModuleDict({
                'attention': MultiheadAttention(
                    d_token=d_token, n_heads=n_heads,
                    dropout=dropout, bias=True, initialization='kaiming'
                ),
                'ffn': Transformer.FFN(
                    d_token=d_token,
                    d_hidden=int(d_token * 4 / 3),
                    bias_first=True, bias_second=True,
                    dropout=dropout, activation='ReGLU'
                ),
                'attention_normalization': nn.LayerNorm(d_token),
                'ffn_normalization': nn.LayerNorm(d_token),
                'attention_residual_dropout': nn.Dropout(dropout),
                'ffn_residual_dropout': nn.Dropout(dropout),
            })

    def _transformer_block(self, block, x):
        """单个 pre-norm Transformer block 前向传播"""
        # Self-Attention
        x_res = block['attention_normalization'](x)
        x_res, _ = block['attention'](x_res, x_res, None, None)
        x_res = block['attention_residual_dropout'](x_res)
        x = x + x_res
        # FFN
        x_res = block['ffn_normalization'](x)
        x_res = block['ffn'](x_res)
        x_res = block['ffn_residual_dropout'](x_res)
        x = x + x_res
        return x

    def forward(self, bottleneck: Tensor,
                f1_embed: Tensor, f2_embed: Tensor):
        """
        Args:
            bottleneck: [B, K, d]
            f1_embed:   [B, M1, d]   (模态1压缩后的 token)
            f2_embed:   [B, M2, d]   (模态2压缩后的 token)
        Returns:
            bottleneck: [B, K, d]  (更新后)
            f1_embed:   [B, M1, d] (更新后)
            f2_embed:   [B, M2, d] (更新后)
        """
        M1 = f1_embed.shape[1]
        M2 = f2_embed.shape[1]

        btn_hats = []
        new_embeds = {}

        for name, embed, M in [('f1', f1_embed, M1), ('f2', f2_embed, M2)]:
            # 拼接: [模态tokens, bottleneck tokens]
            combined = torch.cat([embed, bottleneck], dim=1)
            # Transformer block
            combined = self._transformer_block(self.attn_blocks[name], combined)
            # 分离
            new_embeds[name] = combined[:, :M, :]
            btn_hats.append(combined[:, M:, :])

        # 对两个模态产生的 bottleneck 取平均
        bottleneck = torch.mean(torch.stack(btn_hats, dim=0), dim=0)

        return bottleneck, new_embeds['f1'], new_embeds['f2']


class BottleneckFusion(nn.Module):
    """
    多层瓶颈融合模块。
    """
    def __init__(self, d_token: int, n_layers: int = 2,
                 num_bottleneck: int = 4, n_heads: int = 4,
                 dropout: float = 0.1, btn_init: str = 'embed'):
        super().__init__()
        self.btn_init = btn_init
        self.num_bottleneck = num_bottleneck
        self.d_token = d_token

        # Bottleneck 初始化策略 (借鉴 MSFT)
        if btn_init == 'embed':
            # 从两个模态的全局表示联合生成初始 bottleneck
            self.btn_proj = nn.Linear(d_token * 2, num_bottleneck * d_token)
            nn.init.normal_(self.btn_proj.weight, mean=0, std=0.02)
            nn.init.zeros_(self.btn_proj.bias)

        # 多层融合
        self.layers = nn.ModuleList([
            BottleneckFusionLayer(d_token, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, f1_embed: Tensor, f2_embed: Tensor) -> tuple:
        """
        Args:
            f1_embed: [B, M1, d]
            f2_embed: [B, M2, d]
        Returns:
            bottleneck: [B, K, d]
            f1_embed:   [B, M1, d]
            f2_embed:   [B, M2, d]
        """
        B = f1_embed.shape[0]

        # 初始化 bottleneck tokens
        if self.btn_init == 'embed':
            # 联合两个模态的全局表示，直接生成 K 个 bottleneck token
            global_feat = torch.cat(
                [f1_embed.mean(dim=1), f2_embed.mean(dim=1)],
                dim=-1,
            )  # [B, 2*d]
            bottleneck = self.btn_proj(global_feat).view(
                B, self.num_bottleneck, self.d_token
            )  # [B, K, d]
        else:
            bottleneck = torch.randn(
                B, self.num_bottleneck, self.d_token,
                device=f1_embed.device
            ) * 0.02

        # 多层融合
        for layer in self.layers:
            bottleneck, f1_embed, f2_embed = layer(bottleneck, f1_embed, f2_embed)

        return bottleneck, f1_embed, f2_embed


# ============================================================
# 主模型: XXXMicro
# ============================================================
class XXXMicro(nn.Module):
    """
    多模态微生物组疾病预测框架。

    流程:
    1. Tokenization: 各模态独立 token 化
    2. 去噪增强: Self-Attention + 掩码去噪 (半监督正则化)
    3. 特征压缩: 可学习 query 聚合关键信息
    4. 瓶颈融合: 共享 bottleneck 实现多模态交互
    5. 分类: CLS + Bottleneck → MLP → logits
    """
    def __init__(self,
                 inputs_dim: Dict[str, tuple],  # {'f1_input': (n_samples, n_feat1), 'f2_input': (n_samples, n_feat2)}
                 d_token: int = 32,
                 n_query: int = 8,
                 num_bottleneck: int = 4,
                 n_attn_heads: int = 4,
                 n_enhance_layers: int = 1,
                 n_fusion_layers: int = 2,
                 dropout: float = 0.1,
                 mask_ratio: float = 0.15,
                 btn_init: str = 'embed'):
        super().__init__()

        self.inputs_dim = inputs_dim
        self.d_token = d_token

        f_names = list(inputs_dim.keys())
        assert len(f_names) == 2, "XXXMicro 需要恰好两个模态的输入"

        # --- Stage 1: Tokenizer + 去噪增强 ---
        self.tokenizers = nn.ModuleDict()
        self.enhancers = nn.ModuleDict()
        for name, (_, n_features) in inputs_dim.items():
            self.tokenizers[name] = NumericalFeatureTokenizer(
                n_features=n_features, d_token=d_token,
                bias=True, initialization='uniform'
            )
            self.enhancers[name] = DenoisingEncoder(
                d_token=d_token, n_heads=n_attn_heads,
                n_blocks=n_enhance_layers, dropout=dropout,
                mask_ratio=mask_ratio
            )

        # --- Stage 2: 特征压缩 ---
        self.compressors = nn.ModuleDict()
        for name in inputs_dim.keys():
            self.compressors[name] = FeatureCompressor(
                d_token=d_token, n_query=n_query,
                n_heads=n_attn_heads, dropout=dropout
            )

        # --- Stage 3: 瓶颈融合 ---
        self.fusion = BottleneckFusion(
            d_token=d_token, n_layers=n_fusion_layers,
            num_bottleneck=num_bottleneck, n_heads=n_attn_heads,
            dropout=dropout, btn_init=btn_init
        )

        # --- Stage 4: 分类头 ---
        # 输入: CLS_f1 + CLS_f2 + flatten(bottleneck)
        cls_dim = d_token * (2 + num_bottleneck)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cls_dim),
            nn.Linear(cls_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, **features) -> tuple:
        """
        Args:
            **features: 关键字参数, 如 f1_input=[B, N1], f2_input=[B, N2]
        Returns:
            logits:        [B, 1]
            loss_denoise:  scalar (去噪辅助损失)
        """
        # numpy → tensor (兼容 skorch)
        device = next(self.parameters()).device
        for name, feat in features.items():
            if isinstance(feat, np.ndarray):
                features[name] = torch.from_numpy(feat).float().to(device)

        f_names = list(self.inputs_dim.keys())

        # Stage 1: Tokenization + 去噪增强
        enhanced = OrderedDict()
        total_denoise_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for name in f_names:
            tokens = self.tokenizers[name](features[name])       # [B, N, d]
            enh, d_loss = self.enhancers[name](tokens)           # [B, N, d], scalar
            enhanced[name] = enh
            total_denoise_loss = total_denoise_loss + d_loss

        total_denoise_loss = total_denoise_loss / len(f_names)

        # Stage 2: 特征压缩
        compressed = OrderedDict()
        for name in f_names:
            compressed[name] = self.compressors[name](enhanced[name])  # [B, M+1, d]

        # Stage 3: 瓶颈融合
        f1_comp = compressed[f_names[0]]
        f2_comp = compressed[f_names[1]]
        bottleneck, f1_fused, f2_fused = self.fusion(f1_comp, f2_comp)

        # Stage 4: 分类
        # 取各模态最后一个 token (CLS)
        cls_f1 = f1_fused[:, -1, :]   # [B, d]
        cls_f2 = f2_fused[:, -1, :]   # [B, d]
        # Bottleneck 读出: 保留每个 bottleneck token 的通道信息
        btn_repr = bottleneck.flatten(1)  # [B, K * d]

        # 拼接并分类
        fused = torch.cat([cls_f1, cls_f2, btn_repr], dim=-1)  # [B, 2d + K*d]
        logits = self.classifier(fused)  # [B, 1]

        return logits, total_denoise_loss

    @classmethod
    def make_default(cls, inputs_dim: Dict[str, tuple], **kwargs) -> 'XXXMicro':
        return cls(inputs_dim=inputs_dim, **kwargs)


# ============================================================
# Skorch 训练包装器
# ============================================================
class XXXMicroNet(NeuralNetBinaryClassifier):
    """
    配合 skorch 训练框架的 XXXMicro 包装器。

    功能:
    - 自定义 get_loss: BCE分类损失 + lambda_recon * 去噪重构损失
    - 自定义 predict_proba: 返回 [N, 2] 概率分布
    - 自定义 evaluation_step: 正确处理多返回值
    """
    def __init__(self, *args, lambda_recon=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_recon = lambda_recon

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """多任务损失: 分类 + 去噪重构"""
        logits, loss_denoise = y_pred
        y_true = y_true.float().view(-1)

        # 分类损失 (BCE)
        loss_bce = super().get_loss(logits, y_true, *args, **kwargs)

        # 总损失
        total_loss = loss_bce + self.lambda_recon * loss_denoise

        return total_loss

    def predict_proba(self, X):
        """返回 [N, 2] 概率分布, 兼容 sklearn/skorch 接口"""
        non_probas = []
        for yp in self.forward_iter(X, training=False):
            if isinstance(yp, dict):
                logits = yp.get('y_pred', list(yp.values())[0])
            elif isinstance(yp, tuple):
                logits = yp[0]
            else:
                logits = yp
            p1 = torch.sigmoid(logits).view(-1, 1)
            non_probas.append(p1)
        
        p1_all = torch.cat(non_probas, dim=0).cpu().numpy()
        p0_all = 1 - p1_all
        return np.hstack([p0_all, p1_all])

    def evaluation_step(self, batch, training=False):
        """评估时正确处理 (logits, loss_denoise) 返回值"""
        X, y = batch
        with torch.set_grad_enabled(training):
            yp = self.infer(X)
            loss = self.get_loss(yp, y)
            return {'loss': loss, 'y_pred': yp[0]}
