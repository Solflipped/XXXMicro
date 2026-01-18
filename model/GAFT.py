import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MBT import MBT   
from model.FT_transformer import FeatureTokenizer, CLSToken
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_


class SubNet(nn.Module):
    """
    Pre-fusion subnetwork for modality-specific processing.
    
    Uses BatchNorm → Dropout → 3-layer MLP with ReLU.
    """

    def __init__(self, in_size, hidden_size, dropout):
        """
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        """
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch_size, in_size)
        Returns:
            tensor of shape (batch_size, hidden_size)
        """
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))
        return y_3


class DualModalityLMF(nn.Module):
    """
    双模态低阶多模态融合：进一步建模species+ko模态的信息交互，利用张量分解对模态相互作用进行高效建模。
    
    融合公式:
        Z = Σ_r (W_r ⊙ [species_h; 1] ⊙ [ko_h; 1])
    """

    def __init__(
        self,
        species_in: int,
        ko_in: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        rank: int = 4,
        dropout: float = 0.1,
        use_subnet: bool = True
    ):
        """
        Args:
            species_in: species embedding dimension (e.g., d_token from MBT)
            ko_in: ko embedding dimension (e.g., d_token from MBT)
            hidden_dim: hidden dimension for SubNet processing
            output_dim: output fusion dimension
            rank: rank for low-rank factorization (controls fusion complexity)
            dropout: dropout probability
            use_subnet: whether to use SubNet for pre-processing (if False, direct fusion)
        """
        super(DualModalityLMF, self).__init__()
        
        self.species_in = species_in
        self.ko_in = ko_in
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rank = rank
        self.use_subnet = use_subnet
        
        # Pre-fusion subnetworks (optional)
        if use_subnet:
            self.species_subnet = SubNet(species_in, hidden_dim, dropout)
            self.ko_subnet = SubNet(ko_in, hidden_dim, dropout)
            species_factor_dim = hidden_dim + 1  # +1 for bias term
            ko_factor_dim = hidden_dim + 1
        else:
            self.species_subnet = None
            self.ko_subnet = None
            species_factor_dim = species_in + 1
            ko_factor_dim = ko_in + 1
        
        # Low-rank fusion factors
        # Each factor: [rank, modality_dim + 1, output_dim]
        self.species_factor = Parameter(torch.Tensor(rank, species_factor_dim, output_dim))
        self.ko_factor = Parameter(torch.Tensor(rank, ko_factor_dim, output_dim))
        
        # Fusion weights and bias
        self.fusion_weights = Parameter(torch.Tensor(1, rank))
        self.fusion_bias = Parameter(torch.Tensor(1, output_dim))
        
        # Post-fusion dropout
        self.post_fusion_dropout = nn.Dropout(p=dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize fusion factors with Xavier normal."""
        xavier_normal_(self.species_factor)
        xavier_normal_(self.ko_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
    
    def forward(self, species_emb, ko_emb):
        """
        Forward pass for dual-modality fusion.
        
        Args:
            species_emb: species CLS embedding, shape (batch_size, species_in)
            ko_emb: ko CLS embedding, shape (batch_size, ko_in)
        
        Returns:
            fused: fused representation, shape (batch_size, output_dim)
        """
        batch_size = species_emb.size(0)
        
        # Optional pre-processing with SubNet
        if self.use_subnet:
            species_h = self.species_subnet(species_emb)  # [B, hidden_dim]
            ko_h = self.ko_subnet(ko_emb)                 # [B, hidden_dim]
        else:
            species_h = species_emb
            ko_h = ko_emb
        
        # Append bias term (constant 1)
        ones = torch.ones(batch_size, 1, dtype=species_h.dtype, device=species_h.device)
        _species_h = torch.cat([ones, species_h], dim=1)  # [B, species_dim + 1]
        _ko_h = torch.cat([ones, ko_h], dim=1)            # [B, ko_dim + 1]
        
        # Low-rank multimodal fusion
        # fusion_species: [B, rank, output_dim]
        fusion_species = torch.matmul(_species_h, self.species_factor)
        # fusion_ko: [B, rank, output_dim]
        fusion_ko = torch.matmul(_ko_h, self.ko_factor)
        
        # Element-wise product (Hadamard product)
        fusion_zy = fusion_species * fusion_ko  # [B, rank, output_dim]
        
        # Weighted sum over rank dimension + bias
        # fusion_weights: [1, rank] → broadcast to [B, rank]
        # fusion_zy.permute(1, 0, 2): [rank, B, output_dim]
        # output: [B, output_dim]
        output = torch.matmul(
            self.fusion_weights, 
            fusion_zy.permute(1, 0, 2)
        ).squeeze(0) + self.fusion_bias
        
        # Post-fusion dropout
        output = self.post_fusion_dropout(output)
        
        return output

class GATLayer(nn.Module):
    """
    GAT（全连接）+ dropout + 残差 + LayerNorm
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        attn_dropout=0.1,
        proj_dropout=0.1,
        attn_act_slope=0.2,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.attn_act_slope = attn_act_slope

        # projection: in_dim → out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        # attention scoring: 2*out_dim → 1
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)

        # dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # residual + LayerNorm
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h):
        """
        h: [N, in_dim]
        return: [N, out_dim]
        """
        # 确保输入是 2D
        if h.dim() > 2:
            h = h.squeeze()
        
        N = h.size(0)

        # 投影: [N, in_dim] → [N, out_dim]
        Wh = self.W(h)
        
        # 确保 Wh 是 2D
        if Wh.dim() > 2:
            Wh = Wh.view(N, -1)

        # 构造 pairwise attention
        # Wh_i: [N, out_dim] → [N, 1, out_dim] → [N, N, out_dim]
        # Wh_j: [N, out_dim] → [1, N, out_dim] → [N, N, out_dim]
        Wh_i = Wh.unsqueeze(1).expand(N, N, self.out_dim)
        Wh_j = Wh.unsqueeze(0).expand(N, N, self.out_dim)

        # 拼接后计算注意力分数: [N, N, 2*out_dim] → [N, N, 1] → [N, N]
        e = self.attn(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1)
        e = F.leaky_relu(e, negative_slope=self.attn_act_slope)

        # softmax 归一化（对每个节点的所有邻居）
        alpha = torch.softmax(e, dim=1)  # [N, N]
        alpha = self.attn_dropout(alpha)

        # 聚合: alpha [N, N] × Wh [N, out_dim] → [N, out_dim]
        h_new = torch.matmul(alpha, Wh)
        h_new = self.proj_dropout(h_new)

        # residual + LayerNorm
        return self.norm(self.res_proj(h) + h_new)


class GAFT(nn.Module):
    """
        species / ko
            ↓
        MBT (CLS embeddings)
            ↓
        Low-rank Multimodal Fusion (LMF)
            ↓
        GAT × 2 
            ↓
        Linear 输出 (AD/NC)
    """

    def __init__(self, mbt_config, lmf_hidden_dim=128, lmf_output_dim=128, lmf_rank=4, 
                 lmf_dropout=0.1, use_lmf_subnet=True, gat_dim=128, 
                 gat_dropout=0.1, finetune_mbt=False):
        super().__init__()

        # 1. 创建 MBT
        self.mbt = mbt_config if isinstance(mbt_config, MBT) else MBT(**mbt_config)
        self.finetune_mbt = finetune_mbt
        if not self.finetune_mbt:
            for p in self.mbt.parameters():
                p.requires_grad = False

        # 支持 dict 配置或直接传入 MBT 实例
        d_token = mbt_config["dim"] if isinstance(mbt_config, dict) else mbt_config.dim

        # 2. Low-rank Multimodal Fusion
        self.lmf = DualModalityLMF(
            species_in=d_token,
            ko_in=d_token,
            hidden_dim=lmf_hidden_dim,
            output_dim=lmf_output_dim,
            rank=lmf_rank,
            dropout=lmf_dropout,
            use_subnet=use_lmf_subnet
        )

        # 3. GAT layers (输入维度改为 lmf_output_dim)
        self.gat1 = GATLayer(
            in_dim=lmf_output_dim, 
            out_dim=gat_dim, 
            attn_dropout=gat_dropout, 
            proj_dropout=gat_dropout
        )
        self.gat2 = GATLayer(
            in_dim=gat_dim,
            out_dim=gat_dim,
            attn_dropout=gat_dropout, 
            proj_dropout=gat_dropout
        )

        # 4. 输出层
        self.out = nn.Linear(gat_dim, 1)


    def get_mbt_embedding(self, species, ko):
        """
        返回：
          cls_species: [N, d_token]
          cls_ko: [N, d_token]
        """
        if self.finetune_mbt:
            x_tokens = self.mbt.tokenize_inputs({
                "species": species,
                "ko": ko
            })
            B = species.size(0)
            bottleneck = self.mbt.bottleneck.expand(B, -1, -1)
            encoded = self.mbt.encoder(x_tokens, bottleneck)

            # 提取 CLS
            offset = 0
            cls_list = []
            for modality in ["species", "ko"]:
                Lm = x_tokens[modality].size(1)
                cls_list.append(encoded[:, offset + (Lm - 1), :])
                offset += Lm

            cls_species, cls_ko = cls_list
        else:
            with torch.no_grad():
                x_tokens = self.mbt.tokenize_inputs({
                    "species": species,
                    "ko": ko
                })
                B = species.size(0)
                bottleneck = self.mbt.bottleneck.expand(B, -1, -1)
                encoded = self.mbt.encoder(x_tokens, bottleneck)

                # 提取 CLS
                offset = 0
                cls_list = []
                for modality in ["species", "ko"]:
                    Lm = x_tokens[modality].size(1)
                    cls_list.append(encoded[:, offset + (Lm - 1), :])
                    offset += Lm

                cls_species, cls_ko = cls_list

        return cls_species, cls_ko

   

    def forward(self, species, ko):
        """
        species: [N, D1]
        ko:      [N, D2]
        返回: logits [N]
        """
        # 1. MBT embedding（可选 finetune）
        cls_species, cls_ko = self.get_mbt_embedding(species, ko)  # 各 [N, d_token]

        # 2. Low-rank Multimodal Fusion (建模双模态交互)
        h = self.lmf(cls_species, cls_ko)  # [N, lmf_output_dim]

        # 3. Graph Attention Network (样本间关系建模)
        h = F.elu(self.gat1(h))          # [N, gat_dim]
        h = F.elu(self.gat2(h))          # [N, gat_dim]

        # 4. 输出
        logits = self.out(h)              # [N, 1]

        return logits
