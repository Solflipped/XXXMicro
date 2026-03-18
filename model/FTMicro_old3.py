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
    def __init__(self, d_token, n_features):
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
                 d_token: int = 128,        # 输入通道数
                 base_channels: int = 128,  # 基础通道数
                 expansion_factor=2,       # 每次扩展的倍数
                 num_layers=4,             # 编码器/解码器层数
                 latent_dim=512):          # 潜在表示维度
        super(UFEN, self).__init__()

        self.d_token = d_token
        self.n_num_features = n_num_features
        self.base_channels = base_channels
        self.num_layers = num_layers
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
        # residual = x - feat_reconstructed
        residual = feat_reconstructed
        
        # 1. 计算单模态预测结果 y_i: [batch_size, 1]
        y_i = self.unimodal_predictor(residual)
        
        # 2. 将增强后的特征转置回 [batch_size, n_num_features, d_token] 供下游多模态融合
        output_features = residual.transpose(1, 2)

        return output_features, y_i
    
    @classmethod
    def make_default(
        cls, # 类本身（UFEN）
        n_num_features: int,     # 原始特征维度
        d_token: int = 128,      # 输入通道数
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

class HhyperLearningLayer(nn.Module):
    """
    AHL层：主模态引导辅助模态
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads # 内部维度 = 每头维度 * 头数  最终还是等于d_token
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)     # 主模态的Q
        self.to_k_aux = nn.Linear(dim, inner_dim, bias=False) # 辅助模态的K
        self.to_v_aux = nn.Linear(dim, inner_dim, bias=False) # 辅助模态的V

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, h_main, h_aux, h_hyper):
        b, n, d = h_main.shape
        h = self.heads

        q = self.to_q(h_main)
        k = self.to_k_aux(h_aux)
        v = self.to_v_aux(h_aux)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # 计算主模态对辅助模态的引导注意力
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        
        # 提取被引导后的辅助模态信息
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # 更新超模态表示 (Hyper-modality)
        # 注意：这里需要根据N_main和N_hyper做对齐，通常让h_hyper与主模态长度一致
        h_hyper_update = self.to_out(out)
        h_hyper = h_hyper + h_hyper_update
        return h_hyper



class HhyperLearningEncoder(nn.Module):
    """
    适配 FTMicro 的 AHL 编码器容器
    """
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAHL(dim, HhyperLearningLayer(dim, heads = heads, dim_head = dim_head, dropout = dropout))
            ]))

    def forward(self, h_main_list, h_aux, h_hyper):
        # 自动遍历主模态的多尺度特征列表进行引导学习
        for i, attn in enumerate(self.layers):
            # h_main_list[i] 是主模态（物种）在第 i 层的特征
            h_hyper = attn[0](h_main_list[i], h_aux, h_hyper)
        return h_hyper
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))  # 维度变换
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class PreNormAHL(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_main = nn.LayerNorm(dim)  # 主模态规范化
        self.norm_aux = nn.LayerNorm(dim)   # 辅助模态规范化
        self.norm_hyper = nn.LayerNorm(dim) # 超模态潜在表示规范化
        self.fn = fn

    def forward(self, h_main, h_aux, h_hyper):
        # 在执行 self.fn (即 HhyperLearningLayer) 前进行标准化
        return self.fn(
            self.norm_main(h_main), 
            self.norm_aux(h_aux), 
            self.norm_hyper(h_hyper)
        )

class PreNormForward(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x


class Transformer(nn.Module):
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
             self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
             self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n+self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)

        return x


class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        extra_token = repeat(self.extra_token, '1 1 d -> b 1 d', b = b)

        source_x = torch.cat((extra_token, source_x), dim=1)
        source_x = source_x + self.pos_embedding_s[:, : n_s+1]

        target_x = torch.cat((extra_token, target_x), dim=1)
        target_x = target_x + self.pos_embedding_t[:, : n_t+1]

        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t


class FTMicro(nn.Module):
    """
    FTMicro：(Species)为主模态，引导KO功能基因
    """
    def __init__(self, args):
        super().__init__()

        # 保存参数供后续使用
        self.batch_size = args.batch_size
        self.n_species = args.n_species
        self.n_ko = args.n_ko
        self.d_token = args.d_token
        self.dst_embedding_length_species = self.dst_embedding_length_ko = args.dst_embedding_length
        self.AHL_depth = args.AHL_depth
        self.h_hyper_param = nn.Parameter(torch.ones(1, self.dst_embedding_length_species, args.d_token))
        
        
       
        # 1. 单模态特征增强层
        # config.n_species: 物种数量, config.n_ko: KO基因数量, config.d_token: 映射维度(如128)
        # self.species_ufen = UFEN(n_num_features=args.n_species, d_token=args.d_token)
        # self.ko_ufen = UFEN(n_num_features=args.n_ko, d_token=args.d_token)

         # 使用FT-Transformer的NumericalFeatureTokenizer将数值特征转换为token序列
        self.tokenizer_species = NumericalFeatureTokenizer(n_features=self.n_species, d_token=self.d_token, bias=True, initialization='uniform')
        self.tokenizer_ko = NumericalFeatureTokenizer(n_features=self.n_ko, d_token=self.d_token, bias=True, initialization='uniform')

        # 2. 特征嵌入 （ Modality Embedding ）
        self.embedding_species = Transformer(
            num_frames=args.n_species, 
            token_len=self.dst_embedding_length_species, 
            save_hidden=False, 
            dim=args.d_token, 
            depth=1, 
            heads=8, 
            mlp_dim=args.d_token,
        )

        self.embedding_ko = Transformer(
            num_frames=args.n_ko, 
            token_len=self.dst_embedding_length_ko, 
            save_hidden=False, 
            dim=args.d_token, 
            depth=1,
            heads=8, 
            mlp_dim=args.d_token
        )

        # 这里使用原代码中的 Transformer 结构，通过 save_hidden=True 获取各层输出
        self.main_encoder = Transformer(
            num_frames=self.dst_embedding_length_species,  
            save_hidden=True, 
            token_len=None, 
            dim=args.d_token, 
            depth=self.AHL_depth-1,
            heads=8, 
            mlp_dim=args.d_token
        )

        # 3. AHL 模块 (Species 引导 KO)
        # 初始化一个可学习的超模态 Token
        self.ahl_encoder = HhyperLearningEncoder(
            dim=args.d_token,
            depth=self.AHL_depth,
            heads=8,
            dim_head=(args.d_token // 8), # 每个头的维度，通常设置为 d_token // heads
            dropout = 0
        )

        # 4. 多模态融合 (Cross-modality Fusion)
        self.fusion_layer = CrossTransformer(
            source_num_frames = self.dst_embedding_length_species, 
            tgt_num_frames = self.dst_embedding_length_species,
            dim = args.d_token, 
            depth = args.fusion_depth, 
            heads=8, 
            mlp_dim = args.d_token
        )

        # 5. 预测头
        self.classifier = nn.Sequential(
            nn.LayerNorm(args.d_token),
            nn.Linear(args.d_token, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 假设是二分类或回归
        )

    def forward(self, species_raw, ko_raw):
        # B: Batch Size
        b = species_raw.size(0)
        # 初始化 h_hyper
        h_hyper = repeat(self.h_hyper_param, '1 n d -> b n d', b = b)

        # Step 1: UFEN 特征增强
        # feat_s: [B, n_species, D], y_s: [B, 1] (物种单模态预测结果)
        #feat_s, y_s = self.species_ufen(species_raw)
        # feat_ko: [B, n_ko, D], y_ko: [B, 1] (KO单模态预测结果)
        # feat_ko, y_ko = self.ko_ufen(ko_raw)
       
        feat_s = self.tokenizer_species(species_raw)  # (batch_size, n_num_features, d_token)
        feat_ko = self.tokenizer_ko(ko_raw)  # (batch_size, n_num_features, d_token)

        # Step 2: 特征嵌入
        h_s = self.embedding_species(feat_s)[:, :self.dst_embedding_length_species] # [B, 8, D]
        h_ko = self.embedding_ko(feat_ko)[:, :self.dst_embedding_length_ko] # [B, 8, D]


        # Step 3: 获取主模态的多尺度特征
        h_main_list = self.main_encoder(h_s)

        # Step 4: AHL 引导学习
        h_hyper = self.ahl_encoder(h_main_list, h_ko, h_hyper)

        # Step 5: 最终融合
        # 使用主模态最后一个尺度的特征作为target（包含CLS）
        # 使用超模态特征作为source
        fusion = self.fusion_layer(
            source_x=h_hyper,        
            target_x=h_main_list[-1]   
        )[:, 0]

        # Step 6: 分类
        output = self.classifier(fusion)
        
        # 返回主输出以及两个单模态的辅助预测结果
        return output
    
    @classmethod
    def make_default(
        cls,
        batch_size: int,
        n_species: int,
        n_ko: int,
        d_token: int = 128,
        dst_embedding_length: int = 8, 
        AHL_depth: int = 3,             
        fusion_depth: int = 4,
        **kwargs
    ) -> 'FTMicro':
        class FTMicroArgs:
            pass
        
        args = FTMicroArgs()
        args.batch_size = batch_size
        args.n_species = n_species
        args.n_ko = n_ko
        args.d_token = d_token
        args.dst_embedding_length = dst_embedding_length 
        args.AHL_depth = AHL_depth                       
        args.fusion_depth = fusion_depth
        
        return cls(args)