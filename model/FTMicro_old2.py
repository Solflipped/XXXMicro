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
    UFEN: 单模态特征提取网络 (Unimodal Feature Extraction Network)，用于数据的特征增强
    Attention U-Net + Residual Learning
    """

    def __init__(self, 
                 n_num_features: int,
                 d_token: int = 128):

        super(UFEN, self).__init__()

        self.d_token = d_token
        self.n_num_features = n_num_features

        # Tokenizer
        self.tokenizer = NumericalFeatureTokenizer(
            n_features=n_num_features,
            d_token=d_token,
            bias=True,
            initialization='uniform'
        )

        # =================
        # Encoder
        # =================

        # conv1d_1 : 128 → 128
        self.conv1 = nn.Sequential(
            nn.Conv1d(128,128,3,padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )

        # conv1d_2 : 128 → 256
        self.conv2 = nn.Sequential(
            nn.Conv1d(128,256,3,padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )

        # conv1d_3 : 256 → 256
        self.conv3 = nn.Sequential(
            nn.Conv1d(256,256,3,padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )

        # =================
        # Bridge
        # =================

        # conv1d_4 : 256 → 512
        self.conv4 = nn.Sequential(
            nn.Conv1d(256,512,3,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )

        # conv1d_5 : 512 → 512
        self.conv5 = nn.Sequential(
            nn.Conv1d(512,512,3,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )

        # conv1d_6 : 512 → 256
        self.conv6 = nn.Sequential(
            nn.Conv1d(512,256,3,padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )

        # =================
        # Attention Gates
        # =================

        # conv3 ↔ conv6
        self.attention1 = attention_gate(
            F_g=256,
            F_l=256,
            F_int=128
        )

        # conv1 ↔ conv8
        self.attention2 = attention_gate(
            F_g=128,
            F_l=128,
            F_int=64
        )

        # =================
        # Decoder
        # =================

        # conv1d_7 : 512 → 256
        self.conv7 = nn.Sequential(
            nn.Conv1d(512,256,3,padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )

        # conv1d_8 : 256 → 128
        self.conv8 = nn.Sequential(
            nn.Conv1d(256,128,3,padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )

        # conv1d_9 : 256 → 128
        self.conv9 = nn.Sequential(
            nn.Conv1d(256,128,3,padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )

        # =================
        # Unimodal Predictor
        # =================

        self.unimodal_predictor = UnimodalPredictor(
            d_token,
            n_num_features
        )


    def forward(self, raw_x):

        # =================
        # Tokenization
        # =================

        x_tokens = self.tokenizer(raw_x)      # [B,N,128]
        x = x_tokens.transpose(1,2)           # [B,128,N]

        # =================
        # Encoder
        # =================

        conv1 = self.conv1(x)                 # [B,128,N]
        conv2 = self.conv2(conv1)             # [B,256,N]
        conv3 = self.conv3(conv2)             # [B,256,N]

        # =================
        # Bridge
        # =================

        conv4 = self.conv4(conv3)             # [B,512,N]
        conv5 = self.conv5(conv4)             # [B,512,N]
        conv6 = self.conv6(conv5)             # [B,256,N]

        # =================
        # Decoder stage 1
        # =================

        attn1 = self.attention1(conv6, conv3)

        x = torch.cat([conv6, attn1], dim=1)  # 256+256 = 512

        x = self.conv7(x)                     # [B,256,N]

        x = self.conv8(x)                     # [B,128,N]

        # =================
        # Decoder stage 2
        # =================

        attn2 = self.attention2(x, conv1)

        x = torch.cat([x, attn2], dim=1)      # 128+128 = 256

        feat_reconstructed = self.conv9(x)    # [B,128,N]

        # =================
        # Residual Learning
        # =================

        residual = x_tokens.transpose(1,2) - feat_reconstructed

        # =================
        # Unimodal prediction
        # =================

        y_i = self.unimodal_predictor(residual)

        # 输出给下游模型
        output_features = residual.transpose(1,2)

        return output_features, y_i

class HhyperLearningLayer(nn.Module):
    """
    AHL层：主模态引导辅助模态
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
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
            # 注意：这里的 Layer 内部实现应为您 FTMicro.py 中定义的双模态版本
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
        inner_dim = dim_head *  heads
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

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
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
        self.norm_hyper = nn.LayerNorm(dim) # 超潜在表示规范化
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
        self.n_species = args.n_species
        self.n_ko = args.n_ko
        self.d_token = args.d_token
        self.dst_embedding_length_species = self.dst_embedding_length_ko = args.dst_embedding_length
        self.AHL_depth = args.AHL_depth
        self.h_hyper_param = nn.Parameter(torch.ones(1, self.dst_embedding_length_species, args.d_token))
        
        # 1. 单模态特征增强层
        # config.n_species: 物种数量, config.n_ko: KO基因数量, config.d_token: 映射维度(如128)
        self.species_ufen = UFEN(n_num_features=args.n_species, d_token=args.d_token)
        self.ko_ufen = UFEN(n_num_features=args.n_ko, d_token=args.d_token)

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
            dim_head=int(args.d_token / 8)
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
        feat_s, y_s = self.species_ufen(species_raw)
        # feat_ko: [B, n_ko, D], y_ko: [B, 1] (KO单模态预测结果)
        feat_ko, y_ko = self.ko_ufen(ko_raw)

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
        return output, y_s, y_ko
    
    @classmethod
    def make_default(
        cls,
        n_species: int,
        n_ko: int,
        d_token: int = 128,
        dst_embedding_length: int = 8, 
        AHL_depth: int = 3,             
        fusion_depth: int = 2,
        **kwargs
    ) -> 'FTMicro':
        class FTMicroArgs:
            pass
        
        args = FTMicroArgs()
        args.n_species = n_species
        args.n_ko = n_ko
        args.d_token = d_token
        args.dst_embedding_length = dst_embedding_length 
        args.AHL_depth = AHL_depth                       
        args.fusion_depth = fusion_depth
        
        return cls(args)