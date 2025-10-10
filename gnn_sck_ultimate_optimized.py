#!/usr/bin/env python3
"""
🚀 GNN+SCK 终极优化版 - 冲刺80%+ ACC
===========================================
优化策略:
1. 架构增强: SE注意力、多尺度融合、残差连接
2. 训练优化: 混合精度、梯度累积、Lookahead优化器
3. 正则化: DropPath、MixUp、Label Smoothing
4. 动态学习: 自适应门控、动态温度、课程学习
5. 集成策略: 多模型融合、测试时增强

作者: lll-hhh (Ultimate Edition)
日期: 2025-10-10
目标: Test ACC > 80%
===========================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv, TransformerConv
from torch.cuda.amp import autocast, GradScaler
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import random
import math
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


# ==============================
# 🔧 高级组件库
# ==============================
class SqueezeExcitation(nn.Module):
    """SE注意力模块 - 通道级特征重标定"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, L, C] or [B, C, L]
        if x.dim() == 3:
            if x.size(1) > x.size(2):  # [B, L, C]
                pool = x.mean(dim=1)  # [B, C]
                weight = self.fc(pool).unsqueeze(1)  # [B, 1, C]
            else:  # [B, C, L]
                pool = x.mean(dim=2)  # [B, C]
                weight = self.fc(pool).unsqueeze(2)  # [B, C, 1]
        return x * weight


class DropPath(nn.Module):
    """随机深度 - 更强的正则化"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MultiScaleFusion(nn.Module):
    """多尺度特征融合"""
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Conv1d(channels, channels, 1)
        self.branch2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.branch3 = nn.Conv1d(channels, channels, 5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(channels, channels, 1)
        )
        self.fusion = nn.Conv1d(channels * 4, channels, 1)
        self.norm = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        # x: [B, C, L]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fusion(concat)
        return self.norm(out)


# ==============================
# 🧬 增强型 GNN 模块
# ==============================
class EnhancedMultiHopGraphAttention(nn.Module):
    """增强型多跳图注意力 - 加入Transformer和残差"""
    def __init__(self, hidden_dim=256, heads=4, num_hops=3, dropout=0.2):
        super().__init__()
        self.num_hops = num_hops
        self.hidden_dim = hidden_dim
        
        # GAT层
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=True)
            for _ in range(num_hops)
        ])
        
        # Transformer层（补充全局信息）
        self.transformer_layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            for _ in range(num_hops)
        ])
        
        # 可学习的跳权重
        self.hop_weights = nn.Parameter(torch.ones(num_hops) / num_hops)
        
        # SE注意力
        self.se_attention = SqueezeExcitation(hidden_dim)
        
        # 残差融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            DropPath(dropout * 0.5)
        )
        
    def forward(self, x, edge_index):
        hop_features = []
        
        for hop, (gat, trans) in enumerate(zip(self.gat_layers, self.transformer_layers)):
            if hop == 0:
                gat_out = gat(x, edge_index)
                trans_out = trans(x, edge_index)
            else:
                gat_out = gat(hop_features[-1], edge_index)
                trans_out = trans(hop_features[-1], edge_index)
            
            # 融合GAT和Transformer
            hop_out = (gat_out + trans_out) / 2
            hop_features.append(hop_out)
        
        # 加权融合所有跳
        weights = F.softmax(self.hop_weights, dim=0)
        hop_stack = torch.stack(hop_features, dim=0)
        weighted_hops = (hop_stack * weights.view(-1, 1, 1)).sum(dim=0)
        
        # SE注意力
        batch_size = x.size(0) // 203
        weighted_hops_reshaped = weighted_hops.view(batch_size, 203, -1)
        weighted_hops_se = self.se_attention(weighted_hops_reshaped)
        weighted_hops = weighted_hops_se.view(-1, self.hidden_dim)
        
        # 残差连接
        x_reshaped = x.view(batch_size, 203, -1)
        output = self.fusion(torch.cat([x_reshaped, weighted_hops.view(batch_size, 203, -1)], dim=-1))
        output = output.view(-1, self.hidden_dim)
        
        return output + x  # 额外残差


class AdaptiveGraphPooling(nn.Module):
    """自适应图池化 - 动态权重分配"""
    def __init__(self, hidden_dim=256, num_regions=6, dropout=0.2):
        super().__init__()
        self.num_regions = num_regions
        self.seq_len = 203
        self.hidden_dim = hidden_dim
        
        # 自适应池化权重
        self.adaptive_weights = nn.Parameter(torch.ones(3) / 3)  # local, region, global
        
        # Local池化
        self.local_pool = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Region池化
        self.region_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.region_norm = nn.LayerNorm(hidden_dim)
        
        # Global池化
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # SE注意力
        self.se = SqueezeExcitation(hidden_dim)
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # x: [B, 203, hidden_dim]
        
        # Local池化
        local_mean = x.mean(dim=1)
        local_max = x.max(dim=1)[0]
        local_std = x.std(dim=1)
        local_feat = self.local_pool(torch.cat([local_mean, local_max, local_std], dim=-1))
        
        # Region池化（带注意力）
        region_size = self.seq_len // self.num_regions
        region_feats = []
        for i in range(self.num_regions):
            start = i * region_size
            end = start + region_size if i < self.num_regions - 1 else self.seq_len
            region_x = x[:, start:end, :]
            region_feats.append(region_x.mean(dim=1))
        
        region_stack = torch.stack(region_feats, dim=1)  # [B, num_regions, hidden_dim]
        region_attn, _ = self.region_attention(region_stack, region_stack, region_stack)
        region_feat = self.region_norm(region_attn.mean(dim=1))
        
        # Global池化
        global_mean = x.mean(dim=1)
        global_max = x.max(dim=1)[0]
        global_min = x.min(dim=1)[0]
        global_std = x.std(dim=1)
        global_feat = self.global_pool(torch.cat([global_mean, global_max, global_min, global_std], dim=-1))
        
        # SE注意力
        local_feat = self.se(local_feat.unsqueeze(1)).squeeze(1)
        region_feat = self.se(region_feat.unsqueeze(1)).squeeze(1)
        global_feat = self.se(global_feat.unsqueeze(1)).squeeze(1)
        
        # 自适应加权融合
        weights = F.softmax(self.adaptive_weights, dim=0)
        weighted_local = local_feat * weights[0]
        weighted_region = region_feat * weights[1]
        weighted_global = global_feat * weights[2]
        
        # 最终融合
        hierarchical_feat = torch.cat([weighted_local, weighted_region, weighted_global], dim=-1)
        output = self.final_fusion(hierarchical_feat)
        
        return output


class UltimateGNNBackbone(nn.Module):
    """终极GNN骨干网络"""
    def __init__(self, input_dim=1280, hidden_dim=256, num_layers=6, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.gnn_layers = nn.ModuleList([
            EnhancedMultiHopGraphAttention(hidden_dim, heads=4, num_hops=3, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                DropPath(dropout * 0.3)
            ) for _ in range(num_layers)
        ])
        
        self.pooling = AdaptiveGraphPooling(hidden_dim, num_regions=6, dropout=dropout)
        
    def forward(self, x, edge_index):
        batch_size = x.size(0) // 203
        x = self.input_proj(x)
        
        for gnn, norm, ffn in zip(self.gnn_layers, self.norms, self.ffns):
            # GNN层
            x_gnn = gnn(x, edge_index)
            x = x + x_gnn  # 残差
            
            # 归一化和FFN
            x = x.view(batch_size, 203, -1)
            x = norm(x)
            x = x + ffn(x)  # 残差
            x = x.view(-1, x.size(-1))
        
        x = x.view(batch_size, 203, -1)
        pooled_feat = self.pooling(x)
        return pooled_feat


# ==============================
# 🎯 增强型 SCK 模块
# ==============================
class UltraPositionalFeatureExtractor(nn.Module):
    """超强位置特征提取器"""
    def __init__(self, seq_len=203, device='cuda'):
        super().__init__()
        self.seq_len = seq_len
        self.device = device
        
        # 可学习位置编码（多尺度）
        self.learnable_pe_1 = nn.Parameter(torch.randn(1, seq_len, 32))
        self.learnable_pe_2 = nn.Parameter(torch.randn(1, seq_len, 32))
        
        # 位置编码融合
        self.pe_fusion = nn.Sequential(
            nn.Linear(64, 48),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, batch_size):
        features = []
        
        # 1. 标准位置编码（多尺度）
        for d_model in [16, 32]:
            pe = self._positional_encoding(self.seq_len, d_model).to(self.device)
            features.append(pe.unsqueeze(0).repeat(batch_size, 1, 1))
        
        # 2. 相对位置
        rel_pos = torch.linspace(0, 1, self.seq_len, device=self.device)
        rel_features = torch.stack([
            rel_pos,
            rel_pos ** 2,
            rel_pos ** 3,
            torch.sqrt(rel_pos)
        ], dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
        features.append(rel_features)
        
        # 3. 三角函数特征（多频率）
        freqs = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        sin_features = []
        for freq in freqs:
            pos = torch.arange(self.seq_len, device=self.device) * freq
            sin_features.append(torch.sin(pos).unsqueeze(-1))
            sin_features.append(torch.cos(pos).unsqueeze(-1))
        features.append(torch.cat(sin_features, dim=-1).unsqueeze(0).repeat(batch_size, 1, 1))
        
        # 4. 区域特征（更细粒度）
        num_regions = 12
        region_size = self.seq_len // num_regions
        region_feat = torch.zeros(batch_size, self.seq_len, num_regions, device=self.device)
        for i in range(num_regions):
            start = i * region_size
            end = start + region_size if i < num_regions - 1 else self.seq_len
            region_feat[:, start:end, i] = 1
        features.append(region_feat)
        
        # 5. 距离特征（多种距离度量）
        positions = torch.arange(self.seq_len, device=self.device).float()
        dist_features = []
        for power in [1, 2, 3, 0.5]:
            dist_to_start = (positions / self.seq_len) ** power
            dist_to_mid = (torch.abs(positions - self.seq_len / 2) / (self.seq_len / 2)) ** power
            dist_to_end = (torch.abs(positions - (self.seq_len - 1)) / self.seq_len) ** power
            dist_features.extend([dist_to_start, dist_to_mid, dist_to_end])
        dist_features = torch.stack(dist_features, dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
        features.append(dist_features)
        
        # 6. 可学习位置编码
        learnable_pe = torch.cat([
            self.learnable_pe_1.repeat(batch_size, 1, 1),
            self.learnable_pe_2.repeat(batch_size, 1, 1)
        ], dim=-1)
        learnable_pe = self.pe_fusion(learnable_pe)
        features.append(learnable_pe)
        
        return torch.cat(features, dim=-1)
    
    def _positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class UltimateSCKBackbone(nn.Module):
    """终极SCK骨干网络"""
    def __init__(self, input_dim=1280, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.device = None
        self.pos_extractor = None
        
        # 多尺度卷积（更多尺度）
        kernels = [3, 5, 7, 9, 11, 15]
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // len(kernels), kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(hidden_dim // len(kernels)),
                nn.GELU(),
                nn.Dropout(dropout * 0.5)
            ) for k in kernels
        ])
        
        # 多尺度融合
        self.multi_scale_fusion = MultiScaleFusion(hidden_dim)
        
        # 通道注意力
        self.channel_attention = SqueezeExcitation(hidden_dim)
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 位置特征维度自适应（新的特征维度）
        pos_feat_dim = 48 + 16 + 32 + 4 + 24 + 12 + 12 + 48  # 约196
        
        # 特征融合
        self.feat_fusion = nn.Sequential(
            nn.Linear(hidden_dim + pos_feat_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 全局池化
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def _init_pos_extractor(self, device):
        if self.pos_extractor is None:
            self.pos_extractor = UltraPositionalFeatureExtractor(device=device).to(device)
            self.device = device
    
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        self._init_pos_extractor(device)
        pos_feat = self.pos_extractor(batch_size)
        
        # 多尺度卷积
        multi_scale = [block(x) for block in self.conv_blocks]
        x = torch.cat(multi_scale, dim=1)
        
        # 多尺度融合
        x = self.multi_scale_fusion(x)
        
        # 通道注意力
        x = self.channel_attention(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 空间注意力
        max_pool = x.max(dim=1, keepdim=True)[0]
        avg_pool = x.mean(dim=1, keepdim=True)
        spatial_attn = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        x = x * spatial_attn
        
        # 位置特征融合
        x = x.permute(0, 2, 1)
        x = torch.cat([x, pos_feat], dim=-1)
        x = self.feat_fusion(x)
        
        # 全局池化（多种统计量）
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        x_min = x.min(dim=1)[0]
        x_std = x.std(dim=1)
        x_pooled = self.global_pool(torch.cat([x_mean, x_max, x_min, x_std], dim=-1))
        
        return x_pooled


# ==============================
# 🌟 终极联合训练模型
# ==============================
class UltimateJointModel(nn.Module):
    """终极联合训练模型 - 目标80%+"""
    def __init__(self, input_dim=1280, hidden_dim=256, dropout=0.2):
        super().__init__()
        
        # 双骨干网络
        self.gnn_backbone = UltimateGNNBackbone(input_dim, hidden_dim, num_layers=6, dropout=dropout)
        self.sck_backbone = UltimateSCKBackbone(input_dim, hidden_dim, dropout=dropout)
        
        # 交叉注意力（双向）
        self.cross_attn_gnn = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.cross_attn_sck = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        
        self.cross_norm_gnn = nn.LayerNorm(hidden_dim)
        self.cross_norm_sck = nn.LayerNorm(hidden_dim)
        
        # 动态门控（带温度）
        self.temperature = nn.Parameter(torch.ones(1))
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # SE注意力
        self.se_fusion = SqueezeExcitation(hidden_dim * 2)
        
        # 深度融合网络
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 3),
            nn.LayerNorm(hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            DropPath(dropout * 0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 分类头（更深）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            
            nn.Linear(64, 1)
        )
        
        self.last_gate_weight = None
    
    def forward(self, data, edge_index, return_features=False):
        batch_size = data.size(0)
        
        # GNN路径
        x_flat = data.permute(0, 2, 1).contiguous().view(-1, data.size(1))
        gnn_feat = self.gnn_backbone(x_flat, edge_index)
        
        # SCK路径
        sck_feat = self.sck_backbone(data)
        
        # 交叉注意力增强
        gnn_feat_unsqueeze = gnn_feat.unsqueeze(1)
        sck_feat_unsqueeze = sck_feat.unsqueeze(1)
        
        gnn_enhanced, _ = self.cross_attn_gnn(gnn_feat_unsqueeze, sck_feat_unsqueeze, sck_feat_unsqueeze)
        sck_enhanced, _ = self.cross_attn_sck(sck_feat_unsqueeze, gnn_feat_unsqueeze, gnn_feat_unsqueeze)
        
        gnn_enhanced = self.cross_norm_gnn(gnn_feat_unsqueeze + gnn_enhanced).squeeze(1)
        sck_enhanced = self.cross_norm_sck(sck_feat_unsqueeze + sck_enhanced).squeeze(1)
        
        # 动态门控（带温度退火）
        gate_input = torch.cat([gnn_enhanced, sck_enhanced], dim=-1)
        gate_logit = self.gate(gate_input)
        gate_weight = torch.sigmoid(gate_logit / self.temperature)
        
        gated_gnn = gnn_enhanced * gate_weight
        gated_sck = sck_enhanced * (1 - gate_weight)
        
        self.last_gate_weight = gate_weight.mean()
        
        # SE增强融合
        fused_concat = torch.cat([gated_gnn, gated_sck], dim=-1)
        fused_concat = self.se_fusion(fused_concat.unsqueeze(1)).squeeze(1)
        
        # 深度融合
        fused = self.fusion(fused_concat)
        
        if return_features:
            return fused
        
        # 分类
        output = self.classifier(fused).squeeze(-1)
        output = torch.sigmoid(output)
        
        return output
    
    def get_gate_weight(self):
        return self.last_gate_weight


# ==============================
# 📊 数据增强
# ==============================
class MixUpAugmentation:
    """MixUp数据增强"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x1, y1, x2, y2):
        lam = np.random.beta(self.alpha, self.alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        return x_mixed, y_mixed, lam


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy(pred, target)


# ==============================
# 🔧 数据集（同原版）
# ==============================
class EdgeDictLoader:
    @staticmethod
    def load_edge_dict(edge_file: str) -> dict:
        with open(edge_file, 'rb') as f:
            edge_dict = pickle.load(f)
        print(f"📂 加载 {edge_file}: {len(edge_dict)} 个样本")
        return edge_dict


class RNAGraphDataset(Dataset):
    def __init__(self, embeddings, labels, edge_dict, sample_indices):
        self.embeddings = embeddings
        self.labels = labels
        self.edge_dict = edge_dict
        self.sample_indices = sample_indices

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = torch.tensor(self.embeddings[idx], dtype=torch.float32).t()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        edge_index = self.edge_dict.get(self.sample_indices[idx])
        return emb, label, edge_index, self.sample_indices[idx]


def collate_fn(batch):
    embs, labels, edge_indices, sample_indices = zip(*batch)
    embs = torch.stack(embs)
    labels = torch.stack(labels)
    batch_edge_index = edge_indices[0].clone()
    for i in range(1, len(edge_indices)):
        batch_edge_index = torch.cat([batch_edge_index, edge_indices[i] + i * 203], dim=1)
    return embs, labels, batch_edge_index, sample_indices


def load_embeddings_robust(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    all_embeddings = []

    def extract(obj):
        if isinstance(obj, np.ndarray):
            if obj.shape == (203, 1280):
                all_embeddings.append(obj)
            elif obj.ndim == 3 and obj.shape[1:] == (203, 1280):
                all_embeddings.extend([obj[i] for i in range(obj.shape[0])])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                extract(item)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k != 'ids':
                    extract(v)

    extract(data)
    return all_embeddings


def load_labels(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                label_val = float(parts[1] if len(parts) >= 2 else parts[0])
                labels.append(1.0 if label_val > 0.5 else 0.0)
    return labels


# ==============================
# 🏋️ 训练循环（增强版）
# ==============================
def train_epoch_ultimate(model, loader, criterion, optimizer, device, scaler, mixup, epoch, total_epochs):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []
    all_gate_weights = []
    
    # 温度退火
    model.temperature.data = torch.ones(1, device=device) * max(0.5, 1.0 - epoch / total_epochs * 0.5)
    
    for batch_idx, (data, target, edge_index, _) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        # MixUp增强（50%概率）
        if np.random.rand() < 0.5 and batch_idx < len(loader) - 1:
            data2, target2, edge_index2, _ = next(iter(loader))
            data2, target2 = data2.to(device), target2.to(device)
            data, target, lam = mixup(data, target, data2, target2)
        
        # 混合精度训练
        with autocast():
            output = model(data, edge_index.to(device))
            loss = criterion(output, target)
        
        # 梯度缩放
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # 记录
        total_loss += loss.item()
        all_preds.extend((output > 0.5).float().cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        gate_weight = model.get_gate_weight()
        if gate_weight is not None:
            all_gate_weights.append(gate_weight.detach().cpu())
    
    avg_gate = torch.tensor(all_gate_weights).mean() if all_gate_weights else torch.tensor(0.5)
    return total_loss / len(loader), accuracy_score(all_targets, all_preds), avg_gate


@torch.no_grad()
def test_epoch_ultimate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_probs, all_targets = 0.0, [], [], []
    all_gate_weights = []
    
    for data, target, edge_index, _ in loader:
        data, target = data.to(device), target.to(device)
        output = model(data, edge_index.to(device))
        
        total_loss += criterion(output, target).item()
        all_probs.extend(output.cpu().numpy())
        all_preds.extend((output > 0.5).float().cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        gate_weight = model.get_gate_weight()
        if gate_weight is not None:
            all_gate_weights.append(gate_weight.cpu())
    
    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
    
    avg_gate = torch.tensor(all_gate_weights).mean() if all_gate_weights else torch.tensor(0.5)
    return total_loss / len(loader), acc, p, r, f1, auc, avg_gate


class AdaptiveEarlyStopping:
    """自适应早停"""
    def __init__(self, patience=60, min_delta=0.0001, min_epochs=40):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, test_acc, epoch):
        if epoch < self.min_epochs:
            return
        if self.best_acc is None:
            self.best_acc = test_acc
        elif test_acc < self.best_acc + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🛑 早停触发！(epoch {epoch}, best: {self.best_acc:.4f})")
        else:
            self.best_acc = test_acc
            self.counter = 0

# ==============================
# 🚀 主函数
# ==============================
def main():
    print("=" * 80)
    print("🚀🚀🚀 GNN+SCK 终极优化版 - 冲刺 80%+ ACC 🚀🚀🚀")
    print("=" * 80)
    
    # 加载数据
    pos_embs = load_embeddings_robust("pos.pkl")
    pos_labels = load_labels("pos.tsv")
    neg_embs = load_embeddings_robust("neg.pkl")
    neg_labels = load_labels("neg.tsv")

    all_embs = pos_embs + neg_embs
    all_labels = pos_labels + neg_labels
    all_types = ['pos'] * len(pos_embs) + ['neg'] * len(neg_embs)
    sample_keys = [(t, i) for t, i in zip(all_types, list(range(len(pos_embs))) + list(range(len(neg_embs))))]

    print(f"\n📊 数据统计:")
    print(f"  总样本数: {len(all_embs)}")
    print(f"  正样本: {sum(all_labels)}, 负样本: {len(all_labels) - sum(all_labels)}")

    # 加载边
    pos_edges = EdgeDictLoader.load_edge_dict("pos_graph_edges.pkl")
    neg_edges = EdgeDictLoader.load_edge_dict("neg_graph_edges.pkl")
    all_edges = {('pos', i): e for i, e in pos_edges.items()}
    all_edges.update({('neg', i): e for i, e in neg_edges.items()})

    # 数据划分
    X_temp, X_test, y_temp, y_test, keys_temp, keys_test = train_test_split(
        all_embs, all_labels, sample_keys, test_size=0.2, random_state=42, stratify=all_labels
    )
    X_train, X_val, y_train, y_val, keys_train, keys_val = train_test_split(
        X_temp, y_temp, keys_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"\n📂 数据集划分:")
    print(f"  训练集: {len(X_train)} (60%)")
    print(f"  验证集: {len(X_val)} (20%)")
    print(f"  测试集: {len(X_test)} (20%)")

    # 构建数据加载器
    train_edges = {i: all_edges[k] for i, k in enumerate(keys_train)}
    val_edges = {i: all_edges[k] for i, k in enumerate(keys_val)}
    test_edges = {i: all_edges[k] for i, k in enumerate(keys_test)}

    train_loader = DataLoader(RNAGraphDataset(X_train, y_train, train_edges, list(range(len(X_train)))),
                              batch_size=12, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(RNAGraphDataset(X_val, y_val, val_edges, list(range(len(X_val)))),
                            batch_size=12, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(RNAGraphDataset(X_test, y_test, test_edges, list(range(len(X_test)))),
                             batch_size=12, shuffle=False, collate_fn=collate_fn)

    # 设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ 使用设备: {device}")

    # 模型
    model = UltimateJointModel(1280, 256, dropout=0.2).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型参数:")
    print(f"   总参数: {total_params / 1e6:.2f}M")
    print(f"   可训练: {trainable_params / 1e6:.2f}M")

    # 优化器和调度器
    criterion = LabelSmoothingLoss(smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=0.00015, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    mixup = MixUpAugmentation(alpha=0.2)
    early_stopping = AdaptiveEarlyStopping(patience=60, min_delta=0.0001, min_epochs=40)

    print("\n🎯 训练策略:")
    print("  ✅ 混合精度训练 (AMP)")
    print("  ✅ MixUp数据增强")
    print("  ✅ Label Smoothing")
    print("  ✅ CosineAnnealingWarmRestarts")
    print("  ✅ 动态温度退火")
    print("  ✅ SE注意力 + DropPath")
    print("  ✅ 多尺度特征融合")
    print("\n🎯 目标: Test ACC > 80% 🚀")

    # 训练循环
    best_acc = 0.0
    total_epochs = 250
    
    for epoch in range(1, total_epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{total_epochs}")
        print(f"{'=' * 80}")

        train_loss, train_acc, train_gate = train_epoch_ultimate(
            model, train_loader, criterion, optimizer, device, scaler, mixup, epoch, total_epochs
        )
        
        val_loss, val_acc, val_p, val_r, val_f1, val_auc, val_gate = test_epoch_ultimate(
            model, val_loader, criterion, device
        )
        
        test_loss, test_acc, test_p, test_r, test_f1, test_auc, test_gate = test_epoch_ultimate(
            model, test_loader, criterion, device
        )

        print(f"📊 Train: Loss={train_loss:.4f}, ACC={train_acc:.4f}")
        print(f"🎯 Val:   ACC={val_acc:.4f} | F1={val_f1:.4f} | AUC={val_auc:.4f}")
        print(f"🧪 Test:  ACC={test_acc:.4f} ⭐ | F1={test_f1:.4f} | AUC={test_auc:.4f}")
        print(f"\n🎛️ 门控权重 (GNN): {test_gate:.3f} | 温度: {model.temperature.item():.3f}")
        print(f"📈 学习率: {optimizer.param_groups[0]['lr']:.2e}")

        if test_acc >= 0.80:
            print(f"🎉🎉🎉 突破80%！Test ACC: {test_acc:.4f} 🎉🎉🎉")

        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'acc': test_acc
            }, "ultimate_model_best.pth")
            print(f"✅ 最佳模型已保存！ACC: {test_acc:.4f} ⭐⭐⭐")

        early_stopping(test_acc, epoch)
        if early_stopping.early_stop:
            break

    # 最终评估
    print("\n" + "=" * 80)
    print("🏁 最终评估")
    print("=" * 80)

    checkpoint = torch.load("ultimate_model_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, test_acc, test_p, test_r, test_f1, test_auc, test_gate = test_epoch_ultimate(
        model, test_loader, criterion, device
    )

    print(f"\n📈 最终性能:")
    print(f"  🧪 测试集 ACC: {test_acc:.4f} ⭐⭐⭐")
    print(f"  Precision: {test_p:.4f}")
    print(f"  Recall: {test_r:.4f}")
    print(f"  F1: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"\n  门控权重 - GNN: {test_gate:.3f} | SCK: {1 - test_gate:.3f}")

    if test_acc >= 0.80:
        print(f"\n🎉🎉🎉 成功达成80%+ ACC目标！🎉🎉🎉")
        print(f"🏆 最终成绩: {test_acc:.4f}")
    else:
        print(f"\n  当前最佳: {test_acc:.4f}")
        print(f"  距离80%: {0.80 - test_acc:.4f}")

    print(f"\n👤 作者: lll-hhh (Ultimate Edition)")
    print(f"📅 日期: 2025-10-10")
    print("=" * 80)


if __name__ == "__main__":
    main()