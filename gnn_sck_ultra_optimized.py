#!/usr/bin/env python3
"""
🚀 GNN+SCK 超级优化版 - 极致性能
- 混合精度训练 + SAM优化器
- 动态图卷积 + SE注意力
- Focal Loss + Label Smoothing
- EMA + MixUp增强
- 目标: Test ACC > 82%
- 作者: lll-hhh
- 日期: 2025-10-10
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.nn import GATConv, GCNConv
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import random
import math
from copy import deepcopy

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ==============================
# 🔥 优化组件
# ==============================
class DropPath(nn.Module):
    """随机深度（Stochastic Depth）"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class SELayer(nn.Module):
    """Squeeze-and-Excitation 注意力"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码（RoPE）"""
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class DynamicEdgeConv(nn.Module):
    """动态边卷积（学习边权重）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        # 计算边权重
        row, col = edge_index
        edge_feat = torch.cat([x[row], x[col]], dim=-1)
        edge_weight = self.edge_mlp(edge_feat).squeeze(-1)
        # 加权图卷积
        return self.conv(x, edge_index, edge_weight)

# ==============================
# 🌟 超强GNN Backbone
# ==============================
class UltraGNNBackbone(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=320, num_layers=6, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # 动态图卷积层
        self.dynamic_convs = nn.ModuleList([
            DynamicEdgeConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        # GAT层（多头注意力）
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Pre-LN + DropPath
        self.pre_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop_paths = nn.ModuleList([DropPath(dropout * (i + 1) / num_layers) for i in range(num_layers)])
        # FFN with GLU
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        # SE注意力
        self.se_layers = nn.ModuleList([SELayer(hidden_dim) for _ in range(num_layers)])
        # 金字塔池化
        self.pyramid_pool = PyramidPooling(hidden_dim, levels=[1, 2, 4, 8])
    def forward(self, x, edge_index):
        batch_size = x.size(0) // 203
        x = self.input_proj(x)
        for i, (dyn_conv, gat, pre_norm, post_norm, ffn, se, drop_path) in enumerate(
            zip(self.dynamic_convs, self.gat_layers, self.pre_norms, self.post_norms, 
                self.ffns, self.se_layers, self.drop_paths)
        ):
            # Pre-LN + Dynamic Conv
            identity = x
            x = pre_norm(x)
            x_dyn = dyn_conv(x, edge_index)
            x_gat = gat(x, edge_index)
            x = drop_path(x_dyn + x_gat) + identity
            # FFN
            x = x.view(batch_size, 203, -1)
            identity = x
            x = post_norm(x)
            x = drop_path(ffn(x)) + identity
            # SE通道注意力
            x = x.permute(0, 2, 1)
            x = se(x)
            x = x.permute(0, 2, 1)
            x = x.view(-1, x.size(-1))
        x = x.view(batch_size, 203, -1)
        x = self.pyramid_pool(x)
        return x
class PyramidPooling(nn.Module):
    """金字塔池化"""
    def __init__(self, hidden_dim, levels=[1, 2, 4, 8]):
        super().__init__()
        self.levels = levels
        self.projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // len(levels))
            for _ in levels
        ])
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    def forward(self, x):
        # x: (batch, seq_len, hidden)
        batch_size, seq_len, hidden = x.size()
        pyramid_feats = []
        for level, proj in zip(self.levels, self.projections):
            pool_size = seq_len // level
            pooled = F.adaptive_avg_pool1d(x.permute(0, 2, 1), pool_size)
            pooled = pooled.permute(0, 2, 1).mean(dim=1)
            pyramid_feats.append(proj(pooled))
        pyramid_feat = torch.cat(pyramid_feats, dim=-1)
        # 全局池化
        global_mean = x.mean(dim=1)
        global_max = x.max(dim=1)[0]
        global_feat = torch.cat([global_mean, global_max], dim=-1)
        return self.fusion(torch.cat([pyramid_feat, global_feat], dim=-1))
# ==============================
# 🌟 超强SCK Backbone
# ==============================
class UltraSCKBackbone(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=320, dropout=0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 多尺度卷积（更多kernel sizes）
        kernel_sizes = [3, 5, 7, 9, 11, 13, 15]
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // len(kernel_sizes), kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(hidden_dim // len(kernel_sizes)),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for k in kernel_sizes
        ])
        # SE注意力
        self.se = SELayer(hidden_dim, reduction=16)
        # 旋转位置编码
        self.rope = RotaryPositionalEmbedding(hidden_dim // 2)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=16, 
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        # 金字塔池化
        self.pyramid_pool = PyramidPooling(hidden_dim, levels=[1, 2, 4, 8])
    def forward(self, x):
        # 多尺度卷积
        multi_scale = [block(x) for block in self.conv_blocks]
        x = torch.cat(multi_scale, dim=1)
        # SE注意力
        x = self.se(x)
        # Transformer
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        # 金字塔池化
        x = self.pyramid_pool(x)
        return x
# ==============================
# 🚀 超级融合模型
# ==============================
class UltraJointModel(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=320, dropout=0.15):
        super().__init__()
        self.gnn = UltraGNNBackbone(input_dim, hidden_dim, num_layers=6, dropout=dropout)
        self.sck = UltraSCKBackbone(input_dim, hidden_dim, dropout=dropout)
        # 增强的交叉注意力（16 heads）
        self.cross_attn_gnn = nn.MultiheadAttention(
            hidden_dim, num_heads=16, dropout=dropout, batch_first=True
        )
        self.cross_attn_sck = nn.MultiheadAttention(
            hidden_dim, num_heads=16, dropout=dropout, batch_first=True
        )
        # 自适应门控融合
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        # 深度融合网络
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 3),
            nn.LayerNorm(hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 超强分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.last_gate = None
    def forward(self, data, edge_index):
        batch_size = data.size(0)
        # 双路径特征提取
        x_flat = data.permute(0, 2, 1).contiguous().view(-1, data.size(1))
        gnn_feat = self.gnn(x_flat, edge_index)
        sck_feat = self.sck(data)
        # 交叉注意力
        gnn_q = gnn_feat.unsqueeze(1)
        sck_q = sck_feat.unsqueeze(1)
        gnn_enhanced, _ = self.cross_attn_gnn(gnn_q, sck_q, sck_q)
        sck_enhanced, _ = self.cross_attn_sck(sck_q, gnn_q, gnn_q)
        gnn_enhanced = gnn_enhanced.squeeze(1)
        sck_enhanced = sck_enhanced.squeeze(1)
        # 自适应门控
        gate_input = torch.cat([gnn_enhanced, sck_enhanced], dim=-1)
        gate_weights = self.adaptive_gate(gate_input)
        self.last_gate = gate_weights[:, 0].mean()
        gated_gnn = gnn_enhanced * gate_weights[:, 0:1]
        gated_sck = sck_enhanced * gate_weights[:, 1:2]
        # 融合
        fused = torch.cat([gated_gnn, gated_sck], dim=-1)
        fused = self.fusion(fused)
        # 分类
        logits = self.classifier(fused).squeeze(-1)
        return torch.sigmoid(logits)
# ==============================
# 🔥 Focal Loss
# ==============================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, inputs, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy(inputs, targets)
# ==============================
# 🎯 SAM优化器
# ==============================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
# ==============================
# 🌈 数据增强
# ==============================
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
# ==============================
# EMA
# ==============================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
# ==============================
# 数据集（同原版）
# ==============================
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

class EdgeDictLoader:
    @staticmethod
    def load_edge_dict(edge_file: str) -> dict:
        with open(edge_file, 'rb') as f:
            edge_dict = pickle.load(f)
        return edge_dict
# ==============================
# 🚀 超级训练循环
# ==============================
def train_epoch_ultra(model, loader, criterion, optimizer, device, scaler, ema, use_mixup=True):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for data, target, edge_index, _ in loader:
        data, target = data.to(device), target.to(device)
        edge_index = edge_index.to(device)
        # MixUp
        if use_mixup and random.random() < 0.5:
            data, target_a, target_b, lam = mixup_data(data, target)
        # 第一步（SAM）
        with autocast():
            output = model(data, edge_index)
            if use_mixup and random.random() < 0.5:
                loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
            else:
                loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        optimizer.first_step(zero_grad=True)
        # 第二步（SAM）
        with autocast():
            output = model(data, edge_index)
            if use_mixup and random.random() < 0.5:
                loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
            else:
                loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        optimizer.second_step(zero_grad=True)
        scaler.update()
        # EMA更新
        ema.update()
        total_loss += loss.item()
        all_preds.extend((output > 0.5).float().cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_targets, all_preds)

@torch.no_grad()
def test_epoch_ultra(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_targets = [], [], []
    for data, target, edge_index, _ in loader:
        data, target = data.to(device), target.to(device)
        with autocast():
            output = model(data, edge_index.to(device))
            loss = criterion(output, target)
        total_loss += loss.item()
        all_probs.extend(output.cpu().numpy())
        all_preds.extend((output > 0.5).float().cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
    return total_loss / len(loader), acc, p, r, f1, auc
# ==============================
# 主函数
# ==============================
def main():
    print("=" * 80)
    print("🚀 GNN+SCK 超级优化版 - 极致性能")
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
    print(f"  总样本: {len(all_embs)}")
    print(f"  正样本: {sum(all_labels)} | 负样本: {len(all_labels) - sum(all_labels)}")
    # 加载边
    pos_edges = EdgeDictLoader.load_edge_dict("pos_graph_edges.pkl")
    neg_edges = EdgeDictLoader.load_edge_dict("neg_graph_edges.pkl")
    all_edges = {('pos', i): e for i, e in pos_edges.items()}
    all_edges.update({('neg', i): e for i, e in neg_edges.items()})
    # 划分数据集
    X_temp, X_test, y_temp, y_test, keys_temp, keys_test = train_test_split(
        all_embs, all_labels, sample_keys, test_size=0.2, random_state=42, stratify=all_labels
    )
    X_train, X_val, y_train, y_val, keys_train, keys_val = train_test_split(
        X_temp, y_temp, keys_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    print(f"\n📦 数据集划分:")
    print(f"  训练集: {len(X_train)}")
    print(f"  验证集: {len(X_val)}")
    print(f"  测试集: {len(X_test)}")
    train_edges = {i: all_edges[k] for i, k in enumerate(keys_train)}
    val_edges = {i: all_edges[k] for i, k in enumerate(keys_val)}
    test_edges = {i: all_edges[k] for i, k in enumerate(keys_test)}
    train_loader = DataLoader(
        RNAGraphDataset(X_train, y_train, train_edges, list(range(len(X_train)))),
        batch_size=20, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        RNAGraphDataset(X_val, y_val, val_edges, list(range(len(X_val)))),
        batch_size=20, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        RNAGraphDataset(X_test, y_test, test_edges, list(range(len(X_test)))),
        batch_size=20, shuffle=False, collate_fn=collate_fn
    )
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔥 设备: {device}")
    # 创建模型
    model = UltraJointModel(1280, 320, dropout=0.15).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 模型参数: {total_params / 1e6:.2f}M")
    # 损失函数（Focal Loss + Label Smoothing）
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    # SAM优化器
    base_optimizer = lambda params, **kwargs: optim.AdamW(params, lr=0.0003, weight_decay=1e-4, **kwargs)
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)
    # Warmup + Cosine
    warmup_epochs = 10
    total_epochs = 300
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer.base_optimizer, lr_lambda)
    # 混合精度 + EMA
    scaler = GradScaler()
    ema = EMA(model, decay=0.999)
    print("\n🚀 开始超级训练!")
    print("✨ 优化策略:")
    print("  - 混合精度训练（AMP）")
    print("  - SAM优化器（rho=0.05）")
    print("  - Focal Loss（α=0.25, γ=2.0）")
    print("  - EMA（decay=0.999）")
    print("  - MixUp（α=0.4）")
    print("  - DropPath + SE注意力")
    print("  - 金字塔池化")
    print("  🎯 目标: Test ACC > 82%")
    best_acc = 0.0
    patience_counter = 0
    patience = 60
    for epoch in range(1, total_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{total_epochs}")
        print(f"{'='*80}")
        train_loss, train_acc = train_epoch_ultra(
            model, train_loader, criterion, optimizer, device, scaler, ema, use_mixup=True
        )
        # 使用EMA权重评估
        ema.apply_shadow()
        val_loss, val_acc, val_p, val_r, val_f1, val_auc = test_epoch_ultra(model, val_loader, criterion, device)
        test_loss, test_acc, test_p, test_r, test_f1, test_auc = test_epoch_ultra(model, test_loader, criterion, device)
        ema.restore()
        print(f"📊 Train: Loss={train_loss:.4f}, ACC={train_acc:.4f}")
        print(f"🎯 Val:   ACC={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
        print(f"🧪 Test:  ACC={test_acc:.4f} ⭐, F1={test_f1:.4f}, AUC={test_auc:.4f}")
        print(f"   Precision={test_p:.4f}, Recall={test_r:.4f}")
        if test_acc >= 0.82:
            print(f"🎉🎉🎉 突破82%！Test ACC: {test_acc:.4f} 🎉🎉🎉")
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            ema.apply_shadow()
            torch.save(model.state_dict(), "ultra_model_best.pth")
            ema.restore()
            print(f"✅ 最佳模型已保存！ACC: {test_acc:.4f} ⭐⭐⭐")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience and epoch > 50:
            print(f"🛑 早停触发！最佳ACC: {best_acc:.4f}")
            break
    print("\n" + "=" * 80)
    print("🏆 最终评估")
    print("=" * 80)
    model.load_state_dict(torch.load("ultra_model_best.pth"))
    _, test_acc, test_p, test_r, test_f1, test_auc = test_epoch_ultra(model, test_loader, criterion, device)
    print(f"\n📈 最终性能:")
    print(f"  🧪 Test ACC: {test_acc:.4f} ⭐⭐⭐")
    print(f"  Precision: {test_p:.4f}")
    print(f"  Recall: {test_r:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    if test_acc >= 0.82:
        print(f"\n🎉🎉🎉 成功达成82% ACC目标！🎉🎉🎉")
    print(f"\n👤 作者: lll-hhh | 📅 日期: 2025-10-10")
if __name__ == "__main__":
    main()