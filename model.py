import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class GeoSpatialTemporalNet(nn.Module):
    """
    纯净自适应通道与分支架构：
    - 支持全动态特征通道输入。
    - 支持一键屏蔽 GRU 时间分支。
    - 彻底移除人为先验，卷积核权重分配完全由数据驱动。
    """

    class BranchAttention(nn.Module):
        def __init__(self, num_branches: int, channel: int, reduction: int = 4):
            super().__init__()
            inter_channel = max(channel // reduction, 1)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, inter_channel),
                nn.ReLU(inplace=True),
                nn.Linear(inter_channel, num_branches)
            )
            self.num_branches = num_branches

        def forward(self, branch_outputs: List[torch.Tensor]):
            if self.num_branches == 1:
                return branch_outputs[0], None, torch.ones(branch_outputs[0].size(0), 1, device=branch_outputs[0].device)
                
            U = sum(branch_outputs)
            S = self.gap(U).view(U.size(0), -1)
            
            # 完全由网络自身学习到的原始输出 (Raw Logits)
            attention_logits = self.fc(S)
            
            # 通过标准 Softmax 转化为最终注意力权重
            attention_weights = F.softmax(attention_logits, dim=1)
            
            weights_reshaped = attention_weights.view(-1, self.num_branches, 1, 1, 1)
            stacked_branches = torch.stack(branch_outputs, dim=1)
            fused_feature = torch.sum(stacked_branches * weights_reshaped, dim=1)
            
            # 同时返回 logits 和 weights，用于在训练循环中实施隔离的 L1/L2 正则化
            return fused_feature, attention_logits, attention_weights

    def __init__(self, static_channels: int, dynamic_channels: int, 
                 use_time: bool = True,
                 cnn_kernel_sizes: list[int] = [3, 5, 7], 
                 gru_hidden_size: int = 64, 
                 cnn_output_features: int = 128,
                 image_size: int = 9): 
        super().__init__()
        self.use_time = use_time
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.gru_hidden_size = gru_hidden_size
        self.cnn_output_features = cnn_output_features
        self.center_pixel_idx = image_size // 2

        # --- 1. CNN 分支 (处理静态空间数据) ---
        num_branches = len(cnn_kernel_sizes)
        branch_output_channels = 32
        self.cnn_branches = nn.ModuleList()
        for kernel_size in cnn_kernel_sizes:
            padding = (kernel_size - 1) // 2
            self.cnn_branches.append(nn.Sequential(
                nn.Conv2d(static_channels, branch_output_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            ))
        
        self.attention = self.BranchAttention(num_branches, branch_output_channels)
        self.center_pixel_extractor = nn.Conv2d(branch_output_channels, cnn_output_features, kernel_size=1)

        # --- 2. GRU 分支 (处理动态时间数据) ---
        if self.use_time and dynamic_channels > 0:
            self.gru_branch = nn.GRU(
                input_size=dynamic_channels,
                hidden_size=gru_hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
            fusion_dim = cnn_output_features + gru_hidden_size
        else:
            self.gru_branch = None
            fusion_dim = cnn_output_features

        # --- 3. 融合与回归头 ---
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        nn.init.xavier_uniform_(self.regressor[-1].weight, gain=0.5)
        nn.init.constant_(self.regressor[-1].bias, 2.5)

    def forward(self, x_static: torch.Tensor, x_dynamic: torch.Tensor = None) -> tuple:
        # --- 1. CNN 分支 ---
        branch_outputs = [branch(x_static) for branch in self.cnn_branches]
        fused_feature, attention_logits, attention_weights = self.attention(branch_outputs)

        # --- 2. 提取中心像元空间特征 ---
        center_feature_map = self.center_pixel_extractor(fused_feature)
        static_features = center_feature_map[:, :, self.center_pixel_idx, self.center_pixel_idx]

        # --- 3. GRU 分支及特征组合 ---
        if self.use_time and self.gru_branch is not None and x_dynamic is not None:
            x_dynamic_permuted = x_dynamic.permute(0, 2, 1)
            _, h_n = self.gru_branch(x_dynamic_permuted)
            dynamic_features = h_n[-1, :, :]
            final_features = torch.cat([static_features, dynamic_features], dim=1)
        else:
            final_features = static_features

        # --- 4. 预测 ---
        prediction = self.regressor(final_features)
        return prediction, attention_logits, attention_weights