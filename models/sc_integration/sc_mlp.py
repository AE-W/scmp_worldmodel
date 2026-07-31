"""MLP drop-in that switches fc1/fc2 to SC per the global config.

State-dict compatible with timm's Mlp (same fc1.weight/bias, fc2.weight/bias).
"""
import torch.nn as nn

from .sc_controller import is_op_enabled
from .sc_linear import sc_linear_forward


class SCMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., block_idx=-1, **kwargs):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
        self.block_idx = block_idx

    def forward(self, x):
        b = self.block_idx
        x = sc_linear_forward(x, self.fc1, op="mlp_fc1", block_idx=b) if is_op_enabled("mlp_fc1", b) else self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = sc_linear_forward(x, self.fc2, op="mlp_fc2", block_idx=b) if is_op_enabled("mlp_fc2", b) else self.fc2(x)
        x = self.drop2(x)
        return x
