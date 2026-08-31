import torch
import torch.nn as nn
from .multihead_self_attention import MultiHeadSelfAttentionRoPE
from .swiglu import SwiGLU, SiLU
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len):
        super().__init__()
        self.MHSA = MultiHeadSelfAttentionRoPE(d_model, num_heads)
        # self.swiglu = SwiGLU(d_ff, d_model)
        self.silu = SiLU(d_ff, d_model)
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)

        self.theta = theta
        self.max_seq_len = max_seq_len

    def forward(self, x):
        attn_out = self.MHSA(x, self.theta, self.max_seq_len, torch.arange(x.shape[-2]))
        x = self.rmsnorm1(x + attn_out)
        ffn_out = self.silu(x)
        x = self.rmsnorm2(x + ffn_out)
        return x

    def load_weights(self, state_dict, layer_idx=None):
        if layer_idx is not None:
            prefix = f"layers.{layer_idx}."
        else:
            prefix = ""

        self.MHSA.load_state_dict({
            "wQ": state_dict[prefix + "attn.q_proj.weight"],
            "wK": state_dict[prefix + "attn.k_proj.weight"],
            "wV": state_dict[prefix + "attn.v_proj.weight"],
            "wO": state_dict[prefix + "attn.output_proj.weight"],
        })

        self.rmsnorm1.load_state_dict({
            "g": state_dict[prefix + "ln1.weight"]
        })

        self.rmsnorm2.load_state_dict({
                    "g": state_dict[prefix + "ln2.weight"]
                })

        self.swiglu.load_state_dict({
            "w1": state_dict[prefix + "ffn.w1.weight"],
            "w2": state_dict[prefix + "ffn.w2.weight"],
            "w3": state_dict[prefix + "ffn.w3.weight"],
        })





