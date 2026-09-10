import torch
import torch.nn as nn

BATCH = 2
N_WAYPOINTS = 64
HIDDEN = 64
VLM_SEQ_LEN = 32
N_HEADS = 4

class ExpertBlock(nn.Module):
    """一个 transformer block：self-attn → cross-attn → FFN。"""
    def __init__(self, hidden=HIDDEN, n_heads=N_HEADS):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.SiLU(), nn.Linear(hidden * 4, hidden),
        )
        
    def forward(self, x, condition):
        # ① self-attention：动作 token 之间互相看（非因果！和 LLM 相反）
        x = x + self.self_attn(x, x, x)[0]
        x = self.norm1(x)
        # ② cross-attention：动作当 query，去"查" VLM 条件
        x = x + self.cross_attn(
            query=x,              # 每个动作 token 问："我该往哪走？"
            key=condition,        # 在 VLM 条件里找答案
            value=condition,      # 取回 VLM 条件的实际内容
        )[0]
        x = self.norm2(x)
        # ③ FFN
        x = x + self.ffn(x)
        x = self.norm3(x)
        return x
    
class Expert(nn.Module):
    def __init__(self, hidden=HIDDEN, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([ExpertBlock(hidden) for _ in range(n_blocks)])

        
    def forward(self, x, condition):
        for blk in self.blocks:
            x = blk(x, condition)
            
        return x
if __name__ == "__main__":
    expert = Expert()

    # 假想的"动作 embedding"（Stage 3 的 ActionInProj 输出）
    action_embeds = torch.randn(BATCH, N_WAYPOINTS, HIDDEN)   # (B,64,64)
    # 假想的"VLM 条件"（Stage 5 换成真 VLM）
    condition = torch.randn(BATCH, VLM_SEQ_LEN, HIDDEN)       # (B,32,64)
    
    out=expert(action_embeds, condition)
    print(f"动作 embedding: {tuple(action_embeds.shape)}")
    print(f"VLM 条件:       {tuple(condition.shape)}")
    print(f"expert 输出:    {tuple(out.shape)}   ← 形状不变，内容被 attention 加工")


    # 关键验证：cross-attention 真的在读条件吗？
    out_zero = expert(action_embeds, torch.zeros_like(condition))
    diff = (out - out_zero).abs().mean().item()
    print(f"\n条件=随机 vs 条件=全0 的输出差异: {diff:.4f}")
    print("（>0 说明 cross-attention 确实在'读'条件，不是摆设）")



        
