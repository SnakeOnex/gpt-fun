import torch, torch.nn as nn, torch.nn.functional as F, time
from dataclasses import dataclass
from einops import rearrange

@dataclass
class TransformerConfig:
    n_layers: int
    n_heads: int
    n_embd: int
    block_size: int
    def __post_init__(self):
        self.head_dim = self.n_embd // self.n_heads

class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(Attention, self).__init__()
        for k, v in config.__dict__.items(): setattr(self, k, v)
        self.q = nn.Linear(self.n_embd, self.head_dim)
        self.k = nn.Linear(self.n_embd, self.head_dim)
        self.v = nn.Linear(self.n_embd, self.head_dim)
        self.register_buffer("attn_mask", ~torch.tril(torch.ones(32, 32, dtype=torch.bool)))
    def forward(self, x):
        attn = self.q(x) @ self.k(x).transpose(-2,-1) * (1/self.head_dim)**0.5
        attn.masked_fill_(self.attn_mask[:x.shape[1],:x.shape[1]], float("-inf"))
        return attn.softmax(dim=-1) @ self.v(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(MultiHeadAttention, self).__init__()
        for k, v in config.__dict__.items(): setattr(self, k, v)
        self.heads = nn.ModuleList([Attention(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class TinyAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TinyAttention, self).__init__()
        for k, v in config.__dict__.items(): setattr(self, k, v)
        self.qkv = nn.Linear(self.n_embd, self.n_embd * 3)
        self.register_buffer("attn_mask", ~torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)))
    def forward(self, x):
        q, k, v = rearrange(self.qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.n_heads)
        attn = q @ k.transpose(-2,-1) * (1/self.head_dim)**0.5
        attn.masked_fill_(self.attn_mask, float("-inf"))
        out = attn.softmax(dim=-1) @ v
        return rearrange(out, "b h n d -> b n (h d)", h=self.n_heads)

class TransformerLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerLayer, self).__init__()
        for k, v in config.__dict__.items(): setattr(self, k, v)
        self.multi_attn = MultiHeadAttention(config)
        # self.multi_attn = TinyAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            nn.GELU(),
            nn.Linear(4 * self.n_embd, self.n_embd),
        )
    def forward(self, x):
        x = x + self.multi_attn(F.layer_norm(x, (self.n_embd,)))
        x = x + self.mlp(F.layer_norm(x, (self.n_embd,)))
        return x

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(Transformer, self).__init__()
        for k, v in config.__dict__.items(): setattr(self, k, v)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.n_layers)])
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, trans_config: TransformerConfig):
        super(TinyGPT, self).__init__()
        self.embed = nn.Embedding(vocab_size, trans_config.n_embd)
        self.transformer = Transformer(trans_config)
        self.proj = nn.Linear(trans_config.n_embd, vocab_size)
    def forward(self, x):
        return self.proj(self.transformer(self.embed(x)))

if __name__ == "__main__":
    # with open("shakespear.txt", "r") as f: text = f.read()
    with open("enwik9", "rb") as f: text = f.read()[:10000]
    chars = sorted(list(set(text)))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    tokens = torch.tensor([char2idx[c] for c in text])
    vocab_size = len(chars)
    print("corpus length: ", len(text))
    print("vocab_size: ", vocab_size)

    batch_size = 128
    block_size = 32

    def get_batch():
        idxs = torch.randint(0, len(tokens) - block_size, (batch_size,))
        x = torch.stack([tokens[idx:idx+block_size] for idx in idxs])
        y = torch.stack([tokens[idx+1:idx+block_size+1] for idx in idxs])
        return x, y

    gpt = TinyGPT(vocab_size, TransformerConfig(n_layers=2, n_heads=8, n_embd=512, block_size=block_size))
    print(f"params: {sum(p.numel() for p in gpt.parameters()) / 1e6:.1f}M")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gpt.parameters(), lr=1e-4)

    for i in range(100):
        st = time.time()
        x, y = get_batch()
        preds = gpt(x)
        loss = loss_fn(preds.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        dt = time.time() - st
        print(f"step={i}: loss={loss.item():.4f}, dt={dt:.2f}")

    torch.save(gpt.state_dict(), "gpt.pth")

    for _ in range(10):
        print("attempt: ", end="")
        x = torch.tensor([char2idx[" "]]).unsqueeze(0)
        for i in range(31):
            preds = gpt(x)
            next_char = torch.multinomial(F.softmax(preds[:,-1,:], dim=-1), 1).item()
            print(idx2char[next_char], end="")
            x = torch.cat([x, torch.tensor([next_char]).unsqueeze(0)], dim=1)
        print()
