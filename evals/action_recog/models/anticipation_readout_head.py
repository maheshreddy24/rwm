import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, self.head_dim)
        self.key = nn.Linear(embed_dim, self.head_dim)
        self.value = nn.Linear(embed_dim, self.head_dim)
        self.out = nn.Linear(self.head_dim, embed_dim)

    def forward(self, x):
        bs, seq, embd = x.size()
        x = x.view(bs, seq, self.num_heads, self.head_dim)
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.view(bs, seq, self.head_dim)
        return self.out(attn_output)

class CrossAttn(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttn, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, self.head_dim)
        self.key = nn.Linear(embed_dim, self.head_dim)
        self.value = nn.Linear(embed_dim, self.head_dim)
        self.out = nn.Linear(self.head_dim, embed_dim)

    def forward(self, x, context):
        bs, seq, embd = x.size()
        x = x.view(bs, seq, self.num_heads, self.head_dim)
        context = context.view(bs, seq, self.num_heads, self.head_dim)
        q, k, v = self.query(x), self.key(context), self.value(context)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.view(bs, seq, self.head_dim)
        return self.out(attn_output)

class ReadOutHead(nn.Module):
    def __init__(self, verb_classes, noun_classes):
        super(ReadOutHead, self).__init__()
        self.verb_classifier = nn.Linear()
        self.noun_classifier = nn.Linear()

        self.self_attn1 = SelfAttention(embed_dim=384, num_heads=8)
        self.self_attn2 = SelfAttention(embed_dim=384, num_heads=8)
        self.self_attn3 = SelfAttention(embed_dim=384, num_heads=8)
        self.cross_attn1 = CrossAttn(embed_dim=384, num_heads=8)

    