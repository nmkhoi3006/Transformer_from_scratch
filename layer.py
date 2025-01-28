from torch import nn
import math
import torch


class InputEmbedding(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x = (B, seq_len)
        return self.embedding(x) + math.sqrt(self.d_model)
    

class PositionEmbedding(nn.Module):

    def __init__(self, 
                 max_seq_len: int,
                 d_model: int, 
                 drop_out: float):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.drop_out = nn.Dropout(drop_out)

        even_i = torch.arange(0, self.d_model, 2, dtype=torch.float)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1) #(512, 1)
        div_term = torch.pow(10000, even_i/self.d_model)
        self.pe = torch.zeros((self.max_seq_len, self.d_model)) #(seq_len, 512)

        self.pe[:, 0::2] = torch.sin(position / div_term)
        self.pe[:, 1::2] = torch.cos(position / div_term)

        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        # print(f"Shape x: {x.shape}")
        # print(f"seq len: {self.seq_len}")
        # print(f"shape pe: {self.pe.shape}")
        seq_len = x.shape[1]
        device = x.device
        self.pe.to(device=device)
        x = x + (self.pe[:, :seq_len, :]).requires_grad_(False)
        return self.drop_out(x)
    
class Attention(nn.Module):
    
    def __init__(self,
                d_k: int):
        super().__init__()

        self.d_k = d_k
        self.w_q = nn.Linear(d_k, d_k)
        self.w_k = nn.Linear(d_k, d_k)
        self.w_v = nn.Linear(d_k, d_k)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v) #(1, seq_len, d_k)

        attention = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_k)
        # print(f"{attention.shape=}")
        if mask is not None:
            attention.masked_fill_(mask == 0, 1e-9)
        attention_score = self.softmax(attention) # (1, seq_len, seq_len)

        x = torch.matmul(attention_score, value)
        x.to(q.device)
        # print(f"{x.shape=}")
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self,
                d_model: int,
                num_heads: int,
                drop_out: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads =  num_heads
        self.drop_out = nn.Dropout(drop_out)

        self.head_dim = d_model // num_heads
        self.w_0 = nn.Linear(d_model, d_model)

        self.scaled_dot_product = Attention(self.head_dim)


    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor = None):
        
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1, 2) #(B, num_head, seq_len, head_dim)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1, 2) #(B, num_head, seq_len, head_dim)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1, 2) #(B, num_head, seq_len, head_dim)
        # print(f"{x.shape = }")

        x = self.scaled_dot_product(q, k, v, mask)
        x.to(q.device)
        
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.d_model)
        x = self.w_0(x)
        return self.drop_out(x)

class FeedForward(nn.Module):

    def __init__(self,
                d_model: int,
                ff_dim: int,
                drop_out: float):
        super().__init__()
        self.hidden_layer = nn.Linear(d_model, ff_dim)
        self.out_layer = nn.Linear(ff_dim, d_model)
        self._drop_out = nn.Dropout(drop_out)
    
    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.out_layer(x)
        return self._drop_out(x)


class EncoderBlock(nn.Module):

    def __init__(self, 
                d_model:int,
                ff_dim: int, 
                num_head: int,
                drop_out: float):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_head,
            drop_out=drop_out
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff_layer = FeedForward(
            d_model=d_model,
            ff_dim=ff_dim,
            drop_out=drop_out
        )
    
    def forward(self, x, encoder_mask ): #encoder_mask: (B, 1, seq_len)
        residual = x #(B, Seq_len, d_model)
        x = self.multi_head_attention(x, x, x, encoder_mask)
        x += residual
        x = self.layer_norm1(x)

        residual = x
        x = self.ff_layer(x)
        x += residual
        x = self.layer_norm2(x)

        return x  
    
class DecoderBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 ff_dim: int,
                 num_head: int,
                 drop_out: int):
        super().__init__()
        self.mask_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_head,
            drop_out=drop_out
        )

        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_head,
            drop_out=drop_out
        )

        self.ff_layer = FeedForward(
            d_model=d_model,
            ff_dim=ff_dim,
            drop_out=drop_out
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        seq_len = x.shape[1]
        residual = x
        x = self.mask_attention(x, x, x, decoder_mask)
        x += residual
        x = self.layer_norm1(x)

        residual = x
        x = self.cross_attention(x, encoder_output, encoder_output, encoder_mask)
        x += residual
        x = self.layer_norm2(x)

        residual = x
        x = self.ff_layer(x)
        x += residual
        x = self.layer_norm3(x)

        return x

