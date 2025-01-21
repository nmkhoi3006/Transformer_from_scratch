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
            attention += mask
        attention_score = self.softmax(attention) # (1, seq_len, seq_len)

        x = torch.matmul(attention_score, value)
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

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None):
        
        b, seq_len, d_model = x.shape
        # x = x.reshape(b, self.num_heads, seq_len, self.head_dim)
        x = x.reshape(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # print(f"{x.shape = }")

        attention = Attention(d_k = self.head_dim)
        x = attention(x, x, x, mask)
        
        x = x.transpose(1, 2).reshape(b, seq_len, d_model)
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
        self.layer_norm = nn.LayerNorm(d_model)
        self.ff_layer = FeedForward(
            d_model=d_model,
            ff_dim=ff_dim,
            drop_out=drop_out
        )
    
    def forward(self, x):
        residual = x #(B, Seq_len,, d_model)
        x = self.multi_head_attention(x)
        x += residual
        x = self.layer_norm(x)

        residual = x
        x = self.ff_layer(x)
        x += residual
        x = self.layer_norm(x)

        return x      

class Encoder(nn.Module):

    def __init__(self,
                num_encoder_block: int,
                d_model: int,
                ff_dim: int,
                num_head: int,
                drop_out: int):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, ff_dim, num_head, drop_out) 
                                    for _ in range(num_encoder_block)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    vocab_size = 30000
    max_seq_len = 200
    d_model = 512
    ff_dim = 2048
    num_head = 8
    drop_out = 0.1

    num_encoder_block = 6


    fake_data = torch.randint(low=0, high=10, size=(1, 10))

    embedding = nn.Sequential(
        InputEmbedding(vocab_size=vocab_size, d_model=d_model),
        PositionEmbedding(max_seq_len=350, d_model=d_model, drop_out=drop_out)
    )

    fake_data_embeded = embedding(fake_data)
    # print(fake_data_embeded)
    # print(f"{type(fake_data_embeded)}")
    # print(f"{fake_data_embeded.shape}")
    
    
    encoder = Encoder(
        num_encoder_block=num_encoder_block,
        d_model=d_model,
        ff_dim=ff_dim,
        num_head=num_head,
        drop_out=drop_out
    )

    out = encoder(fake_data_embeded)
    print(out.shape)