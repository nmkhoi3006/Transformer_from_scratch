from torch import nn
import math
import torch
from layer import EncoderBlock, DecoderBlock, InputEmbedding, PositionEmbedding

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
    
    def forward(self, x, encoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_mask)
        return x


class Decoder(nn.Module):

    def __init__(self,
                num_decoder_block: int,
                d_model: int,
                ff_dim: int,
                num_head: int,
                drop_out: int):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, ff_dim, num_head, drop_out) 
                                    for _ in range(num_decoder_block)])
    
    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)
        return x
    
class Transformer(nn.Module):

    def __init__(self,
                 vocab_size: int = 30000,
                 max_seq_len: int = 350,
                 num_encoder_block: int = 6,
                 num_decoder_block: int = 6,
                 d_model: int = 512,
                 ff_dim: int = 2048,
                 num_head: int = 8,
                 drop_out: int = 0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            InputEmbedding(vocab_size, d_model),
            PositionEmbedding(max_seq_len, d_model, drop_out)
        )
        self.encoder = Encoder(
            num_encoder_block=num_encoder_block, 
            d_model=d_model,
            ff_dim=ff_dim,
            num_head=num_head,
            drop_out=drop_out
        )
        self.decoder = Decoder(
            num_decoder_block=num_decoder_block,
            d_model=d_model,
            ff_dim=ff_dim,
            num_head=num_head,
            drop_out=drop_out
        )
        self.projection_layer = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def encode(self, x, encoder_mask):
        x = self.embedding(x)
        x = self.encoder(x, encoder_mask)
        return x

    def decode(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.embedding(x)
        x = self.decoder(x, encoder_output, encoder_mask, decoder_mask)
        return x
    
    def project(self, x):
        x = self.projection_layer(x)
        return self.softmax(x)

if __name__ == "__main__":  
    vocab_size = 30000
    max_seq_len = 200
    d_model = 512
    ff_dim = 2048
    num_head = 8
    drop_out = 0.1

    num_encoder_block = 6
    num_decoder_block = 6


    src_text = torch.randint(low=0, high=10, size=(1, 15))
    tgt_text = torch.randint(low=0, high=10, size=(1, 10))

    