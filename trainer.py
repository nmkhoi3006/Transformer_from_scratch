from model import Transformer
from Dataset import get_dataset
from config import get_args
from tqdm import tqdm
from torch import nn
import torch
import numpy as np

def train(args):
    epochs = args.epochs
    lr = args.learning_rate
    vocab_size = args.vocab_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_epoch = 0
    

    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_dataset(args)
    model = Transformer()
    optimizer = torch.optim.Adam(model.parameters(), lr, eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)


    for epoch in range(init_epoch, epochs):
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0
        batch_iter = tqdm(train_loader, colour="cyan")
        for iter, batch in enumerate(batch_iter):
            encoder_input = batch["encoder_input"].to(device) 
            decoder_input = batch["decoder_input"].to(device) 
            encoder_mask = batch["encoder_mask"].to(device) 
            decoder_mask = batch["decoder_mask"].to(device) 
            label = batch["label"].to(device) 

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            project = model.project(decoder_output)

            loss = criterion(project.view(-1, vocab_size), label.view(-1))
            train_loss += loss.item()
            batch_iter.set_description(f"Epoch: {epoch}/{epochs}. Loss {train_loss/(iter+1)}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




if __name__ == "__main__":
    args = get_args()
    train(args)