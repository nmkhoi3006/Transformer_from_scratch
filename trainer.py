from model import Transformer
from Dataset import get_dataset
from config import get_args
from tqdm import tqdm
from torch import nn
import torch
import numpy as np

def gen_text(model: Transformer, encoder_input, encoder_mask, tokenizer_tgt_lang, max_seq_len, device):
    sos_idx = tokenizer_tgt_lang.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt_lang.token_to_id("[EOS]")

    encoder_output = model.encode(encoder_input, encoder_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)
    while True:
        seq_len = decoder_input.shape[1]
        look_ahead_mask = (torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0).int()
        decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, look_ahead_mask) #(B, seq_len, d_model)
        
        prob = model.project(decoder_output[:, -1]) #(b, vocab_size)
        next_token_id = torch.argmax(prob)

        decoder_input = torch.cat(
            [
                decoder_input,
                torch.tensor(next_token_id).type_as(encoder_input).to(device)
            ]
        )

        if decoder_input.shape[1] == max_seq_len or next_token_id == eos_idx:
            break

    return decoder_input.squeeze(0)

def train(args):
    epochs = args.epochs
    lr = args.learning_rate
    vocab_size = args.vocab_size
    seq_len = args.seq_len
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang

    weight_path = "./weight.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_epoch = 0
    

    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_dataset(args)
    model = Transformer().to(device)
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
            batch_iter.set_description(f"Epoch: {epoch}/{epochs}. Loss {(train_loss/(iter+1)):0.3f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Val
        model.eval()
        count = 0
        num_example = 2
        with torch.inference_mode():
            for batch in val_loader:
                count += 1
                encoder_input = batch["encoder_input"].to(device)
                encoder_mask = batch["encoder_mask"].to(device)

                token_ids = gen_text(model, encoder_input, encoder_mask, tokenizer_tgt, seq_len, device)
                pred_sentence = tokenizer_tgt.decode(token_ids.detach().cpu().numpy())
                src_text = batch["src_text"]
                tgt_text = batch["tgt_text"]

                print(f"{src_lang}: {src_text}")
                print(f"{tgt_lang}: {tgt_text}")
                print(f"predict: {pred_sentence}")
            
            if count == num_example:
                break
        
        check_point = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1
        }

        torch.save(check_point, weight_path)




if __name__ == "__main__":
    args = get_args()
    train(args)