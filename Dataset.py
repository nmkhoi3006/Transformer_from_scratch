from datasets import load_dataset
from tokenizers import Tokenizer, decoders, processors
from tokenizers.models import BPE 
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from torch.utils.data import random_split, Dataset, DataLoader
import torch

from pathlib import Path
from config import get_args

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(args, ds, lang):
    tokenizer_path = Path(args.tokenizer_file.format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        special_tokens = ["[UNK]", "[PAD]", "[EOS]", "[SOS]", "[MASK]", "[CLS]"]
        trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens, min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

class TranslationData(Dataset):

    def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([self.src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        pair_lang = self.ds[index]
        src_text = pair_lang["translation"][self.src_lang]
        tgt_text = pair_lang["translation"][self.tgt_lang]

        src_token_input = torch.tensor([self.src_tokenizer.encode(src_text).ids], dtype=torch.int64)
        tgt_token_input = torch.tensor([self.tgt_tokenizer.encode(tgt_text).ids], dtype=torch.int64)

        enc_num_pad_token = self.seq_len - len(src_token_input) - 2
        dec_num_pad_token = self.seq_len - len(tgt_token_input) - 1

        encoder_input = torch.cat(
            [
                self.sos_token,
                src_token_input,
                self.eos_token,
                torch.tensor([self.pad_token * enc_num_pad_token], dtype=torch.int64)
            ],
            dim=0
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                tgt_token_input,
                torch.tensor([self.pad_token * dec_num_pad_token], dtype=torch.int64)
            ],
            dim=0
        )

        label = torch.cat(
            [
                tgt_token_input,
                self.eos_token,
                torch.tensor([self.pad_token * dec_num_pad_token], dtype=torch.int64)
            ],
            dim=0
        )
        look_ahead_mask = (torch.triu(torch.ones((1, self.seq_len, self.seq_len)), diagonal=1) == 0).int()
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() #(B, num_head, seq_len)
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0) & look_ahead_mask

        return{
            "encoder_input": encoder_input, #(seq_len)
            "decoder_input": decoder_input, #(seq_len)
            "encoder_mask": encoder_mask, #(B, num_head, seq_len) = (1, 1, seq_len) 
            "decoder_mask": decoder_mask, #(B, seq_len, seq_len) = (1, seq_len, seq_len)
            "label": label, #(seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def get_dataset(args):
    ds = load_dataset("Helsinki-NLP/opus_books", f"{args.src_lang}-{args.tgt_lang}", split="train")

    tokenizer_src_lang = get_or_build_tokenizer(args, ds, args.src_lang)
    tokenizer_tgt_lang = get_or_build_tokenizer(args, ds, args.tgt_lang)

    len_train = int(0.9*len(ds))
    len_val = len(ds) - len_train
    
    train_data, val_data = random_split(ds, [len_train, len_val])
    return tokenizer_src_lang, tokenizer_tgt_lang, train_data, val_data

if __name__ == "__main__":
    args = get_args()

    tokenizer_en_lang, tokenizer_fr_lang, train_data, val_data = get_dataset(args)

    sentences = """
    If I try to imagine that first night which I must have spent in my attic, 
    amidst the lumber-rooms on the upper storey, I recall other nights; 
    I am no longer alone in that room; a tall, restless, and friendly shadow moves along its walls and walks to and fro.
    """

    # # sentence = "Let's test this tokenizer."
    encoding = tokenizer_en_lang.encode(sentences)
    start, end = encoding.offsets[4]

    print(tokenizer_en_lang.decode(encoding.ids))