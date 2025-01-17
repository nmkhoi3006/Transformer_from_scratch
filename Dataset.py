from datasets import load_dataset
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE 
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from torch.utils.data import random_split

import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="Train Transformer for translation en to fr")
    parser.add_argument("--src_lang", "-slang", type=str, default="en")
    parser.add_argument("--tgt_lang", "-tlang", type=str, default="fr")
    parser.add_argument("--tokenizer_file", "-tf", type=str, default="./tokenizer_{}")
    arg = parser.parse_args()
    return arg

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(args, ds, lang):
    tokenizer_path = Path(args.tokenizer_file.format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        special_tokens = ["[UNK]", "[PAD]", "[EOS]", "[SOS]", "[MASK]", "[CLS]"]
        trainer = BpeTrainer(vocab_size=30000, special_tokens=special_tokens, min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.decode = decoders.ByteLevel()
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer



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

    encoding = tokenizer_en_lang(sentences)
    print(encoding)

    decoding = tokenizer_en_lang(UnicodeEncodeError.ids)
    print(decoding == sentences)