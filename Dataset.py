from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE 
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="Train Transformer for translation en to fr")
    parser.add_argument("--src_lang", "-srclang", type=str, default="en")
    parser.add_argument("--tgt_lang", "-tgt_lang", type=str, default="fr")
    arg = parser.parse_args()
    return arg

def get_training_corpus(ds, lang):
    pass

def get_or_build_tokenizer(args, ds, lang):
    tokenizer_path = Path(args.tokenizer_file.format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        special_tokens = ["[UNK]", "[PAD]", "[EOS]", "[SOS]", "[MASK]", "[CLS]"]
        trainer = BpeTrainer(vocab_size=30000, special_tokens=special_tokens, min_frequency=2)
        tokenizer.train_from_iterator(get_training_corpus(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer



def get_dataset(args):
    ds = load_dataset("Helsinki-NLP/opus_books", f"{args.src_lang}-{args.tgt_lang}", split="train")
    return ds

if __name__ == "__main__":
    args = get_args()

    get_or_build_tokenizer()