import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train Transformer for translation en to fr")
    parser.add_argument("--src_lang", type=str, default="en")
    parser.add_argument("--tgt_lang", type=str, default="fr")
    parser.add_argument("--tokenizer_file", type=str, default="./tokenizer_{}")

    # for training
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--learning_rate", "-lr", type=int, default=1e-4)
    parser.add_argument("--epochs", "-e", type=int, default=20)

    # for model
    parser.add_argument("--seq_len", type=int, default=350)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_head", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--drop_out", type=float, default=0.1)
    parser.add_argument("--num_block", type=int, default=6)

    arg = parser.parse_args()
    return arg