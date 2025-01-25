import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train Transformer for translation en to fr")
    parser.add_argument("--src_lang", "-slang", type=str, default="en")
    parser.add_argument("--tgt_lang", "-tlang", type=str, default="fr")
    parser.add_argument("--tokenizer_file", "-tf", type=str, default="./tokenizer_{}")

    parser.add_argument("--seq_len", "-sl", type=int, default=350)
    parser.add_argument("--vocab_size", "-vs", type=int, default=30000)

    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--learning_rate", "-lr", type=int, default=1e-4)
    parser.add_argument("--epochs", "-e", type=int, default=20)

    parser.add_argument("--d_model", "-d", type=int, default=512)
    parser.add_argument("--num_head", "-h", type=int, default=8)
    parser.add_argument("--ff_dim", "-ff", type=int, default=2048)
    parser.add_argument("--drop_out", "-do", type=float, default=0.1)
    parser.add_argument("--num_block", "-n", type=int, default=6)

    arg = parser.parse_args()
    return arg