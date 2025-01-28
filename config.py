def get_args():
    return{
        "--src_lang": "en",
        "--tgt_lang": "vi",
        "--tokenizer_file": "./tokenizer_{}",

        "--batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 100,

        "seq_len": 350,
        "vocab_size": 30000,
        "d_model": 512,
        "num_head": 8,
        "ff_dim": 2048,
        "drop_out": 0.1,
        "num_block": 6
    }