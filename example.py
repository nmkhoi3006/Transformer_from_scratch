from Dataset import get_dataset, get_args
from tokenizers import Tokenizer
from tokenizers.models import BPE

if __name__ == "__main__":
    args = get_args()

    sentences = """
    If I try to imagine that first night which I must have spent in my attic, 
    amidst the lumber-rooms on the upper storey, I recall other nights; 
    I am no longer alone in that room; a tall, restless, and friendly shadow moves along its walls and walks to and fro.
    """
    tokenizer_file = "./tokenizer_en"
    tokenizer_en_lang = Tokenizer(BPE()).from_file(tokenizer_file)

    encoding = tokenizer_en_lang.encode(sentences)
    print(encoding)

    decoding = tokenizer_en_lang.decode(encoding.ids)
    print(decoding)
