from transformers import GPT2TokenizerFast

def fen_iterator(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()

def train_tokenizer(file_path, model):
    tokenizer = GPT2TokenizerFast.from_pretrained(model)

    # Special tokens
    # special_tokens_dict = {"sep_token": ";", "cls_token": "/"}
    # tokenizer.add_special_tokens(special_tokens_dict)

    # train the tokenizer on new data
    tokenizer.train_new_from_iterator(fen_iterator(file_path), vocab_size=50000, min_frequency=50)
    tokenizer.save_pretrained("data/fentokenizer" + model)
