from transformers import AutoModelForCausalLM, GPT2TokenizerFast
import torch
from tqdm import tqdm

# Perplexity code from transformers documentation https://huggingface.co/transformers/v3.2.0/perplexity.html

def calc_perplexity(model_type, file):
    f = open(file, "r")
    text = f.read()

    tokenizer = GPT2TokenizerFast.from_pretrained('data/pgntokenizer', eos_token='<|endoftext|>')
    encodings = tokenizer(text, return_tensors='pt')

    model = AutoModelForCausalLM.from_pretrained("data/pgnmodels_" + model_type)
    model.cuda()

    max_length = model.config.n_positions
    stride = 512

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids.cuda(), labels=target_ids.cuda())
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()
