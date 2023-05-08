from transformers import GPT2TokenizerFast, DataCollatorForLanguageModeling
from transformers import Trainer, AutoConfig, TrainingArguments, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import torch
import gc

def prepare_dataset(train_path, test_path, tokenizer):
    try:
        reloaded_datasets = load_from_disk('data/mapped_datasets')
    except:
        files = {"train": train_path, "test": test_path}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raw_datasets = load_dataset("text", data_files=files)

        context_length = 256 # Quarter of max to fit on GPU

        def tokenize(element):
            outputs = tokenizer(
                element["text"],
                truncation=True,
                max_length=context_length,
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == context_length:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

        tokenized_datasets = raw_datasets.map(
            tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
        )
        tokenized_datasets.set_format("pt")
        tokenized_datasets.save_to_disk('data/mapped_datasets')
        reloaded_datasets = load_from_disk('data/mapped_datasets')
    print(reloaded_datasets)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return reloaded_datasets["train"], reloaded_datasets["test"], data_collator


def run_training(model_type, train_path, test_path):
    tokenizer = GPT2TokenizerFast.from_pretrained("data/pgntokenizer", eos_token='<|endoftext|>')
    tokenizer.pad_token = tokenizer.eos_token
    train_data, test_data, data_collator = prepare_dataset(train_path, test_path, tokenizer)

    config = AutoConfig.from_pretrained(
        model_type,
        vocab_size=len(tokenizer),
        n_ctx=256,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_type, config=config).to("cuda")

    training_args = TrainingArguments(
        output_dir="data/pgnmodel" + model_type,
        overwrite_output_dir=False,  # overwrite the content of the output directory
        num_train_epochs=1,  # number of training epochs
        per_device_train_batch_size=16,  # batch size for training
        per_device_eval_batch_size=16,  # batch size for evaluation
        fp16=True,   # FP16 mixed-precision training
        gradient_accumulation_steps=8,  # needed to ease gpu-usage
        dataloader_num_workers=0,  # fixes problem when the training doesn't start
        eval_steps=500,  # Number of update steps between two evaluations.
        save_steps=500,  # after # steps model is saved
        warmup_steps=1000,  # number of warmup steps for learning rate scheduler
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data.shuffle(seed=42).select(range(len(train_data)//2)),
        eval_dataset=test_data.shuffle(seed=42).select(range(len(test_data)//2)),
    )
    print(trainer.train_dataset)
    trainer.train()
    trainer.save_model()