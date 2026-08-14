import os, fire
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model

# Model from Hugging Face hub or Model path
base_model = "meta-llama/Llama-3.1-8B-Instruct"

def run(base_model, new_model, data_files, country="Unknown"):

    import os
    os.environ["WANDB_PROJECT"] = f"CultureLLM-{country}"
    os.environ["WANDB_RUN_NAME"] = f"llama8b-{country}-run1"
    
    dataset = load_dataset('json', data_files=data_files)['train']

    # 80 / 20 split
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    temp_dataset = dataset["test"]
    
    # 10 / 10 split from the 20%
    temp_dataset = temp_dataset.train_test_split(test_size=0.5, seed=42)
    eval_dataset = temp_dataset["train"]
    test_dataset = temp_dataset["test"]

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    #model.quantization_config = quant_config
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
    )
    training_params = TrainingArguments(
        #seed=42,
        #data_seed=42,
        output_dir=new_model,
        num_train_epochs=12,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    trainer.train()
    
    #print("Running final evaluation on test set...")
    #print("Test set size:", len(test_dataset))
    #evaluate_on_test(trainer.model, tokenizer, test_dataset)

    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)
    
    trainer.train()

    best_checkpoint = trainer.state.best_model_checkpoint
    print("Best checkpoint:", best_checkpoint)
    
    model = AutoModelForCausalLM.from_pretrained(best_checkpoint)
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)

def evaluate_on_test(model, tokenizer, test_dataset):
    import re
    model.eval()

    stats = {}

    for example in test_dataset:
        full_text = example["text"]

        prompt_part = full_text.split("### Answer:")[0] + "### Answer:"
        gold_text = full_text.split("### Answer:")[1].strip()
        gold_label = int(re.search(r"\d+", gold_text).group())

        scale = detect_scale(prompt_part)

        if scale not in stats:
            stats[scale] = {"correct": 0, "total": 0}

        inputs = tokenizer(prompt_part, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer_text = decoded.split("### Answer:")[-1].strip()
        match = re.search(r"\d+", answer_text)

        if match:
            predicted_label = int(match.group())
        else:
            predicted_label = None

        if predicted_label == gold_label:
            stats[scale]["correct"] += 1

        stats[scale]["total"] += 1

    print("\nAccuracy per scale:")
    for scale, values in stats.items():
        acc = values["correct"] / values["total"]
        print(f"{scale}: {acc:.4f} ({values['correct']}/{values['total']})")

    #accuracy = correct / total
    #print(f"\nTest Accuracy: {accuracy:.4f}")
    #if log_to_wandb:
    #    wandb.log({"test_accuracy": accuracy})

def evaluate_finetuned_model(base_model, data_files):
    dataset = load_dataset('json', data_files=data_files)['train']

    dataset = dataset.train_test_split(test_size=0.2, seed=42) #seed=42
    temp_dataset = dataset["test"]
    temp_dataset = temp_dataset.train_test_split(test_size=0.5, seed=42) #seed=42
    test_dataset = temp_dataset["test"]

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,  
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Evaluating fine-tuned model...")
    print("Test set size:", len(test_dataset))

    evaluate_on_test(model, tokenizer, test_dataset)

    
def evaluate_base_model(base_model, data_files):

    dataset = load_dataset('json', data_files=data_files)['train']

    # SAME split as training
    dataset = dataset.train_test_split(test_size=0.2, seed=42) #seed=42
    temp_dataset = dataset["test"]
    temp_dataset = temp_dataset.train_test_split(test_size=0.5, seed=42) #seed=42
    test_dataset = temp_dataset["test"]

    print("Test set size:", len(test_dataset))

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Evaluating BASE model...")
    debug_base_outputs(model, tokenizer, test_dataset, num_examples=10)
    evaluate_on_test(model, tokenizer, test_dataset)

def detect_scale(prompt_text):
    if "1 to 10" in prompt_text:
        return "1-10"
    elif "1 to 5" in prompt_text:
        return "1-5"
    elif "1 to 4" in prompt_text:
        return "1-4"
    elif "0 to 2" in prompt_text:
        return "0-2"
    else:
        return "unknown"

def debug_base_outputs(model, tokenizer, test_dataset, num_examples=5):
    model.eval()

    print("\n--- Debugging Base Model Outputs ---\n")

    for i, example in enumerate(test_dataset):
        if i >= num_examples:
            break

        full_text = example["text"]

        prompt_part = full_text.split("### Answer:")[0] + "### Answer:"
        gold_label = full_text.split("### Answer:")[1].strip()[0]

        inputs = tokenizer(prompt_part, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\nExample {i+1}")
        print("Gold label:", gold_label)
        print("Model output:")
        print(decoded)
        print("-" * 60)



def eval():
    prompt = "Who is Leonardo Da Vinci?"
    tokenizer = LlamaTokenizer.from_pretrained("")
    model = LlamaForCausalLM.from_pretrained("", device_map="auto")
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])


if __name__ == '__main__':
    fire.Fire({
        "train": run,
        "eval_base": evaluate_base_model,
        "eval_test": evaluate_finetuned_model
    })
