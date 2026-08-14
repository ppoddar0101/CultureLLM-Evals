import os
import json
import random
import argparse
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def generate_prompt(row, prompt_method):
    system_prompt = 'You are a group of individuals with these shared characteristics:\n'
    system_prompt += str(row['group_prompt_template'])
    prompt = '**Question**: ' + row['input_template'] + '\n'
    if row.get('group_prompt_variable_map'):
        for variable, value in row['group_prompt_variable_map'].items():
            system_prompt = system_prompt.replace(f'{{{variable}}}', str(value))
    if prompt_method == 'token_prob':
        prompt += 'Do not provide any explanation, only answer with one of the following options: ' + ', '.join(
            row['human_answer'].keys()) + '.\n**Answer**: ('
    elif prompt_method == 'verbalized':
        json_format_str = '{' + ', '.join([f'"{key}": X' for key in row['human_answer'].keys()]) + '}'
        prompt += (
            f'\nEstimate what percentage of your group would choose each option. '
            f'Follow these rules:\n'
            f'1. Use whole numbers from 0 to 100\n'
            f'2. Ensure the percentages sum to exactly 100\n'
            f'3. Only include the numbers (no % symbols)\n'
            f'4. Use this exact valid JSON format: {json_format_str} and do NOT include anything else.\n'
            f'5. Only output your final answer and nothing else.\n'
            f'Replace X with your estimated percentages for each option.\n'
            '**Answer**:'
        )
    return system_prompt, prompt

def get_token_probabilities_local_model(model, tokenizer, prompt, target_tokens):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]
        probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
        results = {}
        for token in target_tokens:
            encodings = tokenizer.encode(token, add_special_tokens=False)
            underscored_encodings = tokenizer.encode("_" + token, add_special_tokens=False)
            probability_sum = sum(probabilities[encoding].item() for encoding in encodings)
            probability_sum += sum(probabilities[encoding].item() for encoding in underscored_encodings)
            results[token] = probability_sum
    return results

def parse_and_normalize_json_response(response, expected_keys):
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in response")
    parsed = json.loads(json_match.group())
    values = []
    for key in expected_keys:
        val = parsed.get(key, parsed.get(str(key), 0))
        values.append(float(val))
    total = sum(values)
    if total == 0:
        raise ValueError("All zero values")
    return [v / total for v in values]

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--adapter_path', type=str, default=None, help='Path to LoRA adapter, or None for base model')
parser.add_argument('--method', type=str, default='token_prob')
parser.add_argument('--datasets', type=str, default=None, help='Comma-separated list of datasets to filter')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

set_seed(42)

dataset = pd.read_pickle(args.input_file)

# Filter to target datasets
if args.datasets:
    target_datasets = [d.strip() for d in args.datasets.split(',')]
    dataset = dataset[dataset['dataset_name'].isin(target_datasets)].reset_index(drop=True)
    print(f'Filtered to {len(dataset)} rows from datasets: {target_datasets}')

if args.debug:
    dataset = dataset.sample(frac=1).reset_index(drop=True)[:20]
    print(f'Debug mode: using {len(dataset)} rows')

print(f'Loading base model: {args.base_model}')
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    quantization_config=quant_config,
    device_map='auto',
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if args.adapter_path:
    print(f'Loading LoRA adapter from: {args.adapter_path}')
    model = PeftModel.from_pretrained(model, args.adapter_path)
    print('Adapter loaded successfully')
else:
    print('Running base model (no adapter)')

model_distribution = []
total_system_prompt = []
total_user_prompt = []
total_probs = []

for id in tqdm(range(len(dataset))):
    row = dataset.iloc[id]
    system_prompt, user_prompt = generate_prompt(row, args.method)
    total_system_prompt.append(system_prompt)
    total_user_prompt.append(user_prompt)

    if args.method == 'token_prob':
        target_tokens = list(row['human_answer'].keys())
        overall_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            tokenize=False
        )
        probabilities = get_token_probabilities_local_model(model, tokenizer, overall_prompt, target_tokens)
        total_prob = sum(probabilities.values())
        normalized_probs = [v / total_prob for v in probabilities.values()]
        total_probs.append(total_prob)
        model_distribution.append(normalized_probs)

    elif args.method == 'verbalized':
        max_retries = 5
        retry_count = 0
        temperature = 0.0001
        expected_keys = list(row['human_answer'].keys())
        while retry_count < max_retries:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True
            ).to('cuda')
            output_ids = model.generate(
                input_ids,
                max_new_tokens=250,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True
            )
            response_ids = output_ids[:, input_ids.shape[-1]:]
            response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
            try:
                normalized_probs = parse_and_normalize_json_response(response, expected_keys)
                model_distribution.append(normalized_probs)
                break
            except ValueError:
                retry_count += 1
                temperature = 1
                if retry_count == max_retries:
                    model_distribution.append([1.0/len(expected_keys)] * len(expected_keys))

new_rows = []
for idx, row in dataset.iterrows():
    new_row = row.copy()
    new_row['System_Prompt'] = total_system_prompt[idx]
    new_row['User_Prompt'] = total_user_prompt[idx]
    new_row['Response_Distribution'] = model_distribution[idx]
    new_row['Sum_of_Probs'] = total_probs[idx] if total_probs else np.nan
    new_row['Model'] = args.adapter_path if args.adapter_path else args.base_model
    new_row['Prompt_Method'] = args.method
    new_rows.append(new_row)

new_dataset = pd.DataFrame(new_rows)
new_dataset.to_pickle(args.output_file)
print(f'Saved {len(new_dataset)} results to {args.output_file}')
