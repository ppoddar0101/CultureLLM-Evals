import json

INPUT_FILE = "data/Spanish/Finetune/WVQ_Spanish_1000.jsonl"
OUTPUT_FILE = "data/Spanish/Finetune/WVQ_Spanish_llama_1000.jsonl"

def convert_to_llama_format(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            data = json.loads(line)
            messages = data.get("messages", [])

            user_msg = None
            assistant_msg = None

            for msg in messages:
                if msg["role"] == "user":
                    user_msg = msg["content"].strip()
                elif msg["role"] == "assistant":
                    assistant_msg = msg["content"].strip()

            # skip malformed rows
            if not user_msg or not assistant_msg:
                continue

            # build llama-style text
            text = f"{user_msg}\n### Answer: {assistant_msg}"

            json.dump({"text": text}, outfile, ensure_ascii=False)
            outfile.write("\n")

    print(f"Saved converted file to: {output_path}")


if __name__ == "__main__":
    convert_to_llama_format(INPUT_FILE, OUTPUT_FILE)