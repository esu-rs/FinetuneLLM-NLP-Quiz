import csv, torch
from os import mkdir
from os.path import dirname, join, exists
from json import loads, dumps
from unsloth import FastLanguageModel
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from datasets import Dataset
import pandas as pd

print("no_think.py: finetune Qwen 3 with empty thinking in the training dataset")
print("Expects dataset as a .csv with columns 'case_id', 'input_finding', 'output_disease'")
print("See dataset/example.csv for an example.\n")

with open(join(dirname(dirname(__file__)), "inference", "prompts", "system.md"), "r") as file:
	system_prompt = file.read()
	assert system_prompt

model_name = input("Enter LLM to use (e.g. 'unsloth/Qwen3-4B-unsloth-bnb-4bit'): ")
data = input("Enter training dataset .csv: dataset/")
output_rows: list[list[str]] = []

model, tokenizer = FastLanguageModel.from_pretrained(
	model_name = model_name,
	max_seq_length = 16384,
	load_in_4bit = True, # reduces memory usage significantly
	load_in_8bit = False,
	full_finetuning = False
)

model = FastLanguageModel.get_peft_model(
	model,
	r = 64,
	target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
	lora_alpha = 64,
	lora_dropout = 0,
	bias = "none",
	use_gradient_checkpointing = "unsloth",
	use_rslora = False,
	loftq_config = None
)

rows: list[str] = []
with open(join(dirname(dirname(__file__)), "dataset", data), "r", encoding="cp1252") as file:
	reader = csv.reader(file)
	header = next(reader)
	assert header[:3] == ["case_id", "input_finding", "output_disease"]

	for row in reader:
		input_finding = row[1]
		output_disease = list(map(lambda i: i.lower().strip(), row[2].split(",")))

		text = tokenizer.apply_chat_template([
			{ "role": "system", "content": system_prompt },
			{ "role": "user", "content": input_finding },
			{ "role": "assistant", "content": "<think>\n\n</think>\n```json\n" + dumps(output_disease) + "\n```" }
		], tokenize=False, add_generation_prompt=False)
		rows.append(text)

data = pd.concat([
	pd.Series(rows)
])
data.name = "text"
dataset = Dataset.from_pandas(pd.DataFrame(data))

trainer = SFTTrainer(
	model = model,
	tokenizer = tokenizer, # type: ignore
	train_dataset = dataset,
	eval_dataset = None,
	args = SFTConfig(
		dataset_text_field = "text",
		per_device_train_batch_size = 2,
		gradient_accumulation_steps = 8,
		warmup_steps = 10,
		num_train_epochs = 1,
		learning_rate = 8e-5,
		logging_steps = 1,
		optim = "adamw_8bit",
		weight_decay = 0.01,
		lr_scheduler_type = "linear",
		report_to = "none",
	)
)
trainer.train()

print('Done. Now you can: model.push_to_hub_merged("hf_user/repo", tokenizer, save_method="merged_4bit_forced", token="")')
