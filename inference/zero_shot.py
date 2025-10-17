"""
Run the dataset through zero-shot prompting the LLM to get a baseline. 
"""
import csv
from os import mkdir
from os.path import dirname, join, exists
from json import loads
from inference.generation import generate_tokens

print("zero_shot.py: run test dataset through zero-shot prompting an LLM")
print("Expects dataset as a .csv with columns 'case_id', 'input_finding', 'output_disease'")
print("See dataset/example.csv for an example.\n")

with open(join(dirname(__file__), "prompts", "system.md"), "r") as file:
	system_prompt = file.read()
	assert system_prompt

if not exists(join(dirname(dirname(__file__)), "output")):
	mkdir(join(dirname(dirname(__file__)), "output"))

model_name = input("Enter LLM to use (e.g. 'unsloth/Qwen3-4B-unsloth-bnb-4bit'): ")
data = input("Enter dataset .csv: dataset/")
output_rows: list[list[str]] = []

with open(join(dirname(dirname(__file__)), "dataset", data), "r", encoding="cp1252") as file:
	reader = csv.reader(file)
	header = next(reader)
	assert header[:3] == ["case_id", "input_finding", "output_disease"]

	i = 0
	for row in reader:
		input_finding = row[1]
		thinking, response = generate_tokens(model_name, system_prompt, input_finding)
		response_copy = response

		while response and response[0] != "[": response = response[1:] # left trim until JSON or empty
		while response and response[-1] != "]": response = response[:-1]
		if not response: raise Exception(f"invalid response from LLM: {response_copy}")

		labels = loads(response)
		output_rows.append([row[0], row[1], row[2], thinking, ", ".join(labels)])
		i += 1
		print(f"Completed {i} rows")

# Write output
with open(join(dirname(dirname(__file__)), "output", "output.csv"), "w", encoding="utf-8") as file:
	writer = csv.writer(file)
	writer.writerow(["case_id", "input_finding", "output_disease", "llm_thinking", "llm_labels"])
	writer.writerows(output_rows)
