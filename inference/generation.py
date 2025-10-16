from unsloth import FastLanguageModel
import torch

_model, _tokenizer, _model_name = None, None, None

device = "cuda" if torch.cuda.is_available() else "cpu"

# e.g. generate_tokens("unsloth/Qwen3-4B-unsloth-bnb-4bit", "You are a helpful assistant.", "Why is the sky blue?")
def generate_tokens(model_name: str, system_prompt: str, input_str: str) -> tuple[str, str]:
	global _model, _tokenizer, _model_name
	if model_name != _model_name:
		_model, _tokenizer = FastLanguageModel.from_pretrained(
			model_name = model_name,
			max_seq_length = 16384,
			load_in_4bit = True # reduces memory usage significantly
		)
		FastLanguageModel.for_inference(_model)
		_model_name = model_name
	assert _model is not None and _tokenizer is not None

	text = _tokenizer.apply_chat_template([ # type: ignore
		{ "role": "system", "content": system_prompt },
		{ "role": "user", "content": input_str }
	], tokenize=False, add_generation_prompt=True, enable_thinking=True)
	model_inputs = _tokenizer([text], return_tensors="pt").to(device) # type: ignore

	generated_ids = _model.generate(
		**model_inputs,
		max_new_tokens=8192,
		temperature=0.6,
		do_sample=True,
		top_p=0.95,
		top_k=20,
		min_p=0
	) # recommended parameters from https://huggingface.co/unsloth/Qwen3-4B-unsloth-bnb-4bit#best-practices
	output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

	try:
		index = len(output_ids) - output_ids[::-1].index(151668) # find </think>
	except ValueError:
		raise Exception("ERROR: no </think> token found, model did not finish output")
	thinking = _tokenizer.decode(output_ids[:index], skip_special_tokens = True).strip("\n")
	response = _tokenizer.decode(output_ids[index:], skip_special_tokens = True).strip("\n")

	return thinking, response
