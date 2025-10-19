# FinetuneLLM-NLP-Quiz
Evaluating Qwen 3 4B and 8B's performance in generating disease labels given radiology text findings.

## Environment and Requirements
- Ubuntu 24.04 tested, Windows 10+ is likely to also work
- At least 16 GiB VRAM (e.g. RTX 5070 Ti)
- CUDA 12.9
- Python 3.12.3

It is highly recommended to use a virtual environment to prevent dependency conflicts:
```sh
python3 -m venv .venv
source .venv/bin/activate
```

#### Install PyTorch
For optimal performance, install PyTorch with your appropriate compute platform:

https://pytorch.org/get-started/locally/

### Install Packages
```
pip install -r requirements.txt
```

## Dataset
The code requires a test and train dataset in .csv format with the following columns:
```
case_id,input_finding,output_disease
```

An example .csv may be found at [dataset/example.csv](dataset/example.csv).

### Preprocessing
No preprocessing is required.

## Baseline Scores
To get a baseline evaluation of Qwen 3 4B and 8B:
```
python -m inference.zero_shot
```

When prompted, enter a model (e.g. `unsloth/Qwen3-4B-unsloth-bnb-4bit`) and your test dataset .csv.

The system prompt can be found at [inference/prompts/system.md](inference/prompts/system.md).

When the inference completes, rename `output/output.csv` appropriately (e.g. `output_qwen3_4b.csv`).

### Scoring
To score how well the model generates disease labels compared to the expected labels:
```
python -m evaluation.score
```

Again, enter the model and the output .csv from the inference step above.

## Finetuning
To finetune a model on a train dataset .csv:
```
python -i -m finetune.no_think
```

When asked, provide the model and your train dataset .csv filename.

For hyperparameters used, see [here](https://github.com/esu-rs/FinetuneLLM-NLP-Quiz/blob/main/finetune/no_think.py#L31) and [here](https://github.com/esu-rs/FinetuneLLM-NLP-Quiz/blob/main/finetune/no_think.py#L71).


When the Python shell becomes interactive (finetuning finished), you can upload the weights to your Hugging Face account:
```
model.push_to_hub_merged("hf_user/repo", tokenizer, save_method="merged_4bit_forced", token="<TOKEN>")
```

## Finetuned Scores
Follow the same steps as for baseline to run inference with and score the finetuned models.

```
python -m inference.zero_shot
```

```
python -m evaluation.score
```

---

# Results

<!-- table -->

The complete baseline and finetuned output labels and evaluation scores can be found in [results.xlsx](https://github.com/esu-rs/FinetuneLLM-NLP-Quiz/blob/main/output/results.xlsx).

The finetuned models are available on Hugging Face:
- https://huggingface.co/evansu/qwen3_4b_disease_label_ft_v4
- https://huggingface.co/evansu/qwen3_8b_disease_label_ft_v4
