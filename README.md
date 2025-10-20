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

#### Evaluation Metric
To compare LLM-generated disease labels with expected labels, each label is encoded by an [embedding model](https://huggingface.co/Manal0809/medical-term-similarity) specialized in medical term similarities, and the pair of labels (one from expected, one from actual) with the highest cosine similarity is repeatedly removed until any of the lists are empty. The score is determined to be the average of the extracted cosine similarities.

## Finetuning
To finetune a model on a train dataset .csv:
```
python -i -m finetune.no_think
```

When asked, provide the model and your train dataset .csv filename.

For hyperparameters used, see [here](https://github.com/esu-rs/FinetuneLLM-NLP-Quiz/blob/main/finetune/no_think.py#L31) and [here](https://github.com/esu-rs/FinetuneLLM-NLP-Quiz/blob/main/finetune/no_think.py#L71).


When Python becomes interactive (finetuning complete), upload the weights to your Hugging Face:
```
model.push_to_hub_merged("hf_user/repo", tokenizer, save_method="merged_4bit_forced", token="")
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
B = Baseline, F = Finetuned; 0.85 = number of 236 test set rows with at least 0.85 score.

| Model | B 0.85 | F 0.85 |
| ----- | -------- | ------- |
| Qwen 3 4B | 68 | 88 (**+29%**) |
| Qwen 3 8B | 74 | 88 (**+19%**) |

Baseline 4B average score: 0.74 → Finetuned 4B average score: 0.78 (**+5%**)<br>
Baseline 8B average score: 0.75 → Finetuned 8B average score: 0.77 (**+3%**)

The complete baseline and finetuned output labels and evaluation scores can be found in [results.xlsx](https://github.com/esu-rs/FinetuneLLM-NLP-Quiz/blob/main/output/results.xlsx).

The finetuned models are available on Hugging Face:
- https://huggingface.co/evansu/qwen3_4b_disease_label_ft_v7
- https://huggingface.co/evansu/qwen3_8b_disease_label_ft_v7
