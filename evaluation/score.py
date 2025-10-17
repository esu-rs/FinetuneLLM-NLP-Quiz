"""
Evaluate how well actual output labels match expected output labels.
"""
import csv
from os import mkdir
from os.path import dirname, join, exists
from json import loads
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Manal0809/medical-term-similarity")

print("score.py: generate scores between 0.0 and 1.0 for how accurate actual labels are")

filename = input("Enter inference output .csv with LLM-generated labels (e.g. output/baseline.csv): output/")

def score(expected_: list[str], actual_: list[str]) -> float:
	"""
	Evaluate how close the actual list of labels is compared to the expected.
	Outputs a float between 0 and 1 where 1 is perfect matching.

	Idea: repeatedly pick the cross-list pair with highest similarity until a list is empty.
	"""
	if not expected_ and actual_: return float(0) # should not be giving a label
	if expected_ and not actual_: return float(0) # did not give a label

	expected: set[str] = set(map(lambda i: i.lower().strip(), expected_))
	actual: set[str] = set(map(lambda i: i.lower().strip(), actual_))
	similarities: list[tuple[float, str, str]] = []

	# Get similarity scores for all pairs of items in expected x actual
	for i in expected:
		for j in actual:
			similarities.append((model.similarity(model.encode([i])[0], model.encode([j])[0])[0].item(), i, j))
	similarities.sort(reverse=True) # highest similar first

	total_score, count = float(0), 0
	while expected and actual:
		score, i, j = similarities.pop(0)
		if i in expected and j in actual:
			expected.remove(i)
			actual.remove(j)
			total_score += score
			count += 1
	average_score = total_score / count

	return max(average_score, float(0))

output_rows = []
with open(join(dirname(dirname(__file__)), "output", filename), "r", encoding="utf-8") as file:
	reader = csv.reader(file)
	header = next(reader)
	assert header[:5] == ["case_id", "input_finding", "output_disease", "llm_thinking", "llm_labels"]

	for row in reader:
		expected = list(map(lambda i: i.lower().strip(), row[2].split(",")))
		actual = list(map(lambda i: i.lower().strip(), row[4].split(",")))
		s = score(expected, actual)
		output_rows.append([row[0], s])

with open(join(dirname(dirname(__file__)), "output", filename.replace(".csv", "") + "_scores.csv"), "w", encoding="utf-8") as file:
	writer = csv.writer(file)
	writer.writerow(["case_id", "score"])
	writer.writerows(output_rows)

print(f"Completed -> {filename.replace(".csv", "")}_scores.csv")
