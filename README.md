# prose-eval

An evaluation harness for a hand-written editorial rubric that flags machine-sounding prose.

The interesting question here is not "can you detect AI text." Plenty of tools claim that. The question is whether a rubric written by an editor carries real signal, how much of that signal survives a change of subject matter, and how much of any reported score is an artifact of the corpus instead of a property of the writing.

The short answer: the rubric works in-domain, cadence matters far more than word choice, and the whole thing falls apart out-of-domain in a way that is worth understanding.

## What it does

Scores a document on 32 features in two families.

Thirteen **rule features** come from a published editorial standard: inflated significance ("stands as a testament to"), promotional register, editorializing, transition crutches, hollow conclusions, negative parallelism ("not just X, it's Y"), participle tails that fake analysis, and so on. Counted per 100 words.

Nineteen **structural features** describe cadence instead of vocabulary: sentence-length variance, paragraph uniformity, type-token ratio, comma density, triplet rate, how often consecutive sentences open with the same word. The editorial standard claims structure is the stronger tell, since a writer can swap out every flagged word and still produce text whose rhythm is flat. This harness exists partly to test that claim.

Four scorers are compared on identical splits, plus a character TF-IDF model as a reference point.

## Results

Corpus is [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3), 4,000 question pairs, one human and one machine answer per question so topic is held roughly constant. 70/30 split, held-out n=2,400.

| model | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| threshold (rule count) | 0.613 | 0.806 | 0.298 | 0.436 | 0.661 |
| logreg: rules only | 0.675 | 0.910 | 0.389 | 0.545 | 0.675 |
| logreg: structure only | 0.907 | 0.901 | 0.914 | 0.908 | 0.963 |
| logreg: rules + structure | 0.915 | 0.912 | 0.919 | 0.915 | 0.969 |
| char TF-IDF (reference) | 0.943 | 0.939 | 0.947 | 0.943 | 0.986 |

Three things fall out of this.

The naive version of the rubric, which is what a regex linter actually does, is close to useless as a classifier at 0.613. Its precision is decent at 0.806, so when it fires it is usually right, but recall of 0.298 means it misses two thirds of machine text. That is the honest ceiling of "count the banned phrases."

Structure beats vocabulary by a wide margin, 0.907 against 0.675. The editorial standard's claim holds. Adding the rule features to the structural ones buys only another 0.008 accuracy, so nearly all the usable signal is in cadence.

A dumb character TF-IDF beats the whole hand-built thing. That's worth noticing: 32 interpretable features get you to 0.915 and 50,000 opaque character n-grams get you to 0.943. The rubric's argument is interpretability and the fact that each feature maps to an editing instruction, not raw score.

## The corpus is leaking, and here is by how much

HC3's human text was detokenized with a whitespace tokenizer. The machine text was not. That leaves a punctuation fingerprint with nothing to do with writing style:

| artifact | machine | human | separation |
|---|---|---|---|
| space before punctuation | 0.002 | 0.736 | 0.734 |
| detached clitic (" n't") | 0.000 | 0.475 | 0.475 |
| tokenizer-split hyphen | 0.011 | 0.234 | 0.224 |

A single regex for space-before-punctuation separates the classes on 73% of documents. Any model trained on raw HC3 is partly learning "was this string detokenized."

`proseeval.normalize` strips the three known artifacts without touching authored style: it leaves em dashes, real contractions, casing, and sentence boundaries alone, because those carry signal a detector is supposed to see. Running the same evaluation before and after quantifies the inflation:

| model | raw ROC-AUC | normalized ROC-AUC | change |
|---|---|---|---|
| logreg: rules + structure | 0.974 | 0.969 | −0.005 |
| char TF-IDF | 0.996 | 0.986 | −0.010 |

Smaller than expected, and the reason is instructive. The feature-based models never looked at punctuation spacing, so they had little to lose. TF-IDF did lose more, which is the point: the artifact-hungry model is the one that pays. A 0.010 drop also means my three regexes did not remove all of it, and character n-grams are still finding residual traces. The honest reading is that the normalizer reduces the leak instead of eliminating it, and that any absolute number from this corpus should be treated as an upper bound.

## Out-of-domain, it collapses

Same fitted models, zero retraining, applied to 2,000 scientific abstracts from [Ateeqq/AI-and-Human-Generated-Text](https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text):

| model | accuracy | F1 | ROC-AUC |
|---|---|---|---|
| threshold (rule count) | 0.569 | 0.401 | 0.533 |
| logreg: rules only | 0.533 | 0.382 | 0.520 |
| logreg: structure only | 0.325 | 0.127 | 0.292 |
| logreg: rules + structure | 0.353 | 0.145 | 0.295 |
| char TF-IDF (reference) | 0.513 | 0.218 | 0.613 |

The structural model does not merely fail. At 0.292 ROC-AUC it is well below chance, which means it is anti-correlated with truth: it is now reliably wrong, and inverting its predictions would score 0.708.

That is the most useful result in this repo. The in-domain model learned that long, multi-paragraph, evenly-cadenced text is machine-written, which is true of Reddit answers where the human comparison is a casual comment. In a corpus of scientific abstracts the polarity flips, because the human-written abstracts are the formal, uniform, evenly-cadenced ones. The feature weights say the same thing: `mean_paragraph_sentences` carries by far the largest weight at −4.361, and paragraph structure is exactly the property that does not transfer between Reddit and academic publishing.

The rule features degrade more gracefully, from 0.675 to 0.520, because a phrase like "it is important to note" means roughly the same thing in any domain. They just do not carry much signal to begin with.

## Where it gets things wrong

Of 204 errors on the held-out set, the confident ones cluster in two recognizable groups.

Human encyclopedic prose gets called machine. Four of the five most confident errors are Wikipedia-derived text on technical subjects: real numbers, time complexity, cumulative distribution functions. Reference writing is uniform by design, which is the exact property the model was trained to treat as suspicious.

Short machine refusals get called human. The clearest example is a 49-word "I'm sorry, but I don't have enough information to accurately answer your question," which is unmistakably machine output to any reader and scores as human because it is short and structurally irregular. Length carries a positive weight of +1.311, so brevity alone pulls toward the human class.

Both failures share a cause. The model learned a proxy, "long and uniform," instead of the target property, and the proxy comes apart at the edges of the length distribution and outside the training domain.

## Running it

```bash
pip install -r requirements.txt
python scripts/run_eval.py --pairs 4000 --ood 2000
```

Downloads both corpora from Hugging Face, prints every table above, and writes `results/metrics.json`. Takes about two minutes on a laptop. Seeded, so it reproduces exactly.

Scoring a single document:

```python
from proseeval import extract, normalize

feats = extract(normalize(open("draft.md").read()))
print(feats["cv_sentence_len"], feats["rule_transition_crutch"])
```

Tests:

```bash
python -m pytest tests/ -q     # 30 tests
```

## What I would do next

Fit on a mixture of domains instead of one, since the OOD collapse is a single-domain artifact. Replace raw length with a length-invariant formulation, which should fix the short-refusal failure. Test against 2026-era model output; HC3 was collected in 2022, so everything here demonstrates detection of three-year-old model prose, and current models write with far more cadence variation. Add a calibration curve, because for an editing tool the useful output is a calibrated probability instead of a label.

## Layout

```
proseeval/
  normalize.py   artifact removal + leakage measurement
  rules.py       the rubric, as compiled patterns by group
  features.py    32 features, rule and structural
  data.py        corpus loading, pairing, held-out OOD set
  evaluate.py    metrics, baselines, failure extraction
scripts/run_eval.py
tests/
results/metrics.json
```

## Sources

Rubric derived from Wikipedia's [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), maintained by WikiProject AI Cleanup, plus a house editing standard. Corpora: [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) (Guo et al., 2023) and [Ateeqq/AI-and-Human-Generated-Text](https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text).

MIT licensed.
