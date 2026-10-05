# My Small Ukrainian Language Model

This is a write-up of what I built and what I found. I trained one small GPT-2-style model on Ukrainian web text, then compared runs that used different amounts of training data.

One thing to make clear: 10M, 100M, and 500M are the numbers of training tokens. All three checkpoints use the same 3M model configuration.

## What I wanted to find out

Does the 3M model get better at predicting the next token when I train it on more text?

My expectation was that more training data would help, but that the gains might get smaller as the training set grows. The results below are from the runs I have saved so far.

## The data

I used the streaming version of `HuggingFaceFW/fineweb-2` with the `ukr_Cyrl` configuration.

The first 100,000 non-empty documents make up the tokenizer training text. After that, I take documents in order and write:

- 500M tokens for training;
- the next 5M tokens for validation;
- the next 5M tokens for testing.

The code separates these parts by skipping documents in the stream. It stores the token IDs as `uint16` files and adds an `[EOS]` token at the end of each document.

I trained a Byte-Level BPE tokenizer with a vocabulary of 16,000 tokens. Its special token IDs are `[UNK]` = 0, `[BOS]` = 1, `[EOS]` = 2, and `[PAD]` = 3. The tokenizer is saved in `tokenizer/tokenizer.json`.

## The model and how I trained it

The model is Hugging Face's `GPT2LMHeadModel`, trained to predict the next token. I defined three model configurations, but only have checkpoints for the 3M version right now.

| Setup | Layers | Attention heads | Hidden size | MLP size | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| 3M | 6 | 4 | 128 | 512 | Checkpoints and test results available |
| 9M | 9 | 7 | 224 | 896 | Defined in code; not trained yet |
| 20M | 12 | 10 | 320 | 1280 | Defined in code; not trained yet |

The 3M model has a 16,000-token vocabulary and a context length of 256. I turned dropout off and used the `gelu_new` activation. Each training step processes 32 sequences of 256 tokens, or 8,192 tokens total.

That works out to about 1,221 steps for 10M tokens, 12,208 for 100M, and 61,036 for 500M. I used AdamW, a learning rate that warms up to 0.001 and then drops to 0.0001, and gradient clipping at 1. The notebook checks validation loss every 250 steps and saves the checkpoint with the best validation loss.

In the code, I pass `labels=x` to GPT-2. The model shifts the labels internally when it calculates next-token loss, so the separate `y` array is not passed to the model.

## How I checked the models

I evaluated each checkpoint on the same 5M-token test file, `data/test/test_5M.bin`. The code reads non-overlapping windows of 256 tokens, in batches of 32. It reports test loss and perplexity; perplexity is `exp(loss)`.

I also tried five prompts: “Столиця України”, “Президент”, “Одного вечора”, “Найбільша річка в Україні”, and `Ukraine is`. The model can generate up to 100 new tokens. I used temperature 0.8, top-k 50, and top-p 0.95.

## Test results

| Model setup | Training tokens | Test loss | Perplexity |
| --- | ---: | ---: | ---: |
| 3M | 10M | 6.0198 | 411.50 |
| 3M | 100M | 4.5218 | 92.01 |
| 3M | 500M | 4.2491 | 70.04 |

These numbers are in `results/model_comparison_results.csv`. The biggest improvement is between 10M and 100M tokens: test loss drops by about 24.9%, and perplexity drops by about 77.6%. Going from 100M to 500M helps too, but the change in test loss is smaller: about 6.0%. Across the full range, perplexity drops by about 83.0%.

So far, more training tokens have meant better test scores for this model. That is what happened in these three runs; three points are not enough to claim a general rule.

## What the generated text looks like

Here are short excerpts from `results/model_comparison_prompt_results.csv`. Each checkpoint got the same prompt, “Одного вечора”. You can see that the continuations change with the training budget, though none of these examples is especially clean.

| Training tokens | Sample |
| ---: | --- |
| 10M | “Одного вечора, в житті не з'яті, а й в п’юю, не лише в п'ять.” |
| 100M | “Одного вечора, що ви не зможете віднайти, коли ви не можете буде насолоджуватися.” |
| 500M | “Одного вечора в Одесі підвалилась до Львова.” |

These are sampled examples, not a score of writing quality. Generation is random, and I did not set a seed, so another run may produce different text. The test metrics above are a better way to compare these checkpoints.

## Why the English prompt looks strange

I trained the model on FineWeb-2's Ukrainian `ukr_Cyrl` subset. I did not add an English training set, so `Ukraine is` is outside the language the model was trained for. Here are the saved results:

| Training tokens | Continuation after `Ukraine is` |
| ---: | --- |
| 10M | “the in frodis a ses-ss to wagned for brors of the a pritle.” |
| 100M | “feachespution.” It then switches to Ukrainian text. |
| 500M | “the evel, which a point and oney.” It then mixes English and Ukrainian. |

In short, the model was trained for Ukrainian text, not English. These examples are what I would expect from giving it an English prompt without providing English training data.

## Things to keep in mind

- I have one run for each training budget, so I do not know how much the results would vary across repeat runs.
- The notebook defines `SEED = 42` but does not set the NumPy or PyTorch random seeds. Training batches and generated samples may differ between runs.
- I did not pin a FineWeb-2 dataset revision, so a future download may not have exactly the same content or order.
- I have only trained the 3M configuration. The 9M and 20M settings do not have checkpoints yet.
- The checked-in `model_prep.ipynb` currently uses 100M for `TRAIN_TOKENS`. To reproduce the 10M and 500M runs, I need to change this setting and save each checkpoint with its training budget in the filename.
- The prompt examples are short, randomly sampled outputs. They do not measure factual accuracy or writing quality on their own.
- The saved notebook output records PyTorch 2.14.0+cu130, Transformers 5.17.0, and an NVIDIA GeForce RTX 4060 Laptop GPU. `requirements.txt` does not pin package versions.

## Files with results

`results/model_comparison_results.csv` has the three test scores. `results/model_comparison_prompt_results.csv` has 15 generated samples: five prompts for each training budget. Git ignores the datasets, tokenizer, checkpoints, and results, so they need to be saved or recreated separately.
