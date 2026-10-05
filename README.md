# Ukrainian Small Transformer

This is a small GPT-2-style model I trained on Ukrainian web text. I wanted to see what happens when I train the same model on more tokens.

## Results so far

All three checkpoints use the same 3M model setup. The thing that changes is how much text the model sees during training.

| Training tokens | Test loss | Perplexity |
| ---: | ---: | ---: |
| 10M | 6.0198 | 411.50 |
| 100M | 4.5218 | 92.01 |
| 500M | 4.2491 | 70.04 |

Both test metrics go down as the training set gets larger. I describe the setup, the results, and a few real text samples in [RESEARCH.md](RESEARCH.md).

I also tried an English prompt, `Ukraine is`. The model's replies look like gibberish. I trained it on FineWeb-2's Ukrainian `ukr_Cyrl` subset and did not add a separate English dataset. The report shows what it generated.

## What's in the project

- `data_prep.ipynb` prepares the text, trains the tokenizer, and writes the train, validation, and test files.
- `create_model.py` contains the 3M, 9M, and 20M model settings.
- `model_prep.ipynb` trains a model, saves the best checkpoint, and evaluates it.
- `model_compare.ipynb` checks the checkpoints that are available and saves text samples.
- `results/model_comparison_results.csv` has the test metrics.
- `results/model_comparison_prompt_results.csv` has the generated text.
- `data/`, `models/`, and `tokenizer/` hold local data and model files. Git ignores these, so they need to be created or downloaded separately.

I only have checkpoints and results for the 3M setup so far. The 9M and 20M settings are in the code, but I haven't trained them yet.

## Running it

Install the packages in `requirements.txt`, then run the notebooks in this order: `data_prep.ipynb`, `model_prep.ipynb`, and `model_compare.ipynb`. Preparing the data and training the models take disk space and compute time.

The training notebook currently uses `TRAIN_TOKENS = 100M`. To repeat the 10M or 500M runs, change this setting and save each checkpoint with the matching budget in its name.
