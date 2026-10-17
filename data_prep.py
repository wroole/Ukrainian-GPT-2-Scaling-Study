# %%
from datasets import load_dataset
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer


# %%
TOKENIZER_DOCS = 100000

DATA_DIR = Path("data")
TOKENIZER_DIR = Path("tokenizer")

DATA_DIR.mkdir(exist_ok=True)

TRAIN_DATA_PATH = DATA_DIR / "train_500M.bin"

VAL_DATA_DIR = DATA_DIR / "validation"
TEST_DATA_DIR = DATA_DIR / "test"
TEST_DATA_DIR.mkdir(exist_ok=True)
VAL_DATA_DIR.mkdir(exist_ok=True)

VAL_DATA_PATH = VAL_DATA_DIR / "validation_5M.bin"
TEST_DATA_PATH = TEST_DATA_DIR / "test_5M.bin"

TOKENS_500M = 500_000_000
VAL_TOKENS = 5_000_000
TEST_TOKENS = 5_000_000


# %%
tokenizer_path = TOKENIZER_DIR / "tokenizer.json"

tokenizer = Tokenizer.from_file(str(tokenizer_path))

print(f"Tokenizer loaded from: {tokenizer_path}")
print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")


# %%
if TRAIN_DATA_PATH.exists():
    tokens = np.memmap(
        TRAIN_DATA_PATH,
        dtype=np.uint16,
        mode="r+",
        shape=(TOKENS_500M,),
    )
else:
    tokens = np.memmap(
        TRAIN_DATA_PATH,
        dtype=np.uint16,
        mode="w+",
        shape=(TOKENS_500M,),
    )
    
train_data = load_dataset(
    "HuggingFaceFW/fineweb-2",
    "ukr_Cyrl",
    split="train",
    streaming=True,
)

train_data = train_data.skip(TOKENIZER_DOCS)

eos_id = tokenizer.token_to_id("[EOS]")
position = 0
documents = 0
for row in train_data:
    text = row["text"].strip()

    if not text:
        continue

    encoded = tokenizer.encode(text)
    
    ids = encoded.ids

    ids.append(eos_id)

    remaining = TOKENS_500M - position

    ids = ids[:remaining]

    tokens[position:position + len(ids)] = ids

    position += len(ids)
    documents += 1

    if documents % 1000 == 0:
        print(
            f"Документів: {documents:,} | "
            f"Токенів: {position:,} / {TOKENS_500M:,}"
        )

    if position >= TOKENS_500M:
        break
tokens.flush()
print(f"Готово: {position:,} токенів із {documents:,} документів")


# %%
if VAL_DATA_PATH.exists():
    val_tokens = np.memmap(
        VAL_DATA_PATH,
        dtype=np.uint16,
        mode="r+",
        shape=(VAL_TOKENS,),
    )
else:
    val_tokens = np.memmap(
        VAL_DATA_PATH,
        dtype=np.uint16,
        mode="w+",
        shape=(VAL_TOKENS,),
    )
val_data = load_dataset(
    "HuggingFaceFW/fineweb-2",
    "ukr_Cyrl",
    split="train",
    streaming=True,
)
val_data = val_data.skip(TOKENIZER_DOCS + documents)
position = 0
val_documents = 0

for row in val_data:
    text = row["text"].strip()

    if not text:
        continue

    ids = tokenizer.encode(text).ids
    ids.append(eos_id)

    remaining = VAL_TOKENS - position
    ids = ids[:remaining]

    val_tokens[position:position + len(ids)] = ids

    position += len(ids)
    val_documents += 1
    
    if position >= VAL_TOKENS:
            break
    
val_tokens.flush()
print(f"Готово: {position:,} токенів із {val_documents:,} документів")


# %%
if TEST_DATA_PATH.exists():
    test_tokens = np.memmap(
        TEST_DATA_PATH,
        dtype=np.uint16,
        mode="r+",
        shape=(TEST_TOKENS,),
    )
else:
    test_tokens = np.memmap(
        TEST_DATA_PATH,
        dtype=np.uint16,
        mode="w+",
        shape=(TEST_TOKENS,),
    )
test_data = load_dataset(
    "HuggingFaceFW/fineweb-2",
    "ukr_Cyrl",
    split="train",
    streaming=True,
)
test_data = test_data.skip(TOKENIZER_DOCS + documents + val_documents)
position = 0
test_documents = 0

for row in test_data:
    text = row["text"].strip()

    if not text:
        continue

    ids = tokenizer.encode(text).ids
    ids.append(eos_id)

    remaining = TEST_TOKENS - position
    ids = ids[:remaining]

    test_tokens[position:position + len(ids)] = ids

    position += len(ids)
    test_documents += 1
    
    if position >= TEST_TOKENS:
            break
    
test_tokens.flush()
print(f"Готово: {position:,} токенів із {test_documents:,} документів")
