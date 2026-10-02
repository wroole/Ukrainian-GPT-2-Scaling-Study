from transformers import GPT2Config, GPT2LMHeadModel


MODEL_CONFIGS = {
    "3M": {
        "n_layer": 6,
        "n_head": 4,
        "n_embd": 128,
        "n_inner": 512,
    },

    "9M": {
        "n_layer": 9,
        "n_head": 7,
        "n_embd": 224,
        "n_inner": 896,
    },

    "20M": {
        "n_layer": 12,
        "n_head": 10,
        "n_embd": 320,
        "n_inner": 1280,
    },
}

def create_model(model_size: str):

    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Choose from {list(MODEL_CONFIGS.keys())}"
        )

    config = GPT2Config(
        vocab_size=16_000,
        n_positions=256,
        n_ctx=256,

        n_embd=MODEL_CONFIGS[model_size]["n_embd"],
        n_layer=MODEL_CONFIGS[model_size]["n_layer"],
        n_head=MODEL_CONFIGS[model_size]["n_head"],
        n_inner=MODEL_CONFIGS[model_size]["n_inner"],

        activation_function="gelu_new",

        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,

        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,

        use_cache=False,
    )

    model = GPT2LMHeadModel(config)

    return model