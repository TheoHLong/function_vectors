import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import random
from typing import *


def load_gpt_model_and_tokenizer(model_name:str, device='cuda'):
    """
    Loads a huggingface model and its tokenizer

    Parameters:
    model_name: huggingface name of the model to load (e.g. GPTJ: "EleutherAI/gpt-j-6B", or "EleutherAI/gpt-j-6b")
    device: 'cuda' or 'cpu'
    
    Returns:
    model: huggingface model
    tokenizer: huggingface tokenizer
    MODEL_CONFIG: config variables w/ standardized names
    """
    assert model_name is not None

    print("Loading: ", model_name)

    if model_name == 'gpt2-xl':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)],
                      "prepend_bos":False}
        
    elif 'gpt-j' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)

        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.out_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)],
                      "prepend_bos":False}

    elif 'gpt-neox' in model_name.lower() or 'pythia' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

        MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                      "n_layers":model.config.num_hidden_layers,
                      "resid_dim": model.config.hidden_size,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'gpt_neox.layers.{layer}.attention.dense' for layer in range(model.config.num_hidden_layers)],
                      "layer_hook_names":[f'gpt_neox.layers.{layer}' for layer in range(model.config.num_hidden_layers)], 
                      "prepend_bos":False}
    
    # Add support for unsloth/Llama-3.2-3B-Instruct
    elif 'unsloth/llama-3.2' in model_name.lower():
        print("Loading unsloth Llama 3.2...")
        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/Llama-3.2-3B-Instruct", 
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Load model in 16-bit precision for efficiency
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-3B-Instruct",
            device_map="auto",  # Automatically handle device placement
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        MODEL_CONFIG = {
            "n_heads": model.config.num_attention_heads,
            "n_layers": model.config.num_hidden_layers,
            "resid_dim": model.config.hidden_size,
            "name_or_path": model.config._name_or_path,
            # Define hook points for the unsloth Llama model
            "attn_hook_names": [f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
            "layer_hook_names": [f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
            "prepend_bos": True
        }

        print(f"Model config: {MODEL_CONFIG['n_layers']} layers, {MODEL_CONFIG['n_heads']} heads")

    else:
        raise NotImplementedError("Model is not yet supported!")

    return model, tokenizer, MODEL_CONFIG


def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments.

    Parameters:
    seed: the number to set the seed to

    Return: None
    """
    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)