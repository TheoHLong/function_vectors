from baukit import TraceDict, get_module
import torch
import re
import bitsandbytes as bnb

def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)

def replace_activation_w_avg(layer_head_token_pairs, avg_activations, model, model_config, idx_map, batched_input=False, last_token_only=False):
    """
    An intervention function for replacing activations with a computed average value.
    This function replaces the output of one (or several) attention head(s) with a pre-computed average value 
    """
    edit_layers = [x[0] for x in layer_head_token_pairs]

    def rep_act(output, layer_name, inputs):
        current_layer = int(layer_name.split('.')[2])
        if current_layer in edit_layers:    
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            
            # Determine shapes for intervention
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads'])
            inputs = inputs.view(*new_shape)
            
            # Perform Intervention:
            if batched_input:
                for i in range(model_config['n_heads']):
                    layer, head_n, token_n = layer_head_token_pairs[i]
                    inputs[i, token_n, head_n] = avg_activations[layer, head_n, idx_map[token_n]]
            elif last_token_only:
                for (layer,head_n,token_n) in layer_head_token_pairs:
                    if layer == current_layer:
                        inputs[-1,-1,head_n] = avg_activations[layer,head_n,idx_map[token_n]]
            else:
                for (layer, head_n, token_n) in layer_head_token_pairs:
                    if layer == current_layer:
                        inputs[-1, token_n, head_n] = avg_activations[layer,head_n,idx_map[token_n]]
            
            inputs = inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight

            # Handle different model architectures
            if 'gpt2-xl' in model_config['name_or_path']:
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj)
            elif any(name in model_config['name_or_path'].lower() for name in ['gpt-j', 'gemma', 'llama-3.2', 'llama-3']):
                new_output = torch.matmul(inputs, out_proj.T)
            elif 'gpt-neox' in model_config['name_or_path'] or 'pythia' in model_config['name_or_path']:
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj.T)
            elif 'llama' in model_config['name_or_path']:
                if isinstance(out_proj, bnb.nn.Linear4bit):
                    out_proj_dequant = bnb.functional.dequantize_4bit(out_proj.weight.data, out_proj.weight.quant_state)
                    new_output = torch.matmul(inputs, out_proj_dequant.T)
                else:
                    new_output = torch.matmul(inputs, out_proj.T)
            
            return new_output
        else:
            return output

    return rep_act

def add_function_vector(edit_layer, fv_vector, device, idx=-1):
    """
    Adds a vector to the output of a specified layer in the model
    """
    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if current_layer == edit_layer:
            if isinstance(output, tuple):
                output[0][:, idx] += fv_vector.to(device)
                return output
            else:
                output[:, idx] += fv_vector.to(device)
                return output
        else:
            return output

    return add_act

def function_vector_intervention(sentence, target, edit_layer, function_vector, model, model_config, tokenizer, compute_nll=False,
                                  generate_str=False):
    """
    Runs the model on the sentence and adds the function_vector to the output of edit_layer as a model intervention.
    """
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1

    if compute_nll:
        target_completion = "".join(sentence + target)
        nll_inputs = tokenizer(target_completion, return_tensors='pt').to(device)
        nll_targets = nll_inputs.input_ids.clone()
        target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze()) 
        nll_targets[:,:-target_len] = -100
        output = model(**nll_inputs, labels=nll_targets)
        clean_nll = output.loss.item()
        clean_output = output.logits[:,original_pred_idx,:]
        intervention_idx = -1 - target_len
    elif generate_str:
        MAX_NEW_TOKENS = 16
        output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,
                                max_new_tokens=MAX_NEW_TOKENS)
        clean_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        intervention_idx = -1
    else:
        clean_output = model(**inputs).logits[:,-1,:]
        intervention_idx = -1

    # Perform Intervention
    intervention_fn = add_function_vector(edit_layer, function_vector.reshape(1, model_config['resid_dim']), model.device, idx=intervention_idx)
    with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):     
        if compute_nll:
            output = model(**nll_inputs, labels=nll_targets)
            intervention_nll = output.loss.item()
            intervention_output = output.logits[:,original_pred_idx,:]
        elif generate_str:
            output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,
                                    max_new_tokens=MAX_NEW_TOKENS)
            intervention_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        else:
            intervention_output = model(**inputs).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction
    
    fvi_output = (clean_output, intervention_output)
    if compute_nll:
        fvi_output += (clean_nll, intervention_nll)
    
    return fvi_output

def fv_intervention_natural_text(sentence, edit_layer, function_vector, model, model_config, tokenizer, max_new_tokens=16, num_interv_tokens=None, do_sample=False):
    """
    Allows for intervention in natural text where we generate and intervene on several tokens in a row.
    """
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)    
    clean_output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    # Perform Intervention
    intervention_fn = add_function_vector(edit_layer, function_vector, model.device)
    
    if num_interv_tokens is not None and num_interv_tokens < max_new_tokens:
        num_extra_tokens = max_new_tokens - num_interv_tokens
        with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):     
            intervention_output = model.generate(**inputs, max_new_tokens = num_interv_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
        intervention_output = model.generate(intervention_output, max_new_tokens=num_extra_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=do_sample)
    else:
        with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):     
            intervention_output = model.generate(**inputs, max_new_tokens = max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)

    return clean_output, intervention_output

def add_avg_to_activation(layer_head_token_pairs, avg_activations, model, model_config, batched_input=False, last_token_only=False):
    """
    An intervention function for adding a computed average value to activations.
    """
    edit_layers = [x[0] for x in layer_head_token_pairs]
    device = model.device

    def add_act(output, layer_name, inputs):
        current_layer = int(layer_name.split('.')[2])
        if current_layer in edit_layers:    
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads'])
            inputs = inputs.view(*new_shape)
            
            if batched_input:
                for i in range(model_config['n_heads']):
                    layer, head_n, token_n = layer_head_token_pairs[i]
                    inputs[i, token_n, head_n] += avg_activations[layer, head_n, token_n].to(device)
            elif last_token_only:
                for (layer,head_n,token_n) in layer_head_token_pairs:
                    if layer == current_layer:
                        inputs[-1,-1,head_n] += avg_activations[layer,head_n,token_n].to(device)
            else:
                for (layer, head_n, token_n) in layer_head_token_pairs:
                    if layer == current_layer:
                        inputs[-1, token_n, head_n] += avg_activations[layer,head_n,token_n].to(device)
            
            inputs = inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight

            # Handle different model architectures 
            if 'gpt2-xl' in model_config['name_or_path']:
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj)
            elif any(name in model_config['name_or_path'].lower() for name in ['gpt-j', 'gemma', 'llama-3.2', 'llama-3']):
                new_output = torch.matmul(inputs, out_proj.T)
            elif 'gpt-neox' in model_config['name_or_path'] or 'pythia' in model_config['name_or_path']:
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj.T)
            elif 'llama' in model_config['name_or_path']:
                if isinstance(out_proj, bnb.nn.Linear4bit):
                    out_proj_dequant = bnb.functional.dequantize_4bit(out_proj.weight.data, out_proj.weight.quant_state)
                    new_output = torch.matmul(inputs, out_proj_dequant.T)
                else:
                    new_output = torch.matmul(inputs, out_proj.T)
            
            return new_output
        else:
            return output

    return add_act