import wandb
import torch

from typing import List, Dict, Literal
from transformers import PreTrainedModel

from src.state import State
from src import TransformersModel
from src.lens import *


# Lenses that can be trained (the LogitLens has no parameters)
LENS_TYPE = Literal["TunedLens", "LoRALens", "AnchorLens"]


def gather_weights(lens: BaseLens):
    """
    Frobenius norms of the lens parameters, keyed by their wandb name.

    Per-layer keys use the zero-padded layer index (e.g. "weight_norm/007.hidden.down"), so wandb
    sorts them in layer order. Global AnchorLens parameters go under "<...>/global.hidden".
    """
    weights = {}

    if isinstance(lens, TunedLens):
        for layer_idx, translator in enumerate(lens.get_translators()):
            weights[f"weight_norm/{str(layer_idx).zfill(3)}.hidden"] = translator.weight.data.norm().item()
            if lens.config.bias:
                weights[f"bias_norm/{str(layer_idx).zfill(3)}.hidden"] = translator.bias.data.norm().item()
    elif isinstance(lens, LoRALens):
        for layer_idx, (translator, bias) in enumerate(zip(*lens.get_translators())):
            weights[f"weight_norm/{str(layer_idx).zfill(3)}.hidden.down"] = translator[0].weight.data.norm().item()
            weights[f"weight_norm/{str(layer_idx).zfill(3)}.hidden.up"] = translator[1].weight.data.norm().item()
            if bias is not None:
                weights[f"bias_norm/{str(layer_idx).zfill(3)}.hidden"] = bias.data.norm().item()
    elif isinstance(lens, AnchorLens):
        global_transform = lens.get_global_transform()
        if global_transform is not None:
            weights["weight_norm/global.hidden"] = global_transform.weight.data.norm().item()
            if global_transform.bias is not None:
                weights["bias_norm/global.hidden"] = global_transform.bias.data.norm().item()
        locals_ = lens.get_locals()
        if locals_ is not None:
            biases = lens.get_bias() if lens.get_bias() is not None else [None] * len(locals_)
            for layer_idx, (local, bias) in enumerate(zip(locals_, biases)):
                weights[f"weight_norm/{str(layer_idx).zfill(3)}.hidden.down"] = local[0].weight.data.norm().item()
                weights[f"weight_norm/{str(layer_idx).zfill(3)}.hidden.up"] = local[1].weight.data.norm().item()
                if bias is not None:
                    weights[f"bias_norm/{str(layer_idx).zfill(3)}.hidden"] = bias.data.norm().item()

    return weights


def gather_gradients(lens: BaseLens):
    """
    Gradient norms of the lens parameters, keyed like in `gather_weights` but under "grad_norm/".

    Must be called after backward and before the optimizer zeroes the gradients. In the training
    loop this happens after gradient clipping, so the logged norms are the clipped ones.
    """
    grads = {}

    if isinstance(lens, TunedLens):
        for layer_idx, translator in enumerate(lens.get_translators()):
            grads[f"grad_norm/{str(layer_idx).zfill(3)}.hidden.weight"] = translator.weight.grad.norm().item()
            if lens.config.bias:
                grads[f"grad_norm/{str(layer_idx).zfill(3)}.hidden.bias"] = translator.bias.grad.norm().item()
    elif isinstance(lens, LoRALens):
        for layer_idx, (translator, bias) in enumerate(zip(*lens.get_translators())):
            grads[f"grad_norm/{str(layer_idx).zfill(3)}.hidden.down"] = translator[0].weight.grad.norm().item()
            grads[f"grad_norm/{str(layer_idx).zfill(3)}.hidden.up"] = translator[1].weight.grad.norm().item()
            if bias is not None:
                grads[f"grad_norm/{str(layer_idx).zfill(3)}.hidden.bias"] = bias.grad.norm().item()
    elif isinstance(lens, AnchorLens):
        # Components can be frozen (e.g. the global transform while training locals), so skip missing grads
        global_transform = lens.get_global_transform()
        if global_transform is not None:
            if global_transform.weight.grad is not None:
                grads["grad_norm/global.hidden.weight"] = global_transform.weight.grad.norm().item()
            if global_transform.bias is not None and global_transform.bias.grad is not None:
                grads["grad_norm/global.hidden.bias"] = global_transform.bias.grad.norm().item()
        locals_ = lens.get_locals()
        if locals_ is not None:
            biases = lens.get_bias() if lens.get_bias() is not None else [None] * len(locals_)
            for layer_idx, (local, bias) in enumerate(zip(locals_, biases)):
                if local[0].weight.grad is not None:
                    grads[f"grad_norm/{str(layer_idx).zfill(3)}.hidden.down"] = local[0].weight.grad.norm().item()
                if local[1].weight.grad is not None:
                    grads[f"grad_norm/{str(layer_idx).zfill(3)}.hidden.up"] = local[1].weight.grad.norm().item()
                if bias is not None and bias.grad is not None:
                    grads[f"grad_norm/{str(layer_idx).zfill(3)}.hidden.bias"] = bias.grad.norm().item()

    return grads


def log_train_step(state: State, losses: Dict[str, List[float]]):
    """
    Log one optimizer step to wandb: progress, LR, per-layer losses, weight and gradient norms.

    Args:
        state: The training state, before `state.step` is incremented for this step.
        losses: Loss values in nats per token, one per forward pass of this step, per layer key
            (e.g. "007.hidden"). They are averaged and logged in bits per byte.
    """
    log_dict = {
        "general/step": state.step + 1,
        "general/progress": (state.step + 1) / state.num_steps * 100,
        # The first optimizer is the main one (Muon, when used); the AdamW for the biases runs at a 10x lower LR
        "general/learning_rate": state.opts[0].param_groups[0]["lr"],
        **{f"loss/{k}": torch.tensor(v).mean() * state.nats_to_bpb for k, v in losses.items()},
        **gather_weights(state.lens),
        **gather_gradients(state.lens),
    }

    wandb.log(log_dict)
