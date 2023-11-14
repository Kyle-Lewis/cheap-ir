"""
torch.prune.remove() and the torch pruning module only *zeros* weights.

This is fine during training, but when evaluating inference time I want to
actually resize the matrices in such a way that the replacement is
mathematically equivalent (and hopefully equivalent to floating point accuracy)
and requires fewer operations.
"""
import torch.nn.utils.prune as prune

import passagerank.pruning.utils as pruning_utils


class IntermediatesPruner:

    def __init__(self, model, exclude_layers=None):

        self.model = model

        self.exclude_layers = exclude_layers if exclude_layers is not None else []
    

    def prune_intermediate_layers(self, amount):
        """ """

        encoder_layers = pruning_utils.get_layers_for_model(self.model)
        
        for layer_num in range(self.model.config.num_hidden_layers):
            if layer_num in self.exclude_layers:
                continue
                
            prune_intermediate_neurons(encoder_layers[layer_num], amount)

    
def prune_intermediate_neurons(layer, amount):
    """ """
    lin1, _lin2 = pruning_utils.get_linears_for_layer(layer)
    _ = prune.ln_structured(lin1, name='weight', amount=amount, dim=0, n=1)

    importance_scores = lin1.weight_mask.abs().sum(dim=1)
    _ = prune.l1_unstructured(
        lin1, name='bias', amount=amount, importance_scores=importance_scores)


def get_prune_schedule(n_steps, n_cycles, target_sparsity, hidden_dim, warmup_steps=0):
    
    schedule = []

    prune_steps = (n_steps - warmup_steps) // n_cycles    
    target_dim = int((1.0 - target_sparsity) * hidden_dim)
    cycle = 0

    div = (hidden_dim - target_dim) // n_cycles
    mod = (hidden_dim - target_dim) % n_cycles
    mod_sum = 0.0

    tot_pruned = 0
    while len(schedule) < n_cycles - 1:

        step = warmup_steps + (prune_steps * cycle)
        n_prune = div
        mod_sum += mod / n_cycles
        if mod_sum >= 1.0:
            n_prune += 1
            mod_sum -= 1.0
        schedule.append((step, n_prune))
        tot_pruned += n_prune
        cycle+=1
    
    step = warmup_steps + (prune_steps * cycle)
    remaining = hidden_dim - tot_pruned
    n_prune = remaining - target_dim
    schedule.append((step, n_prune))

    return schedule


def finalize_model(model, exclude_layers=None):
    """ Call on a loaded checkpoint which has had torch.prune.remove() applied
        to permanantly remove the zero rows and columns (neurons) from the
        linear layers.
    """
    if exclude_layers is None:
        exclude_layers = []

    encoder_layers = pruning_utils.get_layers_for_model(model)
    for layer_num in range(model.config.num_hidden_layers):
        if layer_num in exclude_layers:
            continue
        finalize_layer(encoder_layers[layer_num])


def finalize_layer(layer):
    """ Reshape the two linears pairwise to remove pruned neurons structurally """

    lin1, lin2 = pruning_utils.get_linears_for_layer(layer)
    prune.remove(lin1, 'weight')
    prune.remove(lin1, 'bias')
    keep_indices = lin1.weight.abs().sum(dim=1) != 0
    lin1.weight.data = lin1.weight[keep_indices, :]
    lin1.bias.data = lin1.bias[keep_indices]
    lin2.weight.data = lin2.weight[:, keep_indices]