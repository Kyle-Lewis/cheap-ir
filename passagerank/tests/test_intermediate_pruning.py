from copy import deepcopy

import pytest
import torch
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaLayer
)
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertConfig,
    TransformerBlock
)

import passagerank.pruning.intermediate_pruning as intermediate_pruning
import torch.nn.utils.prune as prune


params = [0.01, 0.05, 0.1, 0.5]
@pytest.mark.parametrize("amount", params)
def test_neuron_pruning_roberta(amount):
    """ Test that my method for 'neuron pruning' results in re-shaped linear
    layers which, when applied in sequence to an input signal, yield the same
    output signal that would have resulted from masking using torch's pruning
    """

    sample = torch.rand((1, 512, 768))
    
    # Both layers need to be in evaluation mode s.t. no random dropout is used
    config = RobertaConfig()
    masked_layer = RobertaLayer(config).eval()
    pruned_layer = deepcopy(masked_layer).eval()
    
    # The reshaped matrices should be equivalent to having masked out rows and
    # biases from the first linear
    lin1 = masked_layer.intermediate.dense
    _ = prune.ln_structured(lin1, name='weight', amount=amount, dim=0, n=1)
    _ = prune.remove(lin1, 'weight')

    importance_scores = lin1.weight.abs().sum(dim=1)
    _ = prune.l1_unstructured(
        lin1, name='bias', amount=amount, importance_scores=importance_scores)
    _ = prune.remove(lin1, 'bias')

        
    out_masked = masked_layer.forward(sample)[0].data

    intermediate_pruning.prune_intermediate_neurons(pruned_layer, amount=amount)
    intermediate_pruning.finalize_layer(pruned_layer)

    out_pruned = pruned_layer.forward(sample)[0].data

    # I suspect the outputs are not exactly the same due to floating point
    # operations. That the outputs are equal within a low tolerance seems good
    # enough to me, but I could have made a mistake!
    assert torch.allclose(out_masked, out_pruned, atol=1e-6)
    assert (pruned_layer.intermediate.dense.weight.nelement() <
            masked_layer.intermediate.dense.weight.nelement())


params = [0.01, 0.05, 0.1, 0.5]
@pytest.mark.parametrize("amount", params)
def test_structured_pruning_distilbert(amount):

    config = DistilBertConfig()
    masked_layer = TransformerBlock(config).eval()
    pruned_layer = deepcopy(masked_layer).eval()

    lin1 = masked_layer.ffn.lin1
    _ = prune.ln_structured(lin1, name='weight', amount=amount, dim=0, n=1)
    _ = prune.remove(lin1, 'weight')

    importance_scores = lin1.weight.abs().sum(dim=1)
    _ = prune.l1_unstructured(
        lin1, name='bias', amount=amount, importance_scores=importance_scores)
    _ = prune.remove(lin1, 'bias')

    # For calling a distilbert layer in isolation I have to provide an attention
    # mask due to assumptions in the internals of modeling_distilbert.py 
    sample = torch.rand((1, 512, 768))
    attn_mask = torch.ones((1, 512))
    out_masked = masked_layer.forward(sample, attn_mask=attn_mask)[0].data

    intermediate_pruning.prune_intermediate_neurons(pruned_layer, amount=amount)
    intermediate_pruning.finalize_layer(pruned_layer)

    out_pruned = pruned_layer.forward(sample, attn_mask=attn_mask)[0].data

    # I suspect the outputs are not exactly the same due to floating point
    # operations. That the outputs are equal within a low tolerance seems good
    # enough to me, but I could have made a mistake!
    assert torch.allclose(out_masked, out_pruned, atol=1e-6)
    assert (pruned_layer.ffn.lin1.weight.nelement() <
            masked_layer.ffn.lin1.weight.nelement())
