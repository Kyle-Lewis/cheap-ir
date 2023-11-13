import pytest
import numpy as np

import passagerank.pruning.head_pruning as head_pruning

def mock_attention(head_mask):
    """ fake out some attention data for test purposes aranged such that the
        smallest index head in the largest indexed layer recieves the lowest scores
        HF models return attention scores as a tuple of B x nH x S x S matrices
    """
    batch_size = 4
    context_size = 4
    n_layers = head_mask.shape[0]
    attentions = []
    totelems = 0
    for layer_i in range(n_layers):
        n_active_heads = np.sum(head_mask[layer_i])
        nelems = batch_size * n_active_heads * context_size * context_size
        totelems += nelems
        att = np.arange(nelems, dtype=np.float32) + nelems * (n_layers - layer_i)**2
        att = np.reshape(att, (batch_size, n_active_heads, context_size, context_size))
        attentions.append(att)
    
    return tuple(attentions)

# mocks 2 layers of 4 heads each, with 2x2 attention for a context window of 2
params = [
    ([[1, 1, 1, 1], [1, 1, 1, 1]]),
    ([[1, 1, 1, 1], [1, 1, 1, 0]]),
    ([[1, 1, 1, 1], [0, 1, 1, 1]]),
    ([[0, 1, 1, 1], [0, 1, 1, 1]]),
    ([[0, 0, 0, 1], [0, 0, 0, 1]]),
    ([[0, 1, 0, 0], [0, 0, 1, 0]]),
]
@pytest.mark.parametrize("head_mask", params)
def test_accumulate_scores(head_mask):
    head_mask = np.asarray(head_mask, dtype=np.int32)
    head_scores = np.zeros_like(head_mask, dtype=np.float32)
    orig_scores = np.copy(head_scores)
    attentions = mock_attention(head_mask)
    
    head_pruning.accumulate_scores(head_mask, head_scores, attentions)

    for li in range(head_mask.shape[0]):
        prev_score = 0
        for hi in range(head_mask.shape[1]):

            # Scores always increase for unmasked heads, never for masked
            if head_mask[li, hi] == 1:
                assert head_scores[li, hi] > orig_scores[li, hi]
            else:
                assert head_scores[li, hi] == orig_scores[li, hi]
            
            # Scores are aligned to the proper head when there were masked heads
            # among them, the fixture creates them in increasing order per layer
            if head_mask[li, hi] == 1:
                assert head_scores[li, hi] > prev_score
                prev_score = head_scores[li, hi]



