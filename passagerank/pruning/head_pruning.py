""" Class to facilitate head pruning during training with confidence scores
Usage, accumulating for 5 batches then pruning:

    pruner = HeadPruner(model)
    for i in range(10):
    
        if i < 5:
            outputs = model(batch)
            # loss, grad, etc.
        
        else:
            outputs = model(batch, output_attentions=True)
            # loss, grad, etc.

            pruner.accumulate(outputs)
    
    pruner.prune_head()  # <- removes head with the lowest average score 
"""

import torch
import numpy as np
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

import passagerank.pruning.utils as pruning_utils


class HeadPruner:

    def __init__(self, model: PreTrainedModel):
        self.model = model

        self.head_mask = get_pruned_head_mask(self.model)
        self.head_scores = np.zeros((
            model.config.num_hidden_layers,
            model.config.num_attention_heads),
            dtype=np.float32
        )


    def accumulate(self, model_outputs: BaseModelOutput):
        """ Accumulate head scores given the attention scores in model outputs

        The forward pass must have been called with output_attentions=True:
            model(**batch, output_attentions=True)
        """
        cpu = torch.device('cpu')
        batch_attentions = \
            [a.to(cpu).detach().numpy() for a in model_outputs.attentions]
        accumulate_scores(self.head_mask, self.head_scores, batch_attentions)

    
    def prune_head(self):
        """ After accumulate, prune one head and reset head mask and scores """

        layer_i, head_i = head_to_prune(self.head_mask, self.head_scores)
        prune_head = {layer_i: {head_i}}
        self.model.prune_heads(prune_head)
        self.head_mask = get_pruned_head_mask(self.model)
        self.head_scores.fill(0.0)
        return prune_head


def head_to_prune(head_mask, head_scores):
    """ Determine the best head to prune from a heuristic 'confidence' score.
    
    Based on https://arxiv.org/pdf/1905.09418.pdf where it was found that the
    confidence score corresponded roughly enough with relevance from layer-wise
    relevance propagation.

    Policy:
        Select the head with the lowest score to prune except:
        - Never prune from a layer with only 1 head remaining
        - Never prune from layer 0 (*)

    * The paper did a functionality analysis of heads and found that head(s) in
    the first layer which had lower confidence scores were pointing to rare or
    infrequent words in each sentence. Theirs was a different pretrained model
    entirely (included a decoder for translation), but on the hunch that this
    behavior may track in the encoders of BERT-like models, I prevent pruning
    the heads from the first layer.
    """    

    allowed_layers = []
    for layer_i in range(head_mask.shape[0]):
        if layer_i != 0 and np.sum(head_mask[layer_i]) > 1:
            allowed_layers.append(layer_i)

    smallest = head_scores[1][0]
    head_to_prune = (1, 0)

    for layer_i in allowed_layers:
        for head_i in range(head_scores.shape[1]):
            if head_mask[layer_i][head_i] == 0:
                continue

            if head_scores[layer_i][head_i] < smallest:
                head_to_prune = (layer_i, head_i)
                smallest = head_scores[layer_i][head_i]
    
    return head_to_prune


def accumulate_scores(head_mask, head_scores, batch_attentions):
    """ Add the maximal attention value for each head to its running score
        taking care to align the attention values to the proper head by skipping
        over masked heads, which will not appear in the batch attention outputs
    """
    n_layers, n_heads = head_mask.shape
    batch_size = batch_attentions[0].shape[0]
    for layer_i in range(n_layers):
        head_num = 0
        for true_head_num in range(n_heads):
            if head_mask[layer_i][true_head_num] == 0:
                continue

            for sample_i in range(batch_size):
                score = np.max(batch_attentions[layer_i][sample_i][head_num])
                head_scores[layer_i][true_head_num] += score
            
            head_num += 1


def get_pruned_head_mask(model):
    """ Get a [n_layers x n_heads] mask representing which heads are pruned """

    head_mask = np.ones((
        model.config.num_hidden_layers,
        model.config.num_attention_heads),
        dtype=np.int32
    )

    encoder_layers = pruning_utils.get_layers_for_model(model)

    for layer_num, layer in enumerate(encoder_layers):
        for head_i in layer.attention.pruned_heads:
            head_mask[layer_num][head_i] = 0

    return head_mask