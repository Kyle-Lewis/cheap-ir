
from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaLayer
)

from transformers.models.distilbert.modeling_distilbert import (
    DistilBertModel,
    TransformerBlock
)


def get_layers_for_model(model):
    
    architecture = model.config.architectures[0]

    if isinstance(model.base_model, RobertaModel):
        encoder_layers = model.base_model.encoder.layer
    
    elif isinstance(model.base_model, DistilBertModel):
        encoder_layers = model.base_model.transformer.layer

    else:
        raise ValueError("Not supporting model type:", architecture)
    
    return encoder_layers


def get_linears_for_layer(layer):

    if isinstance(layer, RobertaLayer):
        lin1 = layer.intermediate.dense
        lin2 = layer.output.dense
    
    elif isinstance(layer, TransformerBlock):
        lin1 = layer.ffn.lin1
        lin2 = layer.ffn.lin2
    
    else:
        raise ValueError(
            "prune_intermediate_neurons not supported for layer type:", layer)

    return lin1, lin2