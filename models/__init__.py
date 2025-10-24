# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.

from ._base import get_model, register_model
from .fno import FNO
from .functional import FFM

def get_flow_model(model_cfg, encoder_cfg, conditional=False):
    """
    Build the functional flow model.
    :param model_cfg: model configs passed to the flow model, type indicates the model type
    :param encoder_cfg: encoder configs passed to the encoder model
    :param conditional: whether the model is conditional
    :return: the flow model
    """
    model_factory = FFM 
    return model_factory(get_model(encoder_cfg), **model_cfg)
