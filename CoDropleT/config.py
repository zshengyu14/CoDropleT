# config.py
"""Configuration factory for the solubility model."""

from __future__ import annotations

import copy
from typing import Optional

import ml_collections

_DEFAULT = ml_collections.ConfigDict(
{
                'global_config': {
                    'deterministic': False,
                    'subbatch_size': 1,
                    'use_remat': True,
                    'zero_init': False,
                },
                'model':{
                    'attention_with_pair_bias': {
                        'dropout_rate': 0.15,
                        'gating': True,
                        'num_head': 8,
                        'shared_dropout': True
                    },
                    'single_transition': {
                        'dropout_rate': 0.0,
                        'num_intermediate_factor': 4,
                        'shared_dropout': True
                    },
                    'pair_transition': {
                        'dropout_rate': 0.0,
                        'num_intermediate_factor': 4,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    },
                    'sequence_concatenate': {
                        'num_channels': 64,
                        'num_intermediate_factor': 4,
                        'dropout_rate': 0.0,
                        'shared_dropout': True

                    },
                    'iteration_num_block':8 #8
                    ,
                    'SASA_head': {
                        'num_channels': 512,
                        'weight': 0.0
                    },
                    'protein_encoder': {
                        'single_channels':256,
                        'pair_channels':32,
                        'single_transition': {
                            'dropout_rate': 0.0,
                            'num_intermediate_factor': 4,
                            'shared_dropout': True
                        },
                        'pair_transition': {
                            'dropout_rate': 0.0,
                            'num_intermediate_factor': 4,
                            'orientation': 'per_row',
                            'shared_dropout': True
                        },
                    },
                    'solubility_head': {
                        'num_bins': 50,
                        'num_channels':512,
                        'solubility_weight':1.0,
                        'classification_weight':1.0,
                    }
                },
                'data':{
                    'pair_channels':128,
                    'single_channels':384,
                    'training':{
                        'batch_size':2, #8
                        'crop_size': 1024, #384
                        'epoch_count':1,
                        'learning_rate':5e-6,
                        'beta_1':0.9,
                        'beta_2':0.999,
                        'l2_rate':0.0  
                    }
                }

})

def model_config(crop_size: Optional[int] = None) -> ml_collections.ConfigDict:
    """
    Return a fresh copy of the model configuration.

    Parameters
    ----------
    crop_size
        If given, sets ``cfg.data.training.crop_size`` to this value.
        Leave ``None`` to use the default (1024).

    Notes
    -----
    The config is deep-copied on every call so modifications in user
    code never bleed back to the template.
    """
    cfg = copy.deepcopy(_DEFAULT)
    if crop_size is not None:
        cfg.data.training.crop_size = int(crop_size)
    return cfg