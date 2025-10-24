# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.

_DATASET_DICT = {}


def register_dataset(name):
    def decorator(cls):
        _DATASET_DICT[name] = cls
        return cls

    return decorator


def get_dataset(cfg):
    d_cfg = cfg.copy()
    d_type = d_cfg.pop('type')
    train_cfg = d_cfg.pop('train', {})
    test_cfg = d_cfg.pop('test', {})
    train_cfg = {**d_cfg, **train_cfg}
    test_cfg = {**d_cfg, **test_cfg}
    train_set = _DATASET_DICT[d_type](split='train', **train_cfg)
    test_set = _DATASET_DICT[d_type](split='test', **test_cfg)
    return train_set, test_set
