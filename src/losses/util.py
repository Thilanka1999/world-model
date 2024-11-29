from omegaconf import OmegaConf, DictConfig, ListConfig


def count_leaves(conf):
    if isinstance(conf, DictConfig) or isinstance(conf, ListConfig):
        total_leaves = 0
        for key, value in conf.items():
            if isinstance(value, DictConfig) or isinstance(conf, ListConfig):
                total_leaves += count_leaves(value)
            else:
                total_leaves += 1
        return total_leaves
    else:
        return 1


def safe_merge(default_weights, defined_weights):
    """
    Validate that merging dict1 with dict2 will not introduce additional leaves to dict1.
    """
    dict1_conf = OmegaConf.create(default_weights)
    dict2_conf = OmegaConf.create(defined_weights)
    merged_conf = OmegaConf.merge(dict1_conf, dict2_conf)

    if count_leaves(merged_conf) > count_leaves(dict1_conf):
        raise ValueError(
            f"Invalid loss_weights definition. Default weights: {default_weights}"
        )

    return merged_conf
