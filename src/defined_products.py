from products import RainforestResin

_config_rainforest = {
    "mean": 1e4,
    "std": 1.48,
    "mt_hard_limit": 0.8,
    "hard_liquidate_target_percentage": 0.05,
    "mm_order_volume": 15,
}

defined_products = {"RAINFOREST_RESIN": RainforestResin(_config_rainforest)}
