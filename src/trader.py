from time import time

import jsonpickle

from datamodel import TradingState
from products import (
    Croissants,
    Djembes,
    Jams,
    Kelp,
    PicnicBasket1,
    PicnicBasket2,
    RainforestResin,
    Squid,
    SyntheticBasket1,
)
from utils import CustomLogger

config_rainforest = {
    # Market taking parameters
    "mt_bid_edge": 1,
    "mt_ask_edge": 1,
    "mt_long_pm": 0,
    "mt_short_pm": 0,
    # Market making parameters
    "mm_default_vol": 15,
    "mm_default_edge": 4,
    "mm_disregard_edge": 1,
    "mm_join_edge": 2,
    "mm_join_volume": 3,
    "mm_join_edge_2": 4,
    "mm_join_volume_2": 1,
}

config_kelp = {
    # General
    "detect_mm_volume": 15,  # Volume to detect market maker
    # Market taking parameters
    "mt_take_width": 1,
    "mt_clear_width": 0,
    "mt_adverse_volume": 15,  # Maximum mt volume
    # Market making parameters
    "mm_default_vol": 20,
    "mm_default_edge": 1,
    "mm_disregard_edge": 1,
    "mm_join_edge": 0,
    "mm_join_volume": 3,
}

config_squid = {
    # General
    "detect_mm_volume": 15,  # Volume to detect market maker
    # Price estimation
    "short_window": 80,
    "long_window": 410,
    # Directional parameters
    "dt_default_vol": 5,
    "dt_signal_strength": 0.0015,
}

config_croissants = {}

config_jams = {}

config_djembes = {}

config_picnic_basket_1 = {
    "detect_mm_volume": 15,  # Volume to detect market maker
    # Market making parameters
    "mm_default_vol": 20,
    "mm_default_edge": 4,
    "mm_disregard_edge": 2,
    "mm_join_edge": 6,
    "mm_join_volume": 5,
}

config_picnic_basket_2 = {
    "detect_mm_volume": 15,  # Volume to detect market maker
    # Market making parameters
    "mm_default_vol": 20,
    "mm_default_edge": 4,
    "mm_disregard_edge": 2,
    "mm_join_edge": 6,
    "mm_join_volume": 5,
}

config_synthetic_basket_1 = {
    "buy_entry": 1.5,
    "buy_exit": 0.5,
    "sell_entry": 2.0,
    "sell_exit": 0.5,
}


class Trader:
    def __init__(self):
        self.logger = CustomLogger()

    def run(self, state: TradingState):
        t1 = time()

        self.logger.print("TRADER_B")
        timestamp = state.timestamp
        self.logger.print(f"timestamp {timestamp}")

        result = {}
        if not state.traderData:
            products = {}
            # products["RAINFOREST_RESIN"] = RainforestResin(config_rainforest)
            # products["KELP"] = Kelp(config_kelp)
            # products["SQUID_INK"] = Squid(config_squid)
            products["CROISSANTS"] = Croissants(config_croissants)
            products["JAMS"] = Jams(config_jams)
            products["DJEMBES"] = Djembes(config_djembes)
            products["PICNIC_BASKET1"] = PicnicBasket1(config_picnic_basket_1)
            # products["PICNIC_BASKET2"] = PicnicBasket2(config_picnic_basket_2)
            synthetic_products = {}
            synthetic_products["SYNTHETIC_BASKET1"] = SyntheticBasket1(
                config_synthetic_basket_1
            )
            # products["SYNTHETIC_BASKET2"] = SyntheticBasket2(config_synthetic_basket_2)
        else:
            traderData = jsonpickle.decode(state.traderData)
            products = traderData["products"]
            synthetic_products = traderData["synthetic_products"]

        for product in state.order_depths:
            if product in products.keys():
                order_depth = state.order_depths[product]
                if product in state.position:
                    position = state.position[product]
                else:
                    position = 0

                if product in state.own_trades:
                    own_trades = state.own_trades[product]
                else:
                    own_trades = []

                products[product].update_product(
                    order_depth, position, own_trades, timestamp
                )
                products[product].calculate_orders()

        for product in synthetic_products.keys():
            synthetic_products[product].calculate_orders(products, timestamp)

        for product in state.order_depths:
            if product in products.keys():
                result[product] = products[product].on_timestep_end()

        traderData = dict()
        traderData["products"] = products
        traderData["synthetic_products"] = synthetic_products
        traderData = jsonpickle.encode(traderData)

        conversions = 1

        t2 = time()

        self.logger.print_numeric("runtime", t2 - t1)
        self.logger.print("TRADER_E")
        self.logger.flush()

        return result, conversions, traderData
