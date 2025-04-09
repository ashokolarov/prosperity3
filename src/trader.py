from time import time

import jsonpickle

from datamodel import TradingState
from products import Kelp, RainforestResin, Squid
from utils import CustomLogger

config_rainforest = {
    "update_order_book": True,
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
    "update_order_book": True,
    "detect_mm_volume": 15,  # Volume to detect market maker
    # Market taking parameters
    "mt_take_width": 1,
    "mt_clear_width": 0,
    "mt_adverse_volume": 15,  # Maximum mt volume
    "mt_reversion_beta": -0.23,
    # Market making parameters
    "mm_default_vol": 20,
    "mm_disregard_edge": 2,
}

config_squid = {
    "update_order_book": True,
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
            products["KELP"] = Kelp(config_kelp)
            # products["SQUID_INK"] = Squid(config_squid)
        else:
            traderData = jsonpickle.decode(state.traderData)
            products = traderData["products"]

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

                orders = products[product].calculate_orders(
                    order_depth, position, own_trades, timestamp
                )
            else:
                orders = []

            result[product] = orders

        traderData = dict()
        traderData["products"] = products
        traderData = jsonpickle.encode(traderData)

        conversions = 1

        t2 = time()

        self.logger.print(f"runtime {t2 - t1}")
        self.logger.print("TRADER_E")
        self.logger.flush()

        return result, conversions, traderData
