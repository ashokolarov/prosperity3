from time import time

import jsonpickle

from datamodel import TradingState
from products import Kelp, RainforestResin
from utils import CustomLogger

config_rainforest = {
    # Market taking parameters
    "mt_bid_edge": 1,
    "mt_ask_edge": 1,
    "mt_short_pm": 0,
    "mt_long_pm": 0,
    # Market making parameters
    "mm_default_vol": 15,
    "skew_factor": 0.1,  # How much to skew quotes based on position
    "max_position_factor": 0.7,  # Maximum position as a factor of position limit
    "spread_widener": 0.2,  # How much to widen spread as position grows
    "min_spread": 2,  # Minimum spread regardless of position
}

config_kelp = {}


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
            products["RAINFOREST_RESIN"] = RainforestResin(config_rainforest)
            products["KELP"] = Kelp(config_kelp)
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
