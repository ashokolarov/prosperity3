from abc import ABC, abstractmethod
from typing import List

from datamodel import Order, OrderDepth, TradingState


class Product(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calculate_orders(self, order_depth):
        pass


class RainforestResin(Product):
    def __init__(self, trigger_params):
        self.name = "Rainforest Resin"
        self.symbol = "RAINFOREST_RESIN"
        self._max_position = 50
        self._min_position = -50
        self._RESIN_MEAN = 1e4
        self._RESIN_STD = 1.48
        self._x, self._y = trigger_params

    def calculate_orders(self, order_depth, position=None):
        moves = []

        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        price = (best_ask + best_bid) / 2

        if price >= int(self._RESIN_MEAN + self._x * self._RESIN_STD):
            ask_price = best_bid + self._y
            ask_volume = 1
            bid_order = Order(self.symbol, ask_price, -ask_volume)
            moves.append(bid_order)
        elif price <= int(self._RESIN_MEAN - self._x * self._RESIN_STD):
            bid_price = best_ask - self._y
            bid_volume = 1
            ask_order = Order(self.symbol, bid_price, bid_volume)
            moves.append(ask_order)
        return moves


products = {"RAINFOREST_RESIN": RainforestResin((2, 1))}


class Trader:
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)

        result = {}
        for product in state.order_depths:
            if product in products.keys():
                order_depth = state.order_depths[product]
                orders = products[product].calculate_orders(order_depth)
            else:
                orders = []
            result[product] = orders

        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
