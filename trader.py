from datamodel import TradingState
from products import RainforestResin
from utils import Logger


products = {"RAINFOREST_RESIN": RainforestResin()}


class Trader:
    def __init__(self):
        self.iter = 0
        self.logger = Logger()

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        self.logger.print(state)

        result = {}
        for product in state.order_depths:
            if product in products.keys():
                order_depth = state.order_depths[product]
                if product in state.position:
                    position = state.position[product]
                else:
                    position = 0
                orders = products[product].calculate_orders(
                    order_depth, position, state.timestamp
                )
            else:
                orders = []
            result[product] = orders

        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        self.iter += 1
        self.logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
