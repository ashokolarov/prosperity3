from datamodel import TradingState
from defined_products import defined_products
from utils import Logger


class Trader:
    def __init__(self):
        self.iter = 0
        self.logger = Logger()

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        self.logger.print(state)

        result = {}
        for product in state.order_depths:
            if product in defined_products.keys():
                order_depth = state.order_depths[product]
                if product in state.position:
                    position = state.position[product]
                else:
                    position = 0

                if product in state.own_trades:
                    own_trades = state.own_trades[product]

                else:
                    own_trades = []
                orders = defined_products[product].calculate_orders(
                    order_depth, position, own_trades, state.timestamp
                )
            else:
                orders = []
            result[product] = orders

        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        self.iter += 1
        self.logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
