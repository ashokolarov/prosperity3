import json
from abc import ABC, abstractmethod
from typing import Any, TypeAlias

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing["symbol"], listing["product"], listing["denomination"]]
            )

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


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

    def calculate_orders(self, order_depth, position, timestamp):
        moves = []

        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        price = (best_ask + best_bid) / 2
        if price >= int(self._RESIN_MEAN + self._x * self._RESIN_STD):
            ask_price = best_bid
            ask_volume = 50 + position
            bid_order = Order(self.symbol, ask_price, -ask_volume)
            moves.append(bid_order)

        elif price <= int(self._RESIN_MEAN - 0.5 * self._x * self._RESIN_STD):
            bid_price = best_ask
            bid_volume = 50 - position
            ask_order = Order(self.symbol, bid_price, bid_volume)
            moves.append(ask_order)

        else:
            ask_price = int(price) + 3
            ask_volume = 50 - abs(position)
            bid_order = Order(self.symbol, ask_price, -ask_volume)
            moves.append(bid_order)

            bid_price = int(price) - 3
            bid_volume = 50 - abs(position)
            ask_order = Order(self.symbol, bid_price, bid_volume)
            moves.append(ask_order)

        return moves


products = {"RAINFOREST_RESIN": RainforestResin((2, 1))}


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
