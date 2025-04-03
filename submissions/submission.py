# Combined Python Files
# Files combined: market_utils.py, products.py, trader.py, utils.py

# Import statements
from abc import ABC, abstractmethod

from collections import deque

from copy import deepcopy

from datamodel import Order

from datamodel import TradingState

from time import time

from typing import Any

import jsonpickle

import numpy as np



# Code from market_utils.py
class OrderBook:
    def __init__(self):
        self.ask_prices = []
        self.ask_volumes = []
        self.bid_prices = []
        self.bid_volumes = []

        self.previous_ask_prices = []
        self.previous_ask_volumes = []
        self.previous_bid_prices = []
        self.previous_bid_volumes = []

    def reset(self, order_depths):
        sell_orders = order_depths.sell_orders
        buy_orders = order_depths.buy_orders

        # Save previous state
        self.previous_ask_prices = deepcopy(self.ask_prices)
        self.previous_ask_volumes = deepcopy(self.ask_volumes)
        self.previous_bid_prices = deepcopy(self.bid_prices)
        self.previous_bid_volumes = deepcopy(self.bid_volumes)

        # Reset order book
        sell_orders = list(sell_orders.items())
        self.ask_prices = [order[0] for order in sell_orders]
        self.ask_volumes = [abs(order[1]) for order in sell_orders]

        buy_orders = list(buy_orders.items())
        self.bid_prices = [order[0] for order in buy_orders]
        self.bid_volumes = [order[1] for order in buy_orders]

    def get_ask_order_at_depth(self, depth):
        assert depth < self.ask_orders_depth and depth >= 0

        if len(self.ask_prices) == 0:
            return None
        else:
            return self.ask_prices[depth], self.ask_volumes[depth]

    def get_bid_order_at_depth(self, depth):
        assert depth < self.bid_orders_depth and depth >= 0

        if len(self.bid_prices) == 0:
            return None
        else:
            return self.bid_prices[depth], self.bid_volumes[depth]

    @property
    def bid_orders_depth(self):
        return len(self.bid_prices)

    @property
    def ask_orders_depth(self):
        return len(self.ask_prices)

    @property
    def spread(self):
        if len(self.bid_prices) == 0 or len(self.ask_prices) == 0:
            return None
        else:
            return self.ask_prices[0] - self.bid_prices[0]

    @property
    def mid_price(self):
        if len(self.bid_prices) == 0 or len(self.ask_prices) == 0:
            return None
        else:
            return (self.ask_prices[0] + self.bid_prices[0]) / 2

    @property
    def vwap(self):
        if len(self.bid_prices) == 0 or len(self.ask_prices) == 0:
            return None
        else:
            bid_vwap = sum(
                [
                    price * volume
                    for price, volume in zip(self.bid_prices, self.bid_volumes)
                ]
            ) / sum(self.bid_volumes)
            ask_vwap = sum(
                [
                    price * volume
                    for price, volume in zip(self.ask_prices, self.ask_volumes)
                ]
            ) / sum(self.ask_volumes)

            vwap = (bid_vwap + ask_vwap) / 2

            return vwap

    @property
    def mm_spread(self):
        if len(self.bid_prices) == 0 or len(self.ask_prices) == 0:
            return None
        else:
            max_ask_volume_index = self.ask_volumes.index(max(self.ask_volumes))
            max_bid_volume_index = self.bid_volumes.index(max(self.bid_volumes))
            ask_price_at_max_volume = self.ask_prices[max_ask_volume_index]
            bid_price_at_max_volume = self.bid_prices[max_bid_volume_index]
            return ask_price_at_max_volume - bid_price_at_max_volume

    @property
    def mm_fair_price(self):
        if len(self.bid_prices) == 0 or len(self.ask_prices) == 0:
            return None
        else:
            if max(self.ask_volumes) > 10 and max(self.bid_volumes) > 10:
                max_ask_volume_index = self.ask_volumes.index(max(self.ask_volumes))
                max_bid_volume_index = self.bid_volumes.index(max(self.bid_volumes))
                ask_price_at_max_volume = self.ask_prices[max_ask_volume_index]
                bid_price_at_max_volume = self.bid_prices[max_bid_volume_index]
                fair_price = (ask_price_at_max_volume + bid_price_at_max_volume) / 2
                return fair_price
            else:
                return None

    @property
    def imbalance(self):
        """Calculate the order book imbalance ratio."""
        total_bid_volume = sum(self.bid_volumes)
        total_ask_volume = sum(self.ask_volumes)

        # Avoid division by zero
        if total_ask_volume == 0:
            return float("inf")  # Extreme buying pressure
        elif total_bid_volume == 0:
            return 0.0  # Extreme selling pressure

        return total_bid_volume / total_ask_volume

    def calculate_ofi(self):
        if len(self.previous_ask_prices) == 0 or len(self.previous_bid_prices) == 0:
            return 0
        else:
            total_bid_volume = sum(self.bid_volumes)
            total_ask_volume = sum(self.ask_volumes)
            total_bid_volume_prev = sum(self.previous_bid_volumes)
            total_ask_volume_prev = sum(self.previous_ask_volumes)

            delta_bid = total_bid_volume - total_bid_volume_prev
            delta_ask = total_ask_volume - total_ask_volume_prev

            return delta_bid - delta_ask

    def __repr__(self):
        repr_str = "BID ORDER PRICE | VOLUME | ASK ORDER PRICE\n"
        length = max(self.ask_prices) - min(self.bid_prices)
        lines = [repr_str]
        for i in range(length + 1):
            price_level = max(self.ask_prices) - i
            if (
                price_level not in self.ask_prices
                and price_level not in self.bid_prices
            ):
                continue
            bid_line = (
                "               "
                if price_level not in self.bid_prices
                else f"    {price_level}      "
            )
            bid_line += (
                " "
                if len(str(price_level)) == 4 and price_level in self.bid_prices
                else ""
            )
            ask_line = (
                "" if price_level not in self.ask_prices else f"     {price_level}"
            )
            volume = (
                self.ask_volumes[self.ask_prices.index(price_level)]
                if price_level in self.ask_prices
                else self.bid_volumes[self.bid_prices.index(price_level)]
            )
            volume_line = f"  {volume}  "
            volume_line += " " if len(str(volume)) == 1 else ""

            lines.append(f"{bid_line} | {volume_line} | {ask_line}\n")

        spread = self.spread
        mid_price = self.mid_price
        vwap = self.vwap
        imbalance = self.imbalance
        lines.append(f"Spread: {spread}\n")
        lines.append(f"Mid Price: {mid_price}\n")
        lines.append(f"VWAP: {vwap:.1f}\n")
        lines.append(f"Order Book Imbalance: {imbalance:.2f}\n")

        return "".join(lines)

    def update(self, order):
        if order.quantity > 0:  # Buy order
            if order.price in self.ask_prices:
                index = self.ask_prices.index(order.price)
                if self.ask_volumes[index] > order.quantity:
                    self.ask_volumes[index] -= order.quantity
                elif self.ask_volumes[index] < order.quantity:
                    self.ask_prices.pop(index)
                    self.ask_volumes.pop(index)
                    self.bid_volumes.append(order.quantity - self.ask_volumes[index])
                    self.bid_prices.append(order.price)
                else:
                    self.ask_prices.pop(index)
                    self.ask_volumes.pop(index)
            else:
                if order.price in self.bid_prices:
                    index = self.bid_prices.index(order.price)
                    self.bid_volumes[index] += order.quantity
                else:
                    self.bid_prices.append(order.price)
                    self.bid_volumes.append(order.quantity)
        else:  # Sell order
            if order.price in self.bid_prices:
                index = self.bid_prices.index(order.price)
                if self.bid_volumes[index] > abs(order.quantity):
                    self.bid_volumes[index] -= abs(order.quantity)
                elif self.bid_volumes[index] < abs(order.quantity):
                    self.bid_prices.pop(index)
                    self.bid_volumes.pop(index)
                    self.ask_volumes.append(
                        abs(order.quantity) - self.bid_volumes[index]
                    )
                    self.ask_prices.append(order.price)
                else:
                    self.bid_prices.pop(index)
                    self.bid_volumes.pop(index)
            else:
                if order.price in self.ask_prices:
                    index = self.ask_prices.index(order.price)
                    self.ask_volumes[index] += abs(order.quantity)
                else:
                    self.ask_prices.append(order.price)
                    self.ask_volumes.append(abs(order.quantity))

        # Sort the order book by price
        self.sell_orders = sorted(
            zip(self.ask_prices, self.ask_volumes), key=lambda x: x[0]
        )
        self.buy_orders = sorted(
            zip(self.bid_prices, self.bid_volumes), key=lambda x: x[0], reverse=True
        )

        self.ask_prices = [order[0] for order in self.sell_orders]
        self.ask_volumes = [order[1] for order in self.sell_orders]
        self.bid_prices = [order[0] for order in self.buy_orders]
        self.bid_volumes = [order[1] for order in self.buy_orders]

# Code from products.py
class Product(ABC):
    name: str = None
    symbol: str = None
    pos_limit: int = None

    def __init__(self, config):
        self.logger = CustomLogger()

    def print_product_begin(self, timestamp):
        self.logger.print(f"PRODUCT_B {self.symbol}")
        self.logger.print_numeric("timestamp", timestamp)

    def print_product_end(self):
        self.logger.print(f"PRODUCT_E {self.symbol}")
        self.logger.flush()

    def print_orders(self, orders):
        for order in orders:
            self.logger.print(f"order {order.quantity}@{order.price}")

    def calculate_position_delta(self, orders):
        delta = sum(order.quantity for order in orders)
        positive_delta = sum(order.quantity for order in orders if order.quantity > 0)
        negative_delta = sum(order.quantity for order in orders if order.quantity < 0)

        return delta, positive_delta, negative_delta

    @abstractmethod
    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        pass


# ------------------RAINFOREST_RESIN-------------------#
class RainforestResin(Product):
    def __init__(self, config):
        super().__init__(config)

        # Rainforest Resin parameters
        self.name = "Rainforest Resin"
        self.symbol = "RAINFOREST_RESIN"
        self.pos_limit = 50
        self.mean = 1e4

        # Order book
        self.order_book = OrderBook()
        self.update_order_book = config.get("update_order_book")

        # Price tracking
        self.history_size = config.get("history_size")
        self.fair_price_type = config.get("fair_price")
        self.price_history = deque(maxlen=self.history_size)

        # Market taking parameters
        self.mt_position = 0
        self.mt_bid_edge = config.get("mt_bid_edge")
        self.mt_ask_edge = config.get("mt_ask_edge")
        self.mt_long_profit_margin = config.get("mt_long_pm")
        self.mt_short_profit_margin = config.get("mt_short_pm")

        # Market making parameters
        self.mm_default_vol = config.get("mm_default_vol")
        self.mm_ofi_sensitivity = config.get("mm_ofi_sensitivity")

    def get_fair_price(self):
        if self.fair_price_type == "vwap":
            fair_price = self.order_book.vwap
        elif self.fair_price_type == "mid_price":
            fair_price = self.order_book.mid_price
        else:
            raise ValueError("Invalid fair price type")

        return fair_price

    def calculate_volatility(self):
        if len(self.price_history) == self.history_size:
            prices_array = np.array(self.price_history, dtype=np.float64)
            return np.std(prices_array)
        return 1.0

    def market_take(self, remaining_buy, remaining_sell):
        orders = []

        # Check if there is an opportunity to market take in ask orders
        for depth_level in range(self.order_book.ask_orders_depth):
            ask_price, ask_volume = self.order_book.get_ask_order_at_depth(depth_level)
            if self.mean - ask_price >= self.mt_ask_edge:
                bid_price = ask_price
                bid_volume = min(remaining_buy, ask_volume)
                bid_order = Order(self.symbol, bid_price, bid_volume)
                orders.append(bid_order)
                # update positions and sell/buy volumes
                remaining_buy -= bid_volume
                self.mt_position += bid_volume
            else:
                break  # If even the best ask doesn't cross the mean, then no need to check further

        # Check if there is an opportunity to market take in bid orders
        for depth_level in range(self.order_book.bid_orders_depth):
            bid_price, bid_volume = self.order_book.get_bid_order_at_depth(depth_level)
            if bid_price - self.mean >= self.mt_bid_edge:
                ask_price = bid_price
                ask_volume = min(remaining_sell, bid_volume)
                ask_order = Order(self.symbol, ask_price, -ask_volume)
                orders.append(ask_order)
                # update positions and sell/buy volumes
                remaining_sell -= ask_volume
                self.mt_position -= ask_volume
            else:
                break  # If even the best bid doesn't cross the mean, then no need to check further

        if self.update_order_book:
            for order in orders:
                self.order_book.update(order)

        return orders, remaining_buy, remaining_sell

    def liquidate_mt_orders(self, position, remaining_buy, remaining_sell):
        orders = []

        # Check if there is an opportunity to liquidate long positions
        for depth_level in range(self.order_book.bid_orders_depth):
            if position > 0:
                bid_price, bid_volume = self.order_book.get_bid_order_at_depth(
                    depth_level
                )
                if bid_price - self.mean >= self.mt_long_profit_margin:
                    qty = min(remaining_sell, bid_volume, position)
                    ask_order = Order(self.symbol, bid_price, -qty)
                    orders.append(ask_order)
                    # update positions and remaining buy/sell volumes
                    remaining_sell -= qty
                    position -= qty
                    self.mt_position -= qty
                else:
                    break
            else:
                break

        # Check if there is an opportunity to liquidate short positions
        for depth_level in range(self.order_book.ask_orders_depth):
            if position < 0:
                ask_price, ask_volume = self.order_book.get_ask_order_at_depth(
                    depth_level
                )
                if self.mean - ask_price >= self.mt_short_profit_margin:
                    qty = min(remaining_buy, ask_volume, abs(position))
                    bid_order = Order(self.symbol, ask_price, qty)
                    orders.append(bid_order)
                    # update positions and remaining buy/sell volumes
                    remaining_buy -= qty
                    position += qty
                    self.mt_position += qty
                else:
                    break
            else:
                break

        if self.update_order_book:
            for order in orders:
                self.order_book.update(order)

        return orders, remaining_buy, remaining_sell

    def market_make(self, positions):
        orders = []

        # Get current market state
        spread = self.order_book.spread
        self.logger.print_numeric("spread", spread)
        mid_price = round(self.order_book.mid_price)

        position = positions["position"]
        position_skew = position / 20

        imbalance = self.order_book.imbalance
        imbalance_skew = imbalance / 40

        # Calculate our bid and ask prices
        half_spread = spread // 2
        bid_price = mid_price - half_spread + 1 - position_skew + imbalance_skew
        ask_price = mid_price + half_spread - 1 - position_skew - imbalance_skew

        bid_price = round(bid_price)
        ask_price = round(ask_price)

        if bid_price > self.mean:
            bid_price = round(self.mean)
        if ask_price < self.mean:
            ask_price = round(self.mean)

        # Scale our order sizes based on how far we are from position limits
        bid_volume = min(self.mm_default_vol, positions["remaining_buy"])
        ask_volume = min(self.mm_default_vol, positions["remaining_sell"])

        # Create the orders if they make sense
        if bid_volume > 0:
            bid_order = Order(self.symbol, bid_price, bid_volume)
            orders.append(bid_order)

        if ask_volume > 0:
            ask_order = Order(self.symbol, ask_price, -ask_volume)
            orders.append(ask_order)

        if self.update_order_book:
            for order in orders:
                self.order_book.update(order)

        return orders

    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)

        # Reset order book, track positions and prices
        self.order_book.reset(order_depths)
        orders = []

        mt_position = self.mt_position
        mm_position = position - mt_position
        self.logger.print_numeric("position", position)
        self.logger.print_numeric("mt_position", mt_position)
        self.logger.print_numeric("mm_position", mm_position)

        remaining_buy = self.pos_limit - position
        remaining_sell = self.pos_limit + position

        mid_price = self.order_book.mid_price
        vwap = self.order_book.vwap
        self.logger.print_numeric("mid_price", mid_price)
        self.logger.print_numeric("vwap", vwap)

        fair_price = self.get_fair_price()
        self.price_history.append(fair_price)
        self.logger.print_numeric("fair_price", fair_price)

        volatility = self.calculate_volatility()
        self.logger.print_numeric("volatility", volatility)

        # ------------------------------------------------
        # Liquidation and market taking
        liquidated_orders, remaining_buy, remaining_sell = self.liquidate_mt_orders(
            position, remaining_buy, remaining_sell
        )
        orders += liquidated_orders

        mt_orders, remaining_buy, remaining_sell = self.market_take(
            remaining_buy, remaining_sell
        )
        orders += mt_orders
        # ------------------------------------------------
        # Market making
        positions = {
            "remaining_buy": remaining_buy,
            "remaining_sell": remaining_sell,
            "position": mm_position,
        }

        mm_orders = self.market_make(positions)
        orders += mm_orders
        # ------------------------------------------------
        self.print_orders(orders)
        self.print_product_end()

        return orders


# ------------------KELP-------------------#
class Kelp(Product):
    def __init__(self, config):
        super().__init__(config)

        # Kelp parameters
        self.name = "Kelp"
        self.symbol = "KELP"
        self.pos_limit = 50

        # Order book
        self.order_book = OrderBook()

        # Direction trading parameters
        self.d_default_vol = config.get("d_default_vol")
        self.d_short_window = config.get("d_short_window")
        self.d_long_window = config.get("d_long_window")
        self.d_short_history = deque(maxlen=self.d_short_window)
        self.d_long_history = deque(maxlen=self.d_long_window)

    def check_directional(self):
        short_history_full = len(self.d_short_history) == self.d_short_window
        long_history_full = len(self.d_long_history) == self.d_long_window

        return short_history_full and long_history_full

    def directional(self, position):
        orders = []
        if self.check_directional():
            # Calculate moving averages
            short_ma = sum(self.d_short_history) / self.d_short_window
            long_ma = sum(self.d_long_history) / self.d_long_window

            if short_ma > long_ma:  # Bullish signal
                remaining_buy = self.pos_limit - position
                total_buy_volume = min(self.d_default_vol, remaining_buy)

                bought_volume = 0
                for depth_level in range(self.order_book.ask_orders_depth):
                    if bought_volume < total_buy_volume:
                        ask_price, ask_volume = self.order_book.get_bid_order_at_depth(
                            depth_level
                        )
                        ask_volume = min(total_buy_volume - bought_volume, ask_volume)
                        bid_order = Order(self.symbol, ask_price, ask_volume)
                        orders.append(bid_order)

                        bought_volume += ask_volume

            elif short_ma < long_ma:  # Bearish signal
                remaining_sell = self.pos_limit + position
                total_sell_volume = min(self.d_default_vol, remaining_sell)

                sold_volume = 0
                for depth in range(self.order_book.bid_orders_depth):
                    if sold_volume < total_sell_volume:
                        bid_price, bid_volume = self.order_book.get_ask_order_at_depth(
                            depth
                        )
                        bid_volume = min(total_sell_volume - sold_volume, bid_volume)
                        ask_order = Order(self.symbol, bid_price, -bid_volume)
                        orders.append(ask_order)

                        sold_volume += bid_volume

        return orders

    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.order_book.reset(order_depths)

        orders = []
        self.logger.print(f"position {position}")

        vwap = self.order_book.vwap
        self.logger.print(f"vwap {vwap}")
        mm_fair = self.order_book.mm_fair_price
        self.logger.print(f"mm_fair {mm_fair}")

        if mm_fair is not None:
            self.d_short_history.append(mm_fair)
            self.d_long_history.append(mm_fair)

        orders = self.directional(position)

        self.print_orders(orders)
        self.print_product_end()

        return orders

# Code from trader.py
config_rainforest = {
    "update_order_book": True,
    # Market taking parameters
    "mt_bid_edge": 1,
    "mt_ask_edge": 1,
    "mt_long_pm": 0,
    "mt_short_pm": 0,
    # Market making parameters
    "history_size": 25,
    "fair_price": "vwap",
    "mm_default_vol": 15,
    "mm_ofi_sensitivity": 0.035,
}

config_kelp = {
    # Market taking parameters
    "d_default_vol": 15,
    "d_short_window": 5,
    "d_long_window": 40,
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
            products["RAINFOREST_RESIN"] = RainforestResin(config_rainforest)
            # products["KELP"] = Kelp(config_kelp)
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

# Code from utils.py
class CustomLogger:
    def __init__(self) -> None:
        self.logs = ""
        self.end = "\n"
        self.sep = " "

    def print(self, *objects: Any) -> None:
        self.logs += self.sep.join(map(str, objects)) + self.end

    def print_numeric(self, label, value, end="\n") -> None:
        """Print a labeled numeric value with consistent formatting."""
        if isinstance(value, float):
            self.logs += f"{label} {value:.2f}"
        else:
            self.logs += f"{label} {value}"

        self.logs += end

    def flush(self):
        print(self.logs)
        self.logs = ""