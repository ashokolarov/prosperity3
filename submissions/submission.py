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

    def reset_to_previous(self):
        self.ask_prices = deepcopy(self.previous_ask_prices)
        self.ask_volumes = deepcopy(self.previous_ask_volumes)
        self.bid_prices = deepcopy(self.previous_bid_prices)
        self.bid_volumes = deepcopy(self.previous_bid_volumes)

    def check_if_no_orders(self):
        if len(self.bid_prices) == 0 and len(self.ask_prices) == 0:
            return True
        else:
            return False

    def get_best_bid(self):
        if len(self.bid_prices) == 0:
            return None, None
        else:
            return self.bid_prices[0], self.bid_volumes[0]

    def get_best_ask(self):
        if len(self.ask_prices) == 0:
            return None
        else:
            return self.ask_prices[0], self.ask_volumes[0]

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

    def get_bid_prices(self):
        return deepcopy(self.bid_prices)

    def get_ask_prices(self):
        return deepcopy(self.ask_prices)

    def get_bid_volumes(self):
        return deepcopy(self.bid_volumes)

    def get_ask_volumes(self):
        return deepcopy(self.ask_volumes)

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

    def get_mm_fair(self, adverse_volume, with_spread=False):
        if (
            self.ask_orders_depth == 0
            or self.bid_orders_depth == 0
            or max(self.ask_volumes) < adverse_volume
            or max(self.bid_volumes) < adverse_volume
        ):
            return None
        else:
            filtered_ask = [
                self.ask_prices[idx]
                for idx in range(self.ask_orders_depth)
                if self.ask_volumes[idx] >= adverse_volume
            ]

            filtered_bid = [
                self.bid_prices[idx]
                for idx in range(self.bid_orders_depth)
                if self.bid_volumes[idx] >= adverse_volume
            ]

            mm_ask = max(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = min(filtered_bid) if len(filtered_bid) > 0 else None

            if mm_ask is None or mm_bid is None:
                return None

            fair_price = (mm_ask + mm_bid) / 2
            spread = mm_ask - mm_bid

            if with_spread:
                return fair_price, spread
            else:
                return fair_price

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

    def update(self, order):
        if order.quantity > 0:  # Buy order
            if order.price in self.ask_prices:
                index = self.ask_prices.index(order.price)
                if self.ask_volumes[index] > order.quantity:
                    self.ask_volumes[index] -= order.quantity
                elif self.ask_volumes[index] < order.quantity:
                    volumes = deepcopy(self.ask_volumes)
                    self.bid_volumes.append(order.quantity - volumes[index])
                    self.bid_prices.append(order.price)
                    self.ask_prices.pop(index)
                    self.ask_volumes.pop(index)
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
                    volumes = deepcopy(self.bid_volumes)
                    self.ask_volumes.append(abs(order.quantity) - volumes[index])
                    self.ask_prices.append(order.price)
                    self.bid_prices.pop(index)
                    self.bid_volumes.pop(index)
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

# Code from products.py
class Product(ABC):
    name: str = None
    symbol: str = None
    pos_limit: int = None
    order_book: OrderBook = None
    orders: list = None
    position: int = None
    remaining_buy: int = None
    remaining_sell: int = None
    timestamp: int = None

    def __init__(self, config):
        self.logger = CustomLogger()
        self.order_book = OrderBook()

    def print_product_begin(self, timestamp):
        self.logger.print(f"PRODUCT_B {self.symbol}")
        self.logger.print_numeric("timestamp", timestamp)

    def print_product_end(self):
        self.logger.print(f"PRODUCT_E {self.symbol}")
        self.logger.flush()

    def print_orders(self, orders):
        for order in orders:
            self.logger.print(f"order {order.quantity}@{order.price}")

    def place_order(self, price, quantity, type="MARKET", update_order_book=True):
        order = Order(self.symbol, price, quantity)
        self.orders.append(order)

        if update_order_book:
            self.order_book.update(order)

        # Updated position and remaining buy/sell volumes
        if type == "MARKET":
            if quantity > 0:  # Buy order
                self.remaining_buy -= quantity
                self.position += quantity
            elif quantity < 0:  # Sell order (negative quantity)
                self.remaining_sell += quantity
                self.position -= quantity

    def on_timestep_end(self):
        self.print_orders(self.orders)
        self.print_product_end()

        return self.orders

    @abstractmethod
    def update_product(self, order_depths, position, own_trades, timestamp):
        pass

    @abstractmethod
    def calculate_orders():
        pass


# ------------------RAINFOREST_RESIN-------------------#
class RainforestResin(Product):
    def __init__(self, config):
        super().__init__(config)

        # Rainforest Resin parameters
        self.name = "Rainforest Resin"
        self.symbol = "RAINFOREST_RESIN"
        self.pos_limit = 50

        # Price estimation
        self.fair_value = 10000

        # Market taking parameters
        self.mt_bid_edge = config.get("mt_bid_edge")
        self.mt_ask_edge = config.get("mt_ask_edge")
        self.mt_long_profit_margin = config.get("mt_long_pm")
        self.mt_short_profit_margin = config.get("mt_short_pm")

        # Market making parameters
        self.mm_default_vol = config.get("mm_default_vol")
        self.mm_default_edge = config.get("mm_default_edge")
        self.mm_disregard_edge = config.get("mm_disregard_edge")
        self.mm_join_edge = config.get("mm_join_edge")
        self.mm_join_volume = config.get("mm_join_volume")
        self.mm_join_edge_2 = config.get("mm_join_edge_2")
        self.mm_join_volume_2 = config.get("mm_join_volume_2")

    def update_product(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.logger.print_numeric("position", position)

        # Set timestamp
        self.timestamp = timestamp

        # Reset order book
        self.order_book.reset(order_depths)

        # Reset orders
        self.orders = []

        # Update position
        self.position = position
        self.remaining_buy = self.pos_limit - position
        self.remaining_sell = self.pos_limit + position

    def market_take(self):
        ask_prices = self.order_book.get_ask_prices()
        ask_volumes = self.order_book.get_ask_volumes()
        ask_orders_depth = self.order_book.ask_orders_depth
        # Check if there is an opportunity to market take in ask orders
        for depth_level in range(ask_orders_depth):
            ask_price = ask_prices[depth_level]
            ask_volume = ask_volumes[depth_level]
            if self.fair_value - ask_price >= self.mt_ask_edge:
                bid_price = ask_price
                bid_volume = min(self.remaining_buy, ask_volume)
                self.place_order(bid_price, bid_volume)
            else:
                break  # If even the best ask doesn't cross the mean, then no need to check further

        bid_prices = self.order_book.get_bid_prices()
        bid_volumes = self.order_book.get_bid_volumes()
        bid_orders_depth = self.order_book.bid_orders_depth
        # Check if there is an opportunity to market take in bid orders
        for depth_level in range(bid_orders_depth):
            bid_price = bid_prices[depth_level]
            bid_volume = bid_volumes[depth_level]
            if bid_price - self.fair_value >= self.mt_bid_edge:
                ask_price = bid_price
                ask_volume = min(self.remaining_sell, bid_volume)
                self.place_order(ask_price, -ask_volume)
            else:
                break  # If even the best bid doesn't cross the mean, then no need to check further

    def liquidate_position(self):
        bid_prices = self.order_book.get_bid_prices()
        bid_volumes = self.order_book.get_bid_volumes()
        bid_orders_depth = self.order_book.bid_orders_depth
        # Check if there is an opportunity to liquidate long positions
        for depth_level in range(bid_orders_depth):
            if self.position > 0:
                bid_price = bid_prices[depth_level]
                bid_volume = bid_volumes[depth_level]
                if bid_price - self.fair_value >= self.mt_long_profit_margin:
                    qty = min(self.remaining_sell, bid_volume, self.position)
                    self.place_order(bid_price, -qty)
                else:
                    break
            else:
                break

        ask_prices = self.order_book.get_ask_prices()
        ask_volumes = self.order_book.get_ask_volumes()
        ask_orders_depth = self.order_book.ask_orders_depth
        # Check if there is an opportunity to liquidate short positions
        for depth_level in range(ask_orders_depth):
            if self.position < 0:
                ask_price = ask_prices[depth_level]
                ask_volume = ask_volumes[depth_level]
                if self.fair_value - ask_price >= self.mt_short_profit_margin:
                    qty = min(self.remaining_buy, ask_volume, abs(self.position))
                    self.place_order(ask_price, qty)
                else:
                    break
            else:
                break

    def market_make(self):
        asks_above_fair = [
            price
            for price in self.order_book.ask_prices
            if price > self.fair_value + self.mm_disregard_edge
        ]
        bids_below_fair = [
            price
            for price in self.order_book.bid_prices
            if price < self.fair_value - self.mm_disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask_price = round(self.fair_value + self.mm_default_edge)
        if best_ask_above_fair is not None:
            best_ask_idx = self.order_book.ask_prices.index(best_ask_above_fair)
            best_ask_volume = self.order_book.ask_volumes[best_ask_idx]
            if (
                abs(best_ask_above_fair - self.fair_value) <= self.mm_join_edge
                and best_ask_volume <= self.mm_join_volume
            ):
                ask_price = best_ask_above_fair
            elif (
                abs(best_ask_above_fair - self.fair_value) <= self.mm_join_edge_2
                and best_ask_volume <= self.mm_join_volume_2
            ):
                ask_price = best_ask_above_fair
            else:
                ask_price = best_ask_above_fair - 1

        bid_price = round(self.fair_value - self.mm_default_edge)
        if best_bid_below_fair is not None:
            best_bid_idx = self.order_book.bid_prices.index(best_bid_below_fair)
            best_bid_volume = self.order_book.bid_volumes[best_bid_idx]
            if (
                abs(self.fair_value - best_bid_below_fair) <= self.mm_join_edge
                and best_bid_volume <= self.mm_join_volume
            ):  # best bid volume 3
                bid_price = best_bid_below_fair  # join BEST 0
            elif (
                abs(self.fair_value - best_bid_below_fair) <= self.mm_join_edge_2
                and best_bid_volume <= self.mm_join_volume_2
            ):  # best bid volume 1
                bid_price = best_bid_below_fair
            else:
                bid_price = best_bid_below_fair + 1  # penny

        bid_price = round(bid_price)
        ask_price = round(ask_price)

        bid_volume = min(self.mm_default_vol, self.remaining_buy)
        ask_volume = min(self.mm_default_vol, self.remaining_sell)

        # Create the orders if they make sense
        if bid_volume > 0:
            self.place_order(bid_price, bid_volume, "LIMIT")

        if ask_volume > 0:
            self.place_order(ask_price, -ask_volume, "LIMIT")

    def calculate_orders(self):
        # Liquidation
        self.liquidate_position()

        # Market taking
        self.market_take()

        # Market making
        self.market_make()


# ------------------KELP-------------------#
class Kelp(Product):
    def __init__(self, config):
        super().__init__(config)

        # Kelp parameters
        self.name = "Kelp"
        self.symbol = "KELP"
        self.pos_limit = 50

        # Price estimation
        self.detect_mm_volume = config.get("detect_mm_volume")
        self.fair_value = None
        self.last_mm_price = None
        self.last_fair_price = None

        # Market taking parameters
        self.mt_take_width = config.get("mt_take_width")
        self.mt_clear_width = config.get("mt_clear_width")
        self.mt_adverse_volume = config.get("mt_adverse_volume")

        # Market making parameters
        self.mm_default_vol = config.get("mm_default_vol")
        self.mm_disregard_edge = config.get("mm_disregard_edge")
        self.mm_default_edge = config.get("mm_default_edge")
        self.mm_join_edge = config.get("mm_join_edge")
        self.mm_join_volume = config.get("mm_join_volume")

    def update_product(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.logger.print_numeric("position", position)

        # Reset order book
        self.order_book.reset(order_depths)

        # Reset orders
        self.orders = []

        # Update position
        self.position = position
        self.remaining_buy = self.pos_limit - position
        self.remaining_sell = self.pos_limit + position

        # --------------Price estimation------------------
        mm_price = self.order_book.get_mm_fair(self.detect_mm_volume)
        self.logger.print_numeric("mm_price", mm_price)

        vwap = self.order_book.vwap
        self.logger.print_numeric("vwap", vwap)

        # DOUBLE CHECK THIS IF LAST MM OR VWAP
        if mm_price is None:
            current_price = self.last_mm_price
        else:
            current_price = mm_price
            self.last_mm_price = mm_price
        self.logger.print_numeric("current_price", current_price)

        self.fair_value = self.estimate_fair_value(current_price)
        self.logger.print_numeric("fair_value", self.fair_value)

    def estimate_fair_value(self, observed_price):
        # Initialize Kalman filter state if not exists
        if not hasattr(self, "kf_price"):
            self.kf_price = None  # Estimated state
            self.kf_variance = 1.0  # Uncertainty in the estimate
            self.process_variance = 0.1  # How quickly the true price changes
            self.measurement_variance = 0.1  # Noise in price observations

        if (
            len(self.order_book.ask_prices) != 0
            and len(self.order_book.bid_prices) != 0
        ):
            # Kalman filter prediction step
            if self.kf_price is None:
                self.kf_price = observed_price

            # Prior update (prediction)
            prior_variance = self.kf_variance + self.process_variance

            # Measurement update (correction)
            kalman_gain = prior_variance / (prior_variance + self.measurement_variance)
            self.kf_price = self.kf_price + kalman_gain * (
                observed_price - self.kf_price
            )
            self.kf_variance = (1 - kalman_gain) * prior_variance

            # Use the Kalman filter estimate as our fair price
            fair = self.kf_price
            self.last_fair_price = fair
            return fair

        return None

    def market_make(self):
        asks_above_fair = [
            price
            for price in self.order_book.ask_prices
            if price > self.fair_value + self.mm_disregard_edge
        ]
        bids_below_fair = [
            price
            for price in self.order_book.bid_prices
            if price < self.fair_value - self.mm_disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(self.fair_value + self.mm_default_edge)
        if best_ask_above_fair is not None:
            baaf_idx = self.order_book.ask_prices.index(best_ask_above_fair)
            best_ask_volume = self.order_book.ask_volumes[baaf_idx]
            if (
                abs(best_ask_above_fair - self.fair_value) <= self.mm_join_edge
                and best_ask_volume <= self.mm_join_volume
            ):
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(self.fair_value - self.mm_default_edge)
        if best_bid_below_fair is not None:
            bbbf_idx = self.order_book.bid_prices.index(best_bid_below_fair)
            best_bid_volume = self.order_book.bid_volumes[bbbf_idx]
            if (
                abs(self.fair_value - best_bid_below_fair) <= self.mm_join_edge
                and best_bid_volume <= self.mm_join_volume
            ):
                bid = best_bid_below_fair

            else:
                bid = best_bid_below_fair + 1

        bid_price = round(bid)
        ask_price = round(ask)

        bid_volume = min(self.mm_default_vol, self.remaining_buy)
        ask_volume = min(self.mm_default_vol, self.remaining_sell)

        # Create the orders if they make sense
        if bid_volume > 0:
            self.place_order(bid_price, bid_volume, "LIMIT")

        if ask_volume > 0:
            self.place_order(ask_price, -ask_volume, "LIMIT")

    def market_take(self):
        ask_prices = self.order_book.get_ask_prices()
        ask_volumes = self.order_book.get_ask_volumes()
        ask_orders_depth = self.order_book.ask_orders_depth
        # Check if there is an opportunity to market take in ask orders
        for depth_level in range(ask_orders_depth):
            ask_price = ask_prices[depth_level]
            ask_volume = ask_volumes[depth_level]
            if ask_volume <= self.mt_adverse_volume:
                if self.fair_value - ask_price >= self.mt_take_width:
                    bid_price = ask_price
                    bid_volume = min(self.remaining_buy, ask_volume)
                    self.place_order(bid_price, bid_volume)
                else:
                    break  # If even the best ask doesn't cross the mean, then no need to check further

        bid_prices = self.order_book.get_bid_prices()
        bid_volumes = self.order_book.get_bid_volumes()
        bid_orders_depth = self.order_book.bid_orders_depth
        # Check if there is an opportunity to market take in bid orders
        for depth_level in range(bid_orders_depth):
            bid_price = bid_prices[depth_level]
            bid_volume = bid_volumes[depth_level]
            if ask_volume <= self.mt_adverse_volume:
                if bid_price - self.fair_value >= self.mt_take_width:
                    ask_price = bid_price
                    ask_volume = min(self.remaining_sell, bid_volume)
                    self.place_order(bid_price, bid_volume)

                else:
                    break  # If even the best bid doesn't cross the mean, then no need to check further

    def liquidate_position(self):
        bid_prices = self.order_book.get_bid_prices()
        bid_volumes = self.order_book.get_bid_volumes()
        bid_orders_depth = self.order_book.bid_orders_depth
        # Check if there is an opportunity to liquidate long positions
        for depth_level in range(bid_orders_depth):
            if self.position > 0:
                bid_price = bid_prices[depth_level]
                bid_volume = bid_volumes[depth_level]
                if bid_volume <= self.mt_adverse_volume:
                    if bid_price - self.fair_value >= self.mt_clear_width:
                        qty = min(self.remaining_sell, bid_volume, self.position)
                        self.place_order(bid_price, -qty)
                    else:
                        break
            else:
                break

        ask_prices = self.order_book.get_ask_prices()
        ask_volumes = self.order_book.get_ask_volumes()
        ask_orders_depth = self.order_book.ask_orders_depth
        # Check if there is an opportunity to liquidate short positions
        for depth_level in range(ask_orders_depth):
            if self.position < 0:
                ask_price = ask_prices[depth_level]
                ask_volume = ask_volumes[depth_level]
                if ask_volume <= self.mt_adverse_volume:
                    if self.fair_value - ask_price >= self.mt_clear_width:
                        qty = min(self.remaining_buy, ask_volume, abs(self.position))
                        self.place_order(ask_price, qty)
                    else:
                        break
            else:
                break

    def calculate_orders(self):
        # Liquidation
        self.liquidate_position()

        # Market taking
        self.market_take()

        # Market making
        self.market_make()


# ------------------Squid Ink-------------------#
class Squid(Product):
    def __init__(self, config):
        super().__init__(config)

        # Squid parameters
        self.name = "Squid Ink"
        self.symbol = "SQUID_INK"
        self.pos_limit = 50

        # Price estimation
        self.detect_mm_volume = config.get("detect_mm_volume")
        self.short_window = config.get("short_window")
        self.short_history = deque(maxlen=self.short_window)
        self.long_window = config.get("long_window")
        self.long_history = deque(maxlen=self.long_window)

        # Directional trading
        self.dt_default_vol = config.get("dt_default_vol")
        self.dt_signal_strength = config.get("dt_signal_strength")

    def update_product(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.logger.print_numeric("position", position)

        # Reset order book
        self.order_book.reset(order_depths)

        # Reset orders
        self.orders = []

        # Update position
        self.position = position
        self.remaining_buy = self.pos_limit - position
        self.remaining_sell = self.pos_limit + position

        # --------------Price estimation------------------
        mm_price = self.order_book.get_mm_fair(self.detect_mm_volume)
        self.logger.print_numeric("mm_price", mm_price)
        vwap = self.order_book.vwap
        self.logger.print_numeric("vwap", vwap)

        if mm_price is None:
            current_price = self.order_book.vwap
        else:
            current_price = mm_price

        self.fair_value = current_price
        self.logger.print_numeric("fair_value", self.fair_value)

        self.short_history.append(self.fair_value)
        self.long_history.append(self.fair_value)

    def directional_trade(self):
        # Check if we have enough data points for both moving averages
        if (
            len(self.long_history) >= self.long_window
            and len(self.short_history) >= self.short_window
        ):
            long_mean = sum(self.long_history) / self.long_window
            short_mean = sum(self.short_history) / self.short_window
            self.logger.print_numeric("long_mean", long_mean)
            self.logger.print_numeric("short_mean", short_mean)

            percentage_diff = abs(long_mean - short_mean) / self.short_history[-1]
            self.logger.print_numeric("percentage_diff", percentage_diff)

            short_below_long = short_mean < long_mean
            if short_below_long:
                # Long signal
                if self.position >= 0:
                    signal_strength = self.dt_signal_strength
                    bid_volume = min(
                        self.dt_default_vol,
                        self.remaining_buy,
                    )
                else:
                    signal_strength = 0
                    bid_volume = min(self.remaining_buy, abs(self.position))

                if percentage_diff > signal_strength and self.remaining_buy > 0:
                    best_bid_price, best_bid_volume = self.order_book.get_best_ask()
                    bid_price = best_bid_price
                    bid_volume = min(bid_volume, best_bid_volume)
                    self.place_order(bid_price, bid_volume)

            elif not short_below_long:
                # Short signal
                if self.position <= 0:
                    signal_strength = self.dt_signal_strength
                    ask_volume = min(
                        self.dt_default_vol,
                        self.remaining_sell,
                    )
                else:
                    signal_strength = 0
                    ask_volume = min(self.remaining_sell, self.position)

                if self.remaining_sell > 0 and percentage_diff > signal_strength:
                    best_ask_price, best_ask_volume = self.order_book.get_best_bid()
                    ask_price = best_ask_price
                    ask_volume = min(ask_volume, best_ask_volume)
                    self.place_order(ask_price, -ask_volume)

    def calculate_orders(self):
        # Directional trading
        self.directional_trade()


# -----------------Croissant-----------------#
class Croissants(Product):
    def __init__(self, config):
        super().__init__(config)

        # Croissant parameters
        self.name = "Croissants"
        self.symbol = "CROISSANTS"
        self.pos_limit = 250

    def update_product(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.logger.print_numeric("position", position)

        # Reset order book
        self.order_book.reset(order_depths)

        # Reset orders
        self.orders = []

        # Update position
        self.position = position
        self.remaining_buy = self.pos_limit - position
        self.remaining_sell = self.pos_limit + position

    def calculate_orders(self):
        pass


# -----------------Jam-----------------#
class Jams(Product):
    def __init__(self, config):
        super().__init__(config)

        # Jam parameters
        self.name = "Jams"
        self.symbol = "JAMS"
        self.pos_limit = 350

    def update_product(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.logger.print_numeric("position", position)

        # Reset order book
        self.order_book.reset(order_depths)

        # Reset orders
        self.orders = []

        # Update position
        self.position = position
        self.remaining_buy = self.pos_limit - position
        self.remaining_sell = self.pos_limit + position

    def calculate_orders(self):
        pass


# -----------------Djembe-----------------#
class Djembes(Product):
    def __init__(self, config):
        super().__init__(config)

        # Djembe parameters
        self.name = "Djembes"
        self.symbol = "DJEMBES"
        self.pos_limit = 60

    def update_product(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.logger.print_numeric("position", position)

        # Reset order book
        self.order_book.reset(order_depths)

        # Reset orders
        self.orders = []

        # Update position
        self.position = position
        self.remaining_buy = self.pos_limit - position
        self.remaining_sell = self.pos_limit + position

    def calculate_orders(self):
        pass


# -------------Picnic Basket 1 ----------------#
class PicnicBasket1(Product):
    def __init__(self, config):
        super().__init__(config)

        # Picnic Basket 1 parameters
        self.name = "Picnic Basket 1"
        self.symbol = "PICNIC_BASKET1"
        self.pos_limit = 60

        # Price estimation
        self.detect_mm_volume = config.get("detect_mm_volume")

        # Market making parameters
        self.mm_default_vol = config.get("mm_default_vol")
        self.mm_default_edge = config.get("mm_default_edge")
        self.mm_disregard_edge = config.get("mm_disregard_edge")
        self.mm_join_edge = config.get("mm_join_edge")
        self.mm_join_volume = config.get("mm_join_volume")

    def update_product(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.logger.print_numeric("position", position)

        # Reset order book
        self.order_book.reset(order_depths)

        # Reset orders
        self.orders = []

        # Update position
        self.position = position
        self.remaining_buy = self.pos_limit - position
        self.remaining_sell = self.pos_limit + position

        # --------------Price estimation------------------
        mm_price = self.order_book.get_mm_fair(self.detect_mm_volume)
        self.logger.print_numeric("mm_price", mm_price)
        vwap = self.order_book.vwap
        self.logger.print_numeric("vwap", vwap)

        if mm_price is None:
            current_price = self.order_book.vwap
        else:
            current_price = mm_price

        self.fair_value = current_price
        self.logger.print_numeric("fair_value", self.fair_value)

    def market_make(self):
        asks_above_fair = [
            price
            for price in self.order_book.ask_prices
            if price > self.fair_value + self.mm_disregard_edge
        ]
        bids_below_fair = [
            price
            for price in self.order_book.bid_prices
            if price < self.fair_value - self.mm_disregard_edge
        ]

        baaf = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        bbbf = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(self.fair_value + self.mm_default_edge)
        if baaf is not None:
            baaf_idx = self.order_book.ask_prices.index(baaf)
            best_ask_volume = self.order_book.ask_volumes[baaf_idx]
            if (
                baaf - self.fair_value <= self.mm_join_edge
                and best_ask_volume <= self.mm_join_volume
            ):
                ask = baaf - 1
            else:
                ask = baaf - 2

        bid = round(self.fair_value - self.mm_default_edge)
        if bbbf is not None:
            bbbf_idx = self.order_book.bid_prices.index(bbbf)
            best_bid_volume = self.order_book.bid_volumes[bbbf_idx]
            if (
                abs(self.fair_value - bbbf) <= self.mm_join_edge
                and best_bid_volume <= self.mm_join_volume
            ):
                bid = bbbf + 1

            else:
                bid = bbbf + 2

        bid_price = round(bid)
        ask_price = round(ask)

        bid_volume = min(self.mm_default_vol, self.remaining_buy)
        ask_volume = min(self.mm_default_vol, self.remaining_sell)

        # Create the orders if they make sense
        if bid_volume > 0:
            self.place_order(bid_price, bid_volume, "LIMIT", update_order_book=False)

        if ask_volume > 0:
            self.place_order(ask_price, -ask_volume, "LIMIT", update_order_book=False)

    def calculate_orders(self):
        # Market making
        self.market_make()

        pass


# -------------Picnic Basket 2 ----------------#
class PicnicBasket2(Product):
    def __init__(self, config):
        super().__init__(config)

        # Picnic Basket 2 parameters
        self.name = "Picnic Basket 2"
        self.symbol = "PICNIC_BASKET2"
        self.pos_limit = 100

        # Price estimation
        self.detect_mm_volume = config.get("detect_mm_volume")

        # Market making parameters
        self.mm_default_vol = config.get("mm_default_vol")
        self.mm_default_edge = config.get("mm_default_edge")
        self.mm_disregard_edge = config.get("mm_disregard_edge")
        self.mm_join_edge = config.get("mm_join_edge")
        self.mm_join_volume = config.get("mm_join_volume")

    def update_product(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.logger.print_numeric("position", position)

        # Reset order book
        self.order_book.reset(order_depths)

        # Reset orders
        self.orders = []

        # Update position
        self.position = position
        self.remaining_buy = self.pos_limit - position
        self.remaining_sell = self.pos_limit + position

        # --------------Price estimation------------------
        mm_price = self.order_book.get_mm_fair(self.detect_mm_volume)
        self.logger.print_numeric("mm_price", mm_price)
        vwap = self.order_book.vwap
        self.logger.print_numeric("vwap", vwap)

        if mm_price is None:
            current_price = self.order_book.vwap
        else:
            current_price = mm_price

        self.fair_value = current_price
        self.logger.print_numeric("fair_value", self.fair_value)

    def market_make(self):
        asks_above_fair = [
            price
            for price in self.order_book.ask_prices
            if price > self.fair_value + self.mm_disregard_edge
        ]
        bids_below_fair = [
            price
            for price in self.order_book.bid_prices
            if price < self.fair_value - self.mm_disregard_edge
        ]

        baaf = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        bbbf = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(self.fair_value + self.mm_default_edge)
        if baaf is not None:
            baaf_idx = self.order_book.ask_prices.index(baaf)
            best_ask_volume = self.order_book.ask_volumes[baaf_idx]
            if (
                baaf - self.fair_value <= self.mm_join_edge
                and best_ask_volume <= self.mm_join_volume
            ):
                ask = baaf
            else:
                ask = baaf - 1

        bid = round(self.fair_value - self.mm_default_edge)
        if bbbf is not None:
            bbbf_idx = self.order_book.bid_prices.index(bbbf)
            best_bid_volume = self.order_book.bid_volumes[bbbf_idx]
            if (
                abs(self.fair_value - bbbf) <= self.mm_join_edge
                and best_bid_volume <= self.mm_join_volume
            ):
                bid = bbbf

            else:
                bid = bbbf + 1

        bid_price = round(bid)
        ask_price = round(ask)

        bid_volume = min(self.mm_default_vol, self.remaining_buy)
        ask_volume = min(self.mm_default_vol, self.remaining_sell)

        # Create the orders if they make sense
        if bid_volume > 0:
            self.place_order(bid_price, bid_volume, "LIMIT")

        if ask_volume > 0:
            self.place_order(ask_price, -ask_volume, "LIMIT")

    def calculate_orders(self):
        # Market making
        # self.market_make()

        pass


# Synthetic Basket 1
class SyntheticBasket1(Product):
    def __init__(self, config):
        super().__init__(config)

        # Synthetic Basket 1 parameters
        self.name = "Synthetic Basket 1"
        self.symbol = "SYNTHETIC_BASKET1"

        # Constituent products
        self.composition = ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]

        # Open and close position thresholds
        self.N = config.get("N")
        self.buy_entry = config.get("buy_entry")
        self.buy_exit = config.get("buy_exit")
        self.sell_entry = config.get("sell_entry")
        self.sell_exit = config.get("sell_exit")

        # Price tracking
        self.converge_window = 25
        self.BUY_SPREAD_MEAN = 59.36
        self.BUY_SPREAD_VAR = 7246.50
        self.SELL_SPREAD_MEAN = 36.72
        self.SELL_SPREAD_VAR = 7250.25

        self.buy_spread_stats = WelfordStatsWithPriors(
            self.BUY_SPREAD_MEAN, self.BUY_SPREAD_VAR, self.N
        )
        self.sell_spread_stats = WelfordStatsWithPriors(
            self.SELL_SPREAD_MEAN, self.SELL_SPREAD_VAR, self.N
        )

        # Theoretical max is 41
        self.max_basket_position = 40
        self.baskets_long = 0
        self.baskets_short = 0

        self.iter = 0

        self.orders = []

    def update_product(self, order_depths, position, own_trades, timestamp):
        pass

    def calculate_orders(self, products, timestamp):
        self.print_product_begin(timestamp)

        # Set timestamp
        self.timestamp = timestamp
        self.iter += 1

        for constituent in self.composition:
            if constituent not in products.keys():
                return
            if products[constituent].order_book.check_if_no_orders():
                return

        pb1 = products["PICNIC_BASKET1"]
        croissants = products["CROISSANTS"]
        jams = products["JAMS"]
        djembes = products["DJEMBES"]

        pb1_ask_price, pb1_ask_volume = pb1.order_book.get_best_ask()
        pb1_bid_price, pb1_bid_volume = pb1.order_book.get_best_bid()

        croissants_ask_price, croissants_ask_volume = (
            croissants.order_book.get_best_ask()
        )
        croissants_bid_price, croissants_bid_volume = (
            croissants.order_book.get_best_bid()
        )

        jams_ask_price, jams_ask_volume = jams.order_book.get_best_ask()
        jams_bid_price, jams_bid_volume = jams.order_book.get_best_bid()

        djembes_ask_price, djembes_ask_volume = djembes.order_book.get_best_ask()
        djembes_bid_price, djembes_bid_volume = djembes.order_book.get_best_bid()

        buy_spread = pb1_ask_price - (
            6 * croissants_bid_price + 3 * jams_bid_price + 1 * djembes_bid_price
        )
        sell_spread = pb1_bid_price - (
            6 * croissants_ask_price + 3 * jams_ask_price + 1 * djembes_ask_price
        )

        self.buy_spread_stats.update(buy_spread)
        self.sell_spread_stats.update(sell_spread)

        self.logger.print_numeric("buy_spread", buy_spread)
        self.logger.print_numeric("sell_spread", sell_spread)

        if self.iter < self.converge_window:
            self.on_timestep_end()
            return

        # BASKET BUY STRATEGY (Long PB1, Short Components)
        # Calculate max basket units based on position limits
        # How many complete baskets can we trade?
        basket_buy_limits = [
            pb1.remaining_buy,  # Each basket needs 1 PB1
            croissants.remaining_sell // 6,  # Each basket needs to short 6 Croissants
            jams.remaining_sell // 3,  # Each basket needs to short 3 Jams
            djembes.remaining_sell,  # Each basket needs to short 1 Djembe
        ]

        # Calculate max basket units based on available market liquidity
        liquidity_buy_limits = [
            pb1_ask_volume,  # Can only buy what's available
            croissants_bid_volume // 6,  # Can only sell what someone's willing to buy
            jams_bid_volume // 3,
            djembes_bid_volume,
        ]
        max_baskets_buy = min(min(basket_buy_limits), min(liquidity_buy_limits))

        # BASKET SELL STRATEGY (Short PB1, Long Components)
        # Calculate max basket units based on position limits
        basket_sell_limits = [
            pb1.remaining_sell,  # Each basket needs to short 1 PB1
            croissants.remaining_buy // 6,  # Each basket needs to buy 6 Croissants
            jams.remaining_buy // 3,  # Each basket needs to buy 3 Jams
            djembes.remaining_buy,  # Each basket needs to buy 1 Djembe
        ]

        # Calculate max basket units based on available market liquidity
        liquidity_sell_limits = [
            pb1_bid_volume,  # Can only sell what someone's willing to buy
            croissants_ask_volume // 6,  # Can only buy what's available
            jams_ask_volume // 3,
            djembes_ask_volume,
        ]

        # The limiting factor is the minimum of both constraints
        max_baskets_sell = min(min(basket_sell_limits), min(liquidity_sell_limits))

        buy_std = self.buy_spread_stats.get_std()
        z_score_buy = (buy_spread - self.BUY_SPREAD_MEAN) / buy_std
        self.logger.print_numeric("z_score_buy", z_score_buy)

        sell_std = self.sell_spread_stats.get_std()
        z_score_sell = (sell_spread - self.SELL_SPREAD_MEAN) / sell_std
        self.logger.print_numeric("z_score_sell", z_score_sell)

        if (
            z_score_buy <= -self.buy_entry
            and max_baskets_buy > 0
            and self.baskets_long < self.max_basket_position
        ):
            available_for_new_positions = self.max_basket_position - self.baskets_long
            new_baskets = min(max_baskets_buy, available_for_new_positions)

            # Calculate exact volumes while respecting the basket ratio
            if new_baskets > 0:
                pb1_buy_volume = new_baskets
                croissants_sell_volume = new_baskets * 6
                jams_sell_volume = new_baskets * 3
                djembes_sell_volume = new_baskets

                # Place orders
                pb1.place_order(pb1_ask_price, pb1_buy_volume)
                croissants.place_order(croissants_bid_price, -croissants_sell_volume)
                jams.place_order(jams_bid_price, -jams_sell_volume)
                djembes.place_order(djembes_bid_price, -djembes_sell_volume)

                self.baskets_long += new_baskets
                self.logger.print_numeric("open_buy_spread", buy_spread)

        elif (
            z_score_buy >= -self.buy_exit
            and self.baskets_long > 0
            and max_baskets_sell > 0
        ):
            # Determine how many baskets to unwind
            baskets_to_unwind = min(max_baskets_sell, self.baskets_long)

            if baskets_to_unwind > 0:
                pb1_sell_volume = baskets_to_unwind
                croissants_buy_volume = baskets_to_unwind * 6
                jams_buy_volume = baskets_to_unwind * 3
                djembes_buy_volume = baskets_to_unwind

                # Place orders
                pb1.place_order(pb1_bid_price, -pb1_sell_volume)
                croissants.place_order(croissants_ask_price, croissants_buy_volume)
                jams.place_order(jams_ask_price, jams_buy_volume)
                djembes.place_order(djembes_ask_price, djembes_buy_volume)

                self.baskets_long -= baskets_to_unwind
                self.logger.print_numeric("close_buy_spread", buy_spread)

        if (
            z_score_sell >= self.sell_entry
            and self.baskets_short < self.max_basket_position
            and max_baskets_sell > 0
        ):
            available_for_new_positions = self.max_basket_position - self.baskets_short
            new_baskets = min(max_baskets_sell, available_for_new_positions)

            if new_baskets > 0:
                pb1_sell_volume = new_baskets
                croissants_buy_volume = new_baskets * 6
                jams_buy_volume = new_baskets * 3
                djembes_buy_volume = new_baskets

                # Place orders
                pb1.place_order(pb1_bid_price, -pb1_sell_volume)
                croissants.place_order(croissants_ask_price, croissants_buy_volume)
                jams.place_order(jams_ask_price, jams_buy_volume)
                djembes.place_order(djembes_ask_price, djembes_buy_volume)

                self.baskets_short += new_baskets
                self.logger.print_numeric("open_sell_spread", sell_spread)
        elif (
            z_score_sell <= self.sell_exit
            and self.baskets_short > 0
            and max_baskets_buy > 0
        ):
            # Determine how many baskets to unwind
            baskets_to_unwind = min(max_baskets_buy, self.baskets_short)

            if baskets_to_unwind:
                pb1_buy_volume = baskets_to_unwind
                croissants_sell_volume = baskets_to_unwind * 6
                jams_sell_volume = baskets_to_unwind * 3
                djembes_sell_volume = baskets_to_unwind

                # Place orders
                pb1.place_order(pb1_ask_price, pb1_buy_volume)
                croissants.place_order(croissants_bid_price, -croissants_sell_volume)
                jams.place_order(jams_bid_price, -jams_sell_volume)
                djembes.place_order(djembes_bid_price, -djembes_sell_volume)

                self.baskets_short -= baskets_to_unwind
                self.logger.print_numeric("close_sell_spread", sell_spread)

        self.on_timestep_end()


# Synthetic Basket 1
class SyntheticBasket2(Product):
    def __init__(self, config):
        super().__init__(config)

        # Synthetic Basket 1 parameters
        self.name = "Synthetic Basket 2"
        self.symbol = "SYNTHETIC_BASKET2"

        # Constituent products
        self.composition = [
            "PICNIC_BASKET2",
            "CROISSANTS",
            "JAMS",
        ]

        # Open and close position thresholds
        self.N = config.get("N")
        self.buy_entry = config.get("buy_entry")
        self.buy_exit = config.get("buy_exit")
        self.sell_entry = config.get("sell_entry")
        self.sell_exit = config.get("sell_exit")

        # Price tracking
        self.converge_window = 25
        self.BUY_SPREAD_MEAN = 36.89
        self.BUY_SPREAD_VAR = 3582.76
        self.SELL_SPREAD_MEAN = 23.58
        self.SELL_SPREAD_VAR = 3583.29

        self.buy_spread_stats = WelfordStatsWithPriors(
            self.BUY_SPREAD_MEAN, self.BUY_SPREAD_VAR, self.N
        )
        self.sell_spread_stats = WelfordStatsWithPriors(
            self.SELL_SPREAD_MEAN, self.SELL_SPREAD_VAR, self.N
        )

        # Theoretical max is 62
        self.max_basket_position = 60
        self.baskets_long = 0
        self.baskets_short = 0

        self.iter = 0

        self.orders = []

    def update_product(self, order_depths, position, own_trades, timestamp):
        pass

    def calculate_orders(self, products, timestamp):
        self.print_product_begin(timestamp)

        self.iter += 1

        for constituent in self.composition:
            if constituent not in products.keys():
                return
            if products[constituent].order_book.check_if_no_orders():
                return

        pb2 = products["PICNIC_BASKET2"]
        croissants = products["CROISSANTS"]
        jams = products["JAMS"]

        pb2_ask_price, pb2_ask_volume = pb2.order_book.get_best_ask()
        pb2_bid_price, pb2_bid_volume = pb2.order_book.get_best_bid()

        croissants_ask_price, croissants_ask_volume = (
            croissants.order_book.get_best_ask()
        )
        croissants_bid_price, croissants_bid_volume = (
            croissants.order_book.get_best_bid()
        )

        jams_ask_price, jams_ask_volume = jams.order_book.get_best_ask()
        jams_bid_price, jams_bid_volume = jams.order_book.get_best_bid()

        buy_spread = pb2_ask_price - (4 * croissants_bid_price + 2 * jams_bid_price)
        sell_spread = pb2_bid_price - (4 * croissants_ask_price + 2 * jams_ask_price)

        self.buy_spread_stats.update(buy_spread)
        self.sell_spread_stats.update(sell_spread)

        self.logger.print_numeric("buy_spread", buy_spread)
        self.logger.print_numeric("sell_spread", sell_spread)

        if self.iter < self.iters_to_converge:
            self.on_timestep_end()
            return

        # BASKET BUY STRATEGY (Long PB2, Short Components)
        # Calculate max basket units based on position limits
        # How many complete baskets can we trade?
        basket_buy_limits = [
            pb2.remaining_buy,  # Each basket needs 1 PB2
            croissants.remaining_sell // 4,  # Each basket needs to short 4 Croissants
            jams.remaining_sell // 2,  # Each basket needs to short 2 Jams
        ]

        # Calculate max basket units based on available market liquidity
        liquidity_buy_limits = [
            pb2_ask_volume,  # Can only buy what's available
            croissants_bid_volume // 4,  # Can only sell what someone's willing to buy
            jams_bid_volume // 2,
        ]
        max_baskets_buy = min(min(basket_buy_limits), min(liquidity_buy_limits))

        # BASKET SELL STRATEGY (Short PB2, Long Components)
        # Calculate max basket units based on position limits
        basket_sell_limits = [
            pb2.remaining_sell,  # Each basket needs to short 1 PB2
            croissants.remaining_buy // 4,  # Each basket needs to buy 6 Croissants
            jams.remaining_buy // 2,  # Each basket needs to buy 3 Jams
        ]

        # Calculate max basket units based on available market liquidity
        liquidity_sell_limits = [
            pb2_bid_volume,  # Can only sell what someone's willing to buy
            croissants_ask_volume // 4,  # Can only buy what's available
            jams_ask_volume // 2,
        ]

        # The limiting factor is the minimum of both constraints
        max_baskets_sell = min(min(basket_sell_limits), min(liquidity_sell_limits))

        buy_std = self.buy_spread_stats.get_std()
        z_score_buy = (buy_spread - self.BUY_SPREAD_MEAN) / buy_std
        self.logger.print_numeric("z_score_buy", z_score_buy)

        sell_std = self.sell_spread_stats.get_std()
        z_score_sell = (sell_spread - self.SELL_SPREAD_MEAN) / sell_std
        self.logger.print_numeric("z_score_sell", z_score_sell)

        if (
            z_score_buy <= -self.buy_entry
            and max_baskets_buy > 0
            and self.baskets_long < self.max_basket_position
        ):
            available_for_new_positions = self.max_basket_position - self.baskets_long
            new_baskets = min(max_baskets_buy, available_for_new_positions)

            # Calculate exact volumes while respecting the basket ratio
            if new_baskets > 0:
                pb2_buy_volume = new_baskets
                croissants_sell_volume = new_baskets * 4
                jams_sell_volume = new_baskets * 2

                # Place orders
                pb2.place_order(pb2_ask_price, pb2_buy_volume)
                croissants.place_order(croissants_bid_price, -croissants_sell_volume)
                jams.place_order(jams_bid_price, -jams_sell_volume)

                self.baskets_long += new_baskets

                self.logger.print_numeric("buy_spread_open", buy_spread)

        elif (
            z_score_buy >= -self.buy_exit
            and self.baskets_long > 0
            and max_baskets_sell > 0
        ):
            # Determine how many baskets to unwind
            baskets_to_unwind = min(max_baskets_sell, self.baskets_long)

            if baskets_to_unwind > 0:
                pb2_sell_volume = baskets_to_unwind
                croissants_buy_volume = baskets_to_unwind * 4
                jams_buy_volume = baskets_to_unwind * 2

                # Place orders
                pb2.place_order(pb2_bid_price, -pb2_sell_volume)
                croissants.place_order(croissants_ask_price, croissants_buy_volume)
                jams.place_order(jams_ask_price, jams_buy_volume)

                self.baskets_long -= baskets_to_unwind

                self.logger.print_numeric("buy_spread_close", buy_spread)

        if (
            z_score_sell >= self.sell_entry
            and self.baskets_short < self.max_basket_position
            and max_baskets_sell > 0
        ):
            available_for_new_positions = self.max_basket_position - self.baskets_short
            new_baskets = min(max_baskets_sell, available_for_new_positions)

            if new_baskets > 0:
                pb2_sell_volume = new_baskets
                croissants_buy_volume = new_baskets * 4
                jams_buy_volume = new_baskets * 2

                # Place orders
                pb2.place_order(pb2_bid_price, -pb2_sell_volume)
                croissants.place_order(croissants_ask_price, croissants_buy_volume)
                jams.place_order(jams_ask_price, jams_buy_volume)

                self.baskets_short += new_baskets

                self.logger.print_numeric("sell_spread_open", sell_spread)
        elif (
            z_score_sell <= self.sell_exit
            and self.baskets_short > 0
            and max_baskets_buy > 0
        ):
            # Determine how many baskets to unwind
            baskets_to_unwind = min(max_baskets_buy, self.baskets_short)

            if baskets_to_unwind:
                pb2_buy_volume = baskets_to_unwind
                croissants_sell_volume = baskets_to_unwind * 4
                jams_sell_volume = baskets_to_unwind * 2

                # Place orders
                pb2.place_order(pb2_ask_price, pb2_buy_volume)
                croissants.place_order(croissants_bid_price, -croissants_sell_volume)
                jams.place_order(jams_bid_price, -jams_sell_volume)

                self.baskets_short -= baskets_to_unwind

                self.logger.print_numeric("sell_spread_close", sell_spread)

        self.on_timestep_end()

# Code from trader.py
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
    "mm_default_vol": 10,
    "mm_default_edge": 4,
    "mm_disregard_edge": 2,
    "mm_join_edge": 6,
    "mm_join_volume": 5,
}

config_picnic_basket_2 = {
    "detect_mm_volume": 15,  # Volume to detect market maker
    # Market making parameters
    "mm_default_vol": 10,
    "mm_default_edge": 4,
    "mm_disregard_edge": 2,
    "mm_join_edge": 6,
    "mm_join_volume": 5,
}

config_synthetic_basket_1 = {
    "N": 10,
    "buy_entry": 1.5,
    "buy_exit": 0.5,
    "sell_entry": 1.5,
    "sell_exit": 0.5,
}

config_synthetic_basket_2 = {
    "N": 50,
    "buy_entry": 1.5,
    "buy_exit": 0.5,
    "sell_entry": 1.5,
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
            # -------------------Normal products -------------------
            products = {}
            # products["RAINFOREST_RESIN"] = RainforestResin(config_rainforest)
            # products["KELP"] = Kelp(config_kelp)
            # products["SQUID_INK"] = Squid(config_squid)
            products["CROISSANTS"] = Croissants(config_croissants)
            products["JAMS"] = Jams(config_jams)
            products["DJEMBES"] = Djembes(config_djembes)
            products["PICNIC_BASKET1"] = PicnicBasket1(config_picnic_basket_1)
            products["PICNIC_BASKET2"] = PicnicBasket2(config_picnic_basket_2)
            products["SYNTHETIC_BASKET1"] = SyntheticBasket1(config_synthetic_basket_1)
            # products["SYNTHETIC_BASKET2"] = SyntheticBasket2(config_synthetic_basket_2)
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

                products[product].update_product(
                    order_depth, position, own_trades, timestamp
                )
                products[product].calculate_orders()

        for product in ["SYNTHETIC_BASKET1", "SYNTHETIC_BASKET2"]:
            try:
                if product in products:
                    # Check that all dependencies exist
                    dependent_products = [
                        "PICNIC_BASKET1",
                        "CROISSANTS",
                        "JAMS",
                        "DJEMBES",
                    ]
                    missing_products = [
                        p for p in dependent_products if p not in products
                    ]

                    if not missing_products:
                        products[product].calculate_orders(products, timestamp)
                        # self.logger.print(
                        #     f"Successfully calculated orders for {product}"
                        # )
                #     else:
                #         self.logger.print(
                #             f"Cannot calculate {product}, missing: {missing_products}"
                #         )
                # # else:
                #     self.logger.print(
                #         f"Warning: {product} not found in products dictionary"
                #     )
            except Exception as e:
                # self.logger.print(f"Error processing {product}: {str(e)}")
                pass

        for product in state.order_depths:
            if product in products.keys():
                result[product] = products[product].on_timestep_end()

        traderData = dict()
        traderData["products"] = products
        traderData = jsonpickle.encode(traderData)

        conversions = 1

        t2 = time()

        self.logger.print_numeric("runtime", t2 - t1)
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
            self.logs += f"{label} {value:.5f}"
        else:
            self.logs += f"{label} {value}"

        self.logs += end

    def flush(self):
        print(self.logs)
        self.logs = ""


class WelfordStatsWithPriors:
    def __init__(self, initial_mean=None, initial_variance=None, initial_count=None):
        self.n = initial_count if initial_mean is not None else 0
        self.mean = initial_mean if initial_mean is not None else 0.0
        self.M2 = (
            initial_variance * initial_count if initial_variance is not None else 0.0
        )

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_std(self):
        return np.sqrt(self.M2 / self.n if self.n > 1 else 1.0)