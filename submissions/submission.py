# Combined Python Files
# Files combined: market_utils.py, products.py, trader.py, utils.py

# Import statements
from abc import ABC, abstractmethod

from collections import deque

from copy import deepcopy

from datamodel import Order

from datamodel import TradingState

from math import ceil, floor

from time import time

from typing import Any

import jsonpickle



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

    def get_best_bid(self):
        if len(self.bid_prices) == 0:
            return None
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
        self.mean = 10000

        # Order book
        self.order_book = OrderBook()
        self.update_order_book = config.get("update_order_book", True)

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

    def market_take(self, fair_value, position, remaining_buy, remaining_sell):
        orders = []

        # Check if there is an opportunity to market take in ask orders
        for depth_level in range(self.order_book.ask_orders_depth):
            ask_price, ask_volume = self.order_book.get_ask_order_at_depth(depth_level)
            if fair_value - ask_price >= self.mt_ask_edge:
                bid_price = ask_price
                bid_volume = min(remaining_buy, ask_volume)
                bid_order = Order(self.symbol, bid_price, bid_volume)
                orders.append(bid_order)
                # update positions and sell/buy volumes
                remaining_buy -= bid_volume
                position += bid_volume
            else:
                break  # If even the best ask doesn't cross the mean, then no need to check further

        # Check if there is an opportunity to market take in bid orders
        for depth_level in range(self.order_book.bid_orders_depth):
            bid_price, bid_volume = self.order_book.get_bid_order_at_depth(depth_level)
            if bid_price - fair_value >= self.mt_bid_edge:
                ask_price = bid_price
                ask_volume = min(remaining_sell, bid_volume)
                ask_order = Order(self.symbol, ask_price, -ask_volume)
                orders.append(ask_order)
                # update positions and sell/buy volumes
                remaining_sell -= ask_volume
                position -= ask_volume
            else:
                break  # If even the best bid doesn't cross the mean, then no need to check further

        if self.update_order_book:
            for order in orders:
                self.order_book.update(order)

        return orders, position, remaining_buy, remaining_sell

    def liquidate_position(self, fair_value, position, remaining_buy, remaining_sell):
        orders = []

        # Check if there is an opportunity to liquidate long positions
        for depth_level in range(self.order_book.bid_orders_depth):
            if position > 0:
                bid_price, bid_volume = self.order_book.get_bid_order_at_depth(
                    depth_level
                )
                if bid_price - fair_value >= self.mt_long_profit_margin:
                    qty = min(remaining_sell, bid_volume, position)
                    ask_order = Order(self.symbol, bid_price, -qty)
                    orders.append(ask_order)
                    # update positions and remaining buy/sell volumes
                    remaining_sell -= qty
                    position -= qty
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
                if fair_value - ask_price >= self.mt_short_profit_margin:
                    qty = min(remaining_buy, ask_volume, abs(position))
                    bid_order = Order(self.symbol, ask_price, qty)
                    orders.append(bid_order)
                    # update positions and remaining buy/sell volumes
                    remaining_buy -= qty
                    position += qty
                else:
                    break
            else:
                break

        if self.update_order_book:
            for order in orders:
                self.order_book.update(order)

        return orders, position, remaining_buy, remaining_sell

    def market_make(self, fair_value, position, remaining_buy, remaining_sell):
        orders = []

        asks_above_fair = [
            price
            for price in self.order_book.ask_prices
            if price > fair_value + self.mm_disregard_edge
        ]
        bids_below_fair = [
            price
            for price in self.order_book.bid_prices
            if price < fair_value - self.mm_disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask_price = round(fair_value + self.mm_default_edge)
        if best_ask_above_fair is not None:
            best_ask_idx = self.order_book.ask_prices.index(best_ask_above_fair)
            best_ask_volume = self.order_book.ask_volumes[best_ask_idx]
            if (
                abs(best_ask_above_fair - fair_value) <= self.mm_join_edge
                and best_ask_volume <= self.mm_join_volume
            ):  # best ask volume 1
                ask_price = best_ask_above_fair
            elif (
                abs(best_ask_above_fair - fair_value) <= self.mm_join_edge_2
                and best_ask_volume <= self.mm_join_volume_2
            ):
                ask_price = best_ask_above_fair
            else:
                ask_price = best_ask_above_fair - 1  #

        bid_price = round(fair_value - self.mm_default_edge)
        if best_bid_below_fair is not None:
            best_bid_idx = self.order_book.bid_prices.index(best_bid_below_fair)
            best_bid_volume = self.order_book.bid_volumes[best_bid_idx]
            if (
                abs(fair_value - best_bid_below_fair) <= self.mm_join_edge
                and best_bid_volume <= self.mm_join_volume
            ):  # best bid volume 3
                bid_price = best_bid_below_fair  # join BEST 0
            elif (
                abs(fair_value - best_bid_below_fair) <= self.mm_join_edge_2
                and best_bid_volume <= self.mm_join_volume_2
            ):  # best bid volume 1
                bid_price = best_bid_below_fair
            else:
                bid_price = best_bid_below_fair + 1  # penny

        bid_price = round(bid_price)
        ask_price = round(ask_price)

        bid_volume = min(self.mm_default_vol, remaining_buy)
        ask_volume = min(self.mm_default_vol, remaining_sell)

        # Create the orders if they make sense
        if bid_volume > 0:
            bid_order = Order(self.symbol, bid_price, bid_volume)
            orders.append(bid_order)

        if ask_volume > 0:
            ask_order = Order(self.symbol, ask_price, -ask_volume)
            orders.append(ask_order)

        return orders

    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)

        # Reset order book and track positions
        self.order_book.reset(order_depths)
        orders = []

        self.logger.print_numeric("position", position)

        fair_value = self.mean
        remaining_buy = self.pos_limit - position
        remaining_sell = self.pos_limit + position
        # ------------------------------------------------
        # Liquidation
        liquidated_orders, position, remaining_buy, remaining_sell = (
            self.liquidate_position(fair_value, position, remaining_buy, remaining_sell)
        )
        orders += liquidated_orders
        # Market taking
        mt_orders, position, remaining_buy, remaining_sell = self.market_take(
            fair_value, position, remaining_buy, remaining_sell
        )
        orders += mt_orders
        # ------------------------------------------------
        # Market making
        mm_orders = self.market_make(
            fair_value, position, remaining_buy, remaining_sell
        )
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
        self.update_order_book = config.get("update_order_book")
        self.order_book = OrderBook()

        # Price estimation
        self.detect_mm_volume = config.get("detect_mm_volume")
        self.last_mm_price = None
        self.last_fair_price = None

        # Market taking parameters
        self.mt_take_width = config.get("mt_take_width")
        self.mt_clear_width = config.get("mt_clear_width")
        self.mt_adverse_volume = config.get("mt_adverse_volume")

        # Market making parameters
        self.mm_disregard_edge = config.get("mm_disregard_edge")
        self.mm_default_vol = config.get("mm_default_vol")

        # Directional trading parameters
        self.dt_default_vol = config.get("dt_default_vol")
        self.dt_long_size = config.get("dt_long_size")
        self.dt_short_size = config.get("dt_short_size")
        self.dt_long_history = deque(maxlen=self.dt_long_size)
        self.dt_short_history = deque(maxlen=self.dt_short_size)

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

    def market_make_2(self, fair_value, position, remaining_buy, remaining_sell):
        orders = []

        disregard_edge = 1
        default_edge = 1
        join_edge = 0
        join_edge_2 = 0

        asks_above_fair = [
            price
            for price in self.order_book.ask_prices
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in self.order_book.bid_prices
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            baaf_idx = self.order_book.ask_prices.index(best_ask_above_fair)
            best_ask_volume = self.order_book.ask_volumes[baaf_idx]
            if (
                abs(best_ask_above_fair - fair_value) <= join_edge
                and best_ask_volume <= 3
            ):
                ask = best_ask_above_fair
            elif (
                abs(best_ask_above_fair - fair_value) <= join_edge_2
                and best_ask_volume <= 1
            ):
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            bbbf_idx = self.order_book.bid_prices.index(best_bid_below_fair)
            best_bid_volume = self.order_book.bid_volumes[bbbf_idx]
            if (
                abs(fair_value - best_bid_below_fair) <= join_edge
                and best_bid_volume <= 3
            ):
                bid = best_bid_below_fair
            elif (
                abs(fair_value - best_bid_below_fair) <= join_edge_2
                and best_bid_volume <= 1
            ):
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        bid_price = round(bid)
        ask_price = round(ask)

        bid_volume = min(self.mm_default_vol, remaining_buy)
        ask_volume = min(self.mm_default_vol, remaining_sell)

        # Create the orders if they make sense
        if bid_volume > 0:
            bid_order = Order(self.symbol, bid_price, bid_volume)
            orders.append(bid_order)

        if ask_volume > 0:
            ask_order = Order(self.symbol, ask_price, -ask_volume)
            orders.append(ask_order)

        return orders

    def market_take(self, fair_value, position, remaining_buy, remaining_sell):
        orders = []

        # Check if there is an opportunity to market take in ask orders
        for depth_level in range(self.order_book.ask_orders_depth):
            ask_price, ask_volume = self.order_book.get_ask_order_at_depth(depth_level)
            if ask_volume <= self.mt_adverse_volume:
                if fair_value - ask_price >= self.mt_take_width:
                    bid_price = ask_price
                    bid_volume = min(remaining_buy, ask_volume)
                    bid_order = Order(self.symbol, bid_price, bid_volume)
                    orders.append(bid_order)
                    # update positions and sell/buy volumes
                    remaining_buy -= bid_volume
                    position += bid_volume
                else:
                    break  # If even the best ask doesn't cross the mean, then no need to check further

        # Check if there is an opportunity to market take in bid orders
        for depth_level in range(self.order_book.bid_orders_depth):
            bid_price, bid_volume = self.order_book.get_bid_order_at_depth(depth_level)
            if ask_volume <= self.mt_adverse_volume:
                if bid_price - fair_value >= self.mt_take_width:
                    ask_price = bid_price
                    ask_volume = min(remaining_sell, bid_volume)
                    ask_order = Order(self.symbol, ask_price, -ask_volume)
                    orders.append(ask_order)
                    # update positions and sell/buy volumes
                    remaining_sell -= ask_volume
                    position -= ask_volume
                else:
                    break  # If even the best bid doesn't cross the mean, then no need to check further

        if self.update_order_book:
            for order in orders:
                self.order_book.update(order)

        return orders, position, remaining_buy, remaining_sell

    def liquidate_position(self, fair_value, position, remaining_buy, remaining_sell):
        orders = []

        # Check if there is an opportunity to liquidate long positions
        for depth_level in range(self.order_book.bid_orders_depth):
            if position > 0:
                bid_price, bid_volume = self.order_book.get_bid_order_at_depth(
                    depth_level
                )
                if bid_volume <= self.mt_adverse_volume:
                    if bid_price - fair_value >= self.mt_clear_width:
                        qty = min(remaining_sell, bid_volume, position)
                        ask_order = Order(self.symbol, bid_price, -qty)
                        orders.append(ask_order)
                        # update positions and remaining buy/sell volumes
                        remaining_sell -= qty
                        position -= qty
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
                if ask_volume <= self.mt_adverse_volume:
                    if fair_value - ask_price >= self.mt_clear_width:
                        qty = min(remaining_buy, ask_volume, abs(position))
                        bid_order = Order(self.symbol, ask_price, qty)
                        orders.append(bid_order)
                        # update positions and remaining buy/sell volumes
                        remaining_buy -= qty
                        position += qty
                    else:
                        break
            else:
                break

        if self.update_order_book:
            for order in orders:
                self.order_book.update(order)

        return orders, position, remaining_buy, remaining_sell

    def market_make(self, fair_value, position, remaining_buy, remaining_sell):
        orders = []

        mm_price = self.order_book.get_mm_fair(self.detect_mm_volume)
        if mm_price is None:
            return orders

        else:
            join_vol = 6

            bbbm = max(
                [price for price in self.order_book.bid_prices if price < mm_price]
            )
            bbbm_idx = self.order_book.bid_prices.index(bbbm)
            bbbm_vol = self.order_book.bid_volumes[bbbm_idx]

            baam = min(
                [price for price in self.order_book.ask_prices if price > mm_price]
            )
            baam_idx = self.order_book.ask_prices.index(baam)
            baam_vol = self.order_book.ask_volumes[baam_idx]

            if bbbm_vol < join_vol:
                bid_price = bbbm
            else:
                bid_price = bbbm + 1
            bid_price = min(bid_price, ceil(mm_price - 1))

            if baam_vol < join_vol:
                ask_price = baam
            else:
                ask_price = baam - 1
            ask_price = max(ask_price, floor(mm_price + 1))

        # Scale our order sizes based on how far we are from position limits
        bid_volume = min(self.mm_default_vol, remaining_buy)
        ask_volume = min(self.mm_default_vol, remaining_sell)

        # Create the orders if they make sense
        if bid_volume > 0:
            bid_order = Order(self.symbol, bid_price, bid_volume)
            orders.append(bid_order)

        if ask_volume > 0:
            ask_order = Order(self.symbol, ask_price, -ask_volume)
            orders.append(ask_order)

        return orders

    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.order_book.reset(order_depths)

        orders = []

        self.logger.print_numeric("position", position)
        remaining_buy = self.pos_limit - position
        remaining_sell = self.pos_limit + position

        # --------------Price estimation------------------
        mm_package = self.order_book.get_mm_fair(
            self.detect_mm_volume, with_spread=True
        )
        if mm_package is None:
            mm_price = None
            mm_spread = None
        else:
            mm_price = mm_package[0]
            mm_spread = mm_package[1]
        self.logger.print_numeric("mm_price", mm_price)
        self.logger.print_numeric("mm_spread", mm_spread)
        vwap = self.order_book.vwap
        self.logger.print_numeric("vwap", vwap)

        # DOUBLE CHECK THIS IF LAST MM OR VWAP
        if mm_price is None:
            current_price = self.last_mm_price
        else:
            current_price = mm_price
            self.last_mm_price = mm_price

        self.logger.print_numeric("current_price", current_price)

        fair_value = self.estimate_fair_value(current_price)
        self.logger.print_numeric("fair_value", fair_value)

        # ------------------------------------------------
        # Liquidation and market taking
        mt_orders, position, remaining_buy, remaining_sell = self.market_take(
            fair_value, position, remaining_buy, remaining_sell
        )
        orders += mt_orders

        liquidated_orders, position, remaining_buy, remaining_sell = (
            self.liquidate_position(fair_value, position, remaining_buy, remaining_sell)
        )
        orders += liquidated_orders
        # ------------------------------------------------
        # Market making
        orders += self.market_make_2(
            fair_value, position, remaining_buy, remaining_sell
        )

        self.print_orders(orders)
        self.print_product_end()

        return orders


# ------------------Squid Ink-------------------#
class Squid(Product):
    def __init__(self, config):
        super().__init__(config)

        # Squid parameters
        self.name = "Squid Ink"
        self.symbol = "SQUID_INK"
        self.pos_limit = 50

        # Order book
        self.update_order_book = config.get("update_order_book")
        self.order_book = OrderBook()

        # Price estimation
        self.detect_mm_volume = config.get("detect_mm_volume")
        self.short_window = config.get("short_window")
        self.short_history = deque(maxlen=self.short_window)
        self.long_window = config.get("long_window")
        self.long_history = deque(maxlen=self.long_window)

        # Directional trading
        self.dt_default_vol = config.get("dt_default_vol")
        self.dt_signal_strength = config.get("dt_signal_strength")

    def directional_trade(self, position, remaining_buy, remaining_sell):
        orders = []

        # Check if we have enough data points for both moving averages
        if (
            len(self.long_history) >= self.long_window
            and len(self.short_history) >= self.short_window
        ):
            long_mean = sum(self.long_history) / self.long_window
            short_mean = sum(self.short_history) / self.short_window
            self.logger.print_numeric("long_mean", long_mean)
            self.logger.print_numeric("short_mean", short_mean)

            # USE FOR THE VOLUME
            percentage_diff = abs(long_mean - short_mean) / self.short_history[-1]
            self.logger.print_numeric("percentage_diff", percentage_diff)
            # Current state
            short_below_long = short_mean < long_mean

            if short_below_long:
                # Long signal
                if position >= 0:
                    signal_strength = self.dt_signal_strength
                    bid_volume = min(
                        self.dt_default_vol,
                        remaining_buy,
                    )
                else:
                    signal_strength = 0
                    bid_volume = min(remaining_buy, abs(position))

                if percentage_diff > signal_strength and remaining_buy > 0:
                    best_bid_price, best_bid_volume = self.order_book.get_best_ask()
                    bid_price = best_bid_price
                    bid_volume = min(bid_volume, best_bid_volume)
                    bid_order = Order(self.symbol, bid_price, bid_volume)
                    orders.append(bid_order)
                    remaining_buy -= bid_volume
                    position += bid_volume

            elif not short_below_long:
                # Short signal
                if position <= 0:
                    signal_strength = self.dt_signal_strength
                    ask_volume = min(
                        self.dt_default_vol,
                        remaining_sell,
                    )
                else:
                    signal_strength = 0
                    ask_volume = min(remaining_sell, position)

                if remaining_sell > 0 and percentage_diff > signal_strength:
                    best_ask_price, best_ask_volume = self.order_book.get_best_bid()
                    ask_price = best_ask_price
                    ask_volume = min(ask_volume, best_ask_volume)
                    ask_order = Order(self.symbol, ask_price, -ask_volume)
                    orders.append(ask_order)
                    remaining_sell -= ask_volume
                    position -= ask_volume

        return orders

    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.order_book.reset(order_depths)

        orders = []

        self.logger.print_numeric("position", position)
        remaining_buy = self.pos_limit - position
        remaining_sell = self.pos_limit + position

        # --------------Price estimation------------------
        mm_price = self.order_book.get_mm_fair(self.detect_mm_volume)
        self.logger.print_numeric("mm_price", mm_price)
        vwap = self.order_book.vwap
        self.logger.print_numeric("vwap", vwap)

        if mm_price is None:
            current_price = self.order_book.vwap
        else:
            current_price = mm_price

        self.logger.print_numeric("current_price", current_price)

        self.short_history.append(current_price)
        self.long_history.append(current_price)
        # ------------------------------------------------

        directional_orders = self.directional_trade(
            position, remaining_buy, remaining_sell
        )
        orders += directional_orders

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
    # Market making parameters
    "mm_default_vol": 20,
}

config_squid = {
    "update_order_book": True,
    "detect_mm_volume": 15,  # Volume to detect market maker
    # Price estimation
    "short_window": 80,
    "long_window": 410,
    # Directional parameters
    "dt_default_vol": 5,
    "dt_signal_strength": 0.0015,
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
            products["KELP"] = Kelp(config_kelp)
            products["SQUID_INK"] = Squid(config_squid)
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
            self.logs += f"{label} {value:.5f}"
        else:
            self.logs += f"{label} {value}"

        self.logs += end

    def flush(self):
        print(self.logs)
        self.logs = ""