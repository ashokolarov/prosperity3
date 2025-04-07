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

    def get_mm_fair(self, adverse_volume):
        if self.ask_orders_depth == 0 or self.bid_orders_depth == 0:
            return None
        else:
            if (
                max(self.ask_volumes) >= adverse_volume
                and max(self.bid_volumes) >= adverse_volume
            ):
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
        self.update_order_book = config.get("update_order_book")

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
                    qty = min(remaining_sell, bid_volume)
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
                    qty = min(remaining_buy, ask_volume)
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

        consider_bid = None
        for depth_level in range(self.order_book.bid_orders_depth):
            bid_price, bid_volume = self.order_book.get_bid_order_at_depth(depth_level)
            if bid_price >= fair_value - self.mm_disregard_edge:
                continue
            elif (
                bid_price >= fair_value - self.mm_join_edge
                and bid_volume <= self.mm_join_volume
            ):
                consider_bid = bid_price
                break
            else:
                consider_bid = bid_price + 1
                break

        consider_ask = None
        for depth_level in range(self.order_book.ask_orders_depth):
            ask_price, ask_volume = self.order_book.get_ask_order_at_depth(depth_level)
            if ask_price <= fair_value + self.mm_disregard_edge:
                continue
            elif (
                ask_price <= fair_value + self.mm_join_edge
                and ask_volume <= self.mm_join_volume
            ):
                consider_ask = ask_price
                break
            else:
                consider_ask = ask_price - 1
                break

        spread = consider_ask - consider_bid
        mid_price = (consider_ask + consider_bid) / 2

        # Calculate our bid and ask prices
        half_spread = spread / 2
        bid_price = mid_price - half_spread + 1
        ask_price = mid_price + half_spread - 1

        bid_price = round(bid_price)
        ask_price = round(ask_price)

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

    def market_make_new(self, fair_value, position, remaining_buy, remaining_sell):
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

    def market_make_luf(self, fair_value, position, remaining_buy, remaining_sell):
        orders = []

        ask_above_fair = [
            price
            for price in self.order_book.ask_prices
            if price - fair_value > self.mm_disregard_edge
        ]
        bid_below_fair = [
            price
            for price in self.order_book.bid_prices
            if self.mm_disregard_edge < fair_value - price
        ]

        baaf = min(ask_above_fair) if len(ask_above_fair) > 0 else None
        bbbf = max(bid_below_fair) if len(bid_below_fair) > 0 else None

        if baaf is not None:
            baaf_idx = self.order_book.ask_prices.index(baaf)
            baaf_volume = self.order_book.ask_volumes[baaf_idx]
            if baaf - fair_value <= self.mm_join_edge:
                baaf = fair_value + 3

        if bbbf is not None:
            bbbf_idx = self.order_book.bid_prices.index(bbbf)
            bbbf_volume = self.order_book.bid_volumes[bbbf_idx]
            if fair_value - bbbf <= self.mm_join_edge:
                bbbf = fair_value - 3

        bid_price = bbbf + 1
        ask_price = baaf - 1

        bid_price = round(bid_price)
        ask_price = round(ask_price)

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

        # Reset order book, track positions and prices
        self.order_book.reset(order_depths)
        orders = []

        self.logger.print_numeric("position", position)

        fair_value = self.mean
        remaining_buy = self.pos_limit - position
        remaining_sell = self.pos_limit + position
        # ------------------------------------------------
        # Liquidation and market taking
        liquidated_orders, position, remaining_buy, remaining_sell = (
            self.liquidate_position(fair_value, position, remaining_buy, remaining_sell)
        )
        orders += liquidated_orders

        mt_orders, position, remaining_buy, remaining_sell = self.market_take(
            fair_value, position, remaining_buy, remaining_sell
        )
        orders += mt_orders
        # ------------------------------------------------
        # Market making
        mm_orders = self.market_make_new(
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

        # General
        self.price_window = config.get("price_window")
        self.price_history = deque(maxlen=self.price_window)
        self.adverse_volume = config.get("adverse_volume")

        self.last_fair_price = None

        # Market taking parameters
        self.mt_take_width = config.get("mt_take_width")
        self.mt_clear_width = config.get("mt_clear_width")
        self.mt_prevent_adverse = config.get("mt_prevent_adverse")
        self.mt_reversion_beta = config.get("mt_reversion_beta")

        # Market making parameters
        self.mm_disregard_edge = config.get("mm_disregard_edge")
        self.mm_default_vol = config.get("mm_default_vol")

    def estimate_fair_value(self):
        if (
            len(self.order_book.ask_prices) != 0
            and len(self.order_book.bid_prices) != 0
        ):
            best_ask = min(self.order_book.ask_prices)
            best_bid = max(self.order_book.bid_prices)
            filtered_ask = [
                self.order_book.ask_prices[idx]
                for idx in range(self.order_book.ask_orders_depth)
                if abs(self.order_book.ask_volumes[idx]) >= self.adverse_volume
            ]
            filtered_bid = [
                self.order_book.bid_prices[idx]
                for idx in range(self.order_book.bid_orders_depth)
                if abs(self.order_book.bid_volumes[idx]) >= self.adverse_volume
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask is None or mm_bid is None:
                if self.last_fair_price is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = self.last_fair_price
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if self.last_fair_price is not None:
                last_price = self.last_fair_price
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.mt_reversion_beta
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            self.last_fair_price = fair
            return fair
        return None

    def market_take(self, fair_value, position, remaining_buy, remaining_sell):
        orders = []

        # Check if there is an opportunity to market take in ask orders
        for depth_level in range(self.order_book.ask_orders_depth):
            ask_price, ask_volume = self.order_book.get_ask_order_at_depth(depth_level)
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
                if bid_volume <= self.adverse_volume:
                    if bid_price - fair_value >= self.mt_clear_width:
                        qty = min(remaining_sell, bid_volume)
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
                if ask_volume <= self.adverse_volume:
                    if fair_value - ask_price >= self.mt_clear_width:
                        qty = min(remaining_buy, ask_volume)
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
        aaf = [
            price
            for price in self.order_book.ask_prices
            if price >= round(fair_value + self.mm_disregard_edge)
        ]
        bbf = [
            price
            for price in self.order_book.bid_prices
            if price <= round(fair_value - self.mm_disregard_edge)
        ]
        baaf = min(aaf) if len(aaf) > 0 else round(fair_value + self.mm_disregard_edge)
        bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - self.mm_disregard_edge)

        bid_price = bbbf + 1
        ask_price = baaf - 1

        bid_price = round(bid_price)
        ask_price = round(ask_price)

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

        self.logger.print(f"position {position}")
        remaining_buy = self.pos_limit - position
        remaining_sell = self.pos_limit + position

        fair_value = self.estimate_fair_value()
        self.logger.print(f"fair_value {fair_value}")

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
        orders += self.market_make(fair_value, position, remaining_buy, remaining_sell)

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
    "price_window": 10,
    "adverse_volume": 10,  # Market taking parameters
    "n_points": 3,
    "p_degree": 1,
    # Market taking parameters
    "mt_take_width": 1,
    "mt_clear_width": 0,
    "mt_reversion_beta": -0.14,
    # Market making parameters
    "mm_default_vol": 15,
    "mm_disregard_edge": 2,
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