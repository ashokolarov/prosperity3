from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from datamodel import Order
from market_utils import OrderBook
from utils import CustomLogger


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

    def market_make_2(self, fair_value, position, remaining_buy, remaining_sell):
        orders = []

        disregard_edge = 1
        default_edge = 1
        join_edge = 1

        asks_above_fair = [
            price
            for price in self.order_book.ask_prices
            if price > round(fair_value + disregard_edge)
        ]
        bids_below_fair = [
            price
            for price in self.order_book.bid_prices
            if price < round(fair_value - disregard_edge)
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask_price = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            best_ask_idx = self.order_book.ask_prices.index(best_ask_above_fair)
            best_ask_volume = self.order_book.ask_volumes[best_ask_idx]
            if abs(best_ask_above_fair - fair_value) <= join_edge:  # best ask volume 1
                ask_price = best_ask_above_fair
            else:
                ask_price = best_ask_above_fair - 1  #

        bid_price = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            best_bid_idx = self.order_book.bid_prices.index(best_bid_below_fair)
            best_bid_volume = self.order_book.bid_volumes[best_bid_idx]
            if abs(fair_value - best_bid_below_fair) <= join_edge:  # best bid volume 3
                bid_price = best_bid_below_fair  # join BEST 0
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

    def market_make_3(self, fair_value, position, remaining_buy, remaining_sell):
        orders = []

        mm_package = self.order_book.get_mm_fair(
            self.detect_mm_volume, with_spread=True
        )
        if mm_package is None:
            return orders
        else:
            mm_price = mm_package[0]
            mm_spread = mm_package[1]

        bid_price = round(mm_price - mm_spread / 2 + 1)
        ask_price = round(mm_price + mm_spread / 2 - 1)

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

    def directional_trade(self, remaining_buy, remaining_sell):
        orders = []

        if (len(self.dt_short_history) < self.dt_short_size) or (
            len(self.dt_long_history) < self.dt_long_size
        ):
            return []

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

        if mm_price is None:
            current_price = vwap
        else:
            current_price = mm_price

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
        orders += self.market_make_3(
            fair_value, position, remaining_buy, remaining_sell
        )

        self.print_orders(orders)
        self.print_product_end()

        return orders


# ------------------KELP-------------------#
class Squid(Product):
    def __init__(self, config):
        super().__init__(config)

        # Kelp parameters
        self.name = "Squid Ink"
        self.symbol = "SQUID_INK"
        self.pos_limit = 50

        # Order book
        self.update_order_book = config.get("update_order_book")
        self.order_book = OrderBook()

    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.order_book.reset(order_depths)

        orders = []

        self.logger.print_numeric("position", position)
        remaining_buy = self.pos_limit - position
        remaining_sell = self.pos_limit + position

        self.print_orders(orders)
        self.print_product_end()

        return orders
