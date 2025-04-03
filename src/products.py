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

        # Calculate our bid and ask prices
        half_spread = spread // 2
        bid_price = mid_price - half_spread + 1
        ask_price = mid_price + half_spread - 1

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
