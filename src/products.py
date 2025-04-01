from abc import ABC, abstractmethod
from collections import deque

from datamodel import Order
from market_utils import OrderBook, PositionBook
from utils import CustomLogger


class Product(ABC):
    name: str = None
    symbol: str = None
    pos_limit: int = None

    def __init__(self, config):
        self.logger = CustomLogger()

    def print_product_begin(self, timestamp):
        self.logger.print(f"PRODUCT_B {self.symbol}")
        self.logger.print(f"timestamp {timestamp}")

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

        # Price tracking
        self.history_size = config.get("history_size")
        self.history = deque(maxlen=self.history_size)
        self.volatility = 1.0

        # Market taking parameters
        self.mt_positions = PositionBook()
        self.mt_bid_edge = config.get("mt_bid_edge")
        self.mt_ask_edge = config.get("mt_ask_edge")
        self.mt_long_profit_margin = config.get("mt_long_pm")
        self.mt_short_profit_margin = config.get("mt_short_pm")

        # Market making parameters
        self.mm_default_vol = config.get("mm_default_vol")

    def market_take(self, remaining_buy, remaining_sell):
        # Check if there is an opportunity to market take in ask orders
        bid_orders = []
        for depth_level in range(self.order_book.ask_orders_depth):
            ask_price, ask_volume = self.order_book.get_ask_order_at_depth(depth_level)
            if self.mean - ask_price >= self.mt_ask_edge:
                bid_price = ask_price
                bid_volume = min(remaining_buy, ask_volume)
                bid_order = Order(self.symbol, bid_price, bid_volume)
                bid_orders.append(bid_order)
                remaining_buy -= bid_volume
            else:
                break  # If even the best ask doesn't cross the mean, then no need to check further

        for bid_order in bid_orders:
            self.order_book.update(bid_order)
            self.mt_positions.add_pos(bid_order)

        # Check if there is an opportunity to market take in bid orders
        ask_orders = []
        for depth_level in range(self.order_book.bid_orders_depth):
            bid_price, bid_volume = self.order_book.get_bid_order_at_depth(depth_level)
            if bid_price - self.mean >= self.mt_bid_edge:
                ask_price = bid_price
                ask_volume = min(remaining_sell, bid_volume)
                ask_order = Order(self.symbol, ask_price, -ask_volume)
                ask_orders.append(ask_order)
                remaining_sell -= ask_volume
            else:
                break  # If even the best bid doesn't cross the mean, then no need to check further

        for ask_order in ask_orders:
            self.order_book.update(ask_order)
            self.mt_positions.add_pos(ask_order)

        return bid_orders + ask_orders, remaining_buy, remaining_sell

    def liquidate_mt_orders(self, remaining_buy, remaining_sell):
        # Check if there is an opportunity to liquidate long positions
        close_long = []
        long_pos = self.mt_positions.long_pos["quantity"]
        for depth_level in range(self.order_book.bid_orders_depth):
            if long_pos > 0:
                bid_price, bid_volume = self.order_book.get_bid_order_at_depth(
                    depth_level
                )
                if (
                    bid_price - self.mt_positions.long_pos["price"]
                    >= self.mt_long_profit_margin
                ):
                    qty = min(remaining_sell, bid_volume, long_pos)
                    ask_order = Order(self.symbol, bid_price, -qty)
                    close_long.append(ask_order)
                    remaining_sell -= qty
                    long_pos -= qty
                else:
                    break
            else:
                break

        for ask_order in close_long:
            self.order_book.update(ask_order)
            self.mt_positions.remove_pos(ask_order)

        # Check if there is an opportunity to liquidate short positions
        close_short = []
        short_pos = self.mt_positions.short_pos["quantity"]
        for depth_level in range(self.order_book.ask_orders_depth):
            if short_pos > 0:
                ask_price, ask_volume = self.order_book.get_ask_order_at_depth(
                    depth_level
                )
                if (
                    self.mt_positions.short_pos["price"] - ask_price
                    >= self.mt_short_profit_margin
                ):
                    qty = min(remaining_buy, ask_volume, short_pos)
                    bid_order = Order(self.symbol, ask_price, qty)
                    close_short.append(bid_order)
                    remaining_buy -= qty
                    short_pos -= qty
                else:
                    break
            else:
                break

        for bid_order in close_short:
            self.order_book.update(bid_order)
            self.mt_positions.remove_pos(bid_order)

        return close_long + close_short, remaining_buy, remaining_sell

    def market_make(self, positions):
        orders = []

        # Get current market state
        spread = self.order_book.spread
        mid_price = int(self.order_book.mid_price)

        imbalance = self.order_book.imbalance
        imbalance_skew = 0
        # If we have strong buying pressure, shift our quotes higher
        if imbalance > 1.5:
            imbalance_skew = 1
        # If we have strong selling pressure, shift our quotes lower
        elif imbalance < 0.5:
            imbalance_skew = -1

        # Calculate our bid and ask prices
        half_spread = spread // 2
        bid_price = mid_price - half_spread + 1 + imbalance_skew
        ask_price = mid_price + half_spread - 1 + imbalance_skew

        if bid_price > self.mean:
            bid_price = int(self.mean)
        if ask_price < self.mean:
            ask_price = int(self.mean)

        # Scale our order sizes based on how far we are from position limits
        bid_volume = min(self.mm_default_vol, positions["remaining_buy"])
        ask_volume = min(self.mm_default_vol, positions["remaining_sell"])

        # Create the orders if they make sense
        if bid_price > 0 and bid_volume > 0:
            bid_order = Order(self.symbol, bid_price, bid_volume)
            orders.append(bid_order)
            self.order_book.update(bid_order)

        if ask_price > 0 and ask_volume > 0:
            ask_order = Order(self.symbol, ask_price, -ask_volume)
            orders.append(ask_order)
            self.order_book.update(ask_order)

        return orders

    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)

        self.order_book.reset(order_depths)
        orders = []

        mt_position = self.mt_positions.tot_position
        mm_position = position - mt_position
        self.logger.print(f"position {position}")
        self.logger.print(f"mt_position {mt_position}")
        self.logger.print(f"mm_position {mm_position}")

        remaining_buy = self.pos_limit - position
        remaining_sell = self.pos_limit + position

        liquidated_orders, remaining_buy, remaining_sell = self.liquidate_mt_orders(
            remaining_buy, remaining_sell
        )
        orders += liquidated_orders

        mt_orders, remaining_buy, remaining_sell = self.market_take(
            remaining_buy, remaining_sell
        )
        orders += mt_orders

        delta, _, _ = self.calculate_position_delta(orders)
        positions = {
            "remaining_buy": remaining_buy,
            "remaining_sell": remaining_sell,
            "mm_position": mm_position,
        }

        mm_orders = self.market_make(positions)
        orders += mm_orders

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
