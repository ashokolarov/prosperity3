from abc import ABC, abstractmethod

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
        self.logger.print(f"timestamp {timestamp}")

    def print_product_end(self):
        self.logger.print(f"PRODUCT_E {self.symbol}")
        self.logger.flush()

    def print_orders(self, orders):
        for order in orders:
            self.logger.print(f"order {order.quantity}@{order.price}")

    @abstractmethod
    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        pass


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

        # Market taking parameters
        self.mt_bid_edge = config.get("mt_bid_edge")
        self.mt_ask_edge = config.get("mt_ask_edge")
        self.mt_long_profit_margin = config.get("mt_long_pm")
        self.mt_short_profit_margin = config.get("mt_short_pm")

        # Market making parameters
        self.mm_default_vol = config.get("mm_default_vol")
        self.mm_max_position_factor = config.get("max_position_factor")

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
            else:
                break  # If even the best ask doesn't cross the mean, then no need to check further

        for bid_order in bid_orders:
            self.order_book.update(bid_order)

        # Check if there is an opportunity to market take in bid orders
        ask_orders = []
        for depth_level in range(self.order_book.bid_orders_depth):
            bid_price, bid_volume = self.order_book.get_bid_order_at_depth(depth_level)
            if bid_price - self.mean >= self.mt_bid_edge:
                ask_price = bid_price
                ask_volume = min(remaining_sell, bid_volume)
                ask_order = Order(self.symbol, ask_price, -ask_volume)
                ask_orders.append(ask_order)
            else:
                break  # If even the best bid doesn't cross the mean, then no need to check further

        for ask_order in ask_orders:
            self.order_book.update(ask_order)

        return bid_orders + ask_orders

    def liquidate_mt_orders(self, position):
        close_long = []
        updated_position = position
        for depth_level in range(self.order_book.bid_orders_depth):
            if updated_position > 0:
                bid_price, bid_volume = self.order_book.get_bid_order_at_depth(
                    depth_level
                )
                if bid_price - self.mean >= self.mt_long_profit_margin:
                    qty = min(position, bid_volume)
                    ask_order = Order(self.symbol, bid_price, -qty)
                    close_long.append(ask_order)
                    updated_position -= qty
                else:
                    break
            else:
                break

        for ask_order in close_long:
            self.order_book.update(ask_order)

        close_short = []
        updated_position = position
        for depth_level in range(self.order_book.ask_orders_depth):
            if updated_position < 0:
                ask_price, ask_volume = self.order_book.get_ask_order_at_depth(
                    depth_level
                )
                if self.mean - ask_price >= self.mt_short_profit_margin:
                    qty = min(abs(position), ask_volume)
                    bid_order = Order(self.symbol, ask_price, qty)
                    close_short.append(bid_order)
                    updated_position += qty
                else:
                    break
            else:
                break

        for bid_order in close_short:
            self.order_book.update(bid_order)

        return close_long + close_short

    def market_make(self, positions):
        orders = []

        # Get current market state
        spread = self.order_book.spread
        mid_price = int(self.order_book.mid_price)
        imbalance = self.order_book.calculate_order_book_imbalance()

        position_ratio = positions["position"] / self.pos_limit

        price_skew = 0
        # If we have strong buying pressure, shift our quotes higher
        if imbalance > 1.5:
            price_skew = 1
        # If we have strong selling pressure, shift our quotes lower
        elif imbalance < 0.5:
            price_skew = -1

        # Calculate our bid and ask prices
        half_spread = spread // 2
        bid_price = mid_price - half_spread + 1 + price_skew
        ask_price = mid_price + half_spread - 1 + price_skew

        # Make sure our prices are sensible relative to the mean value
        if bid_price > self.mean:
            bid_price = int(self.mean)
        if ask_price < self.mean:
            ask_price = int(self.mean)

        buy_volume_factor = max(0, 1 - (position_ratio / self.mm_max_position_factor))
        sell_volume_factor = max(0, 1 + (position_ratio / self.mm_max_position_factor))

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

    def calculate_delta_by_direction(self, orders):
        positive_delta = sum(order.quantity for order in orders if order.quantity > 0)
        negative_delta = sum(order.quantity for order in orders if order.quantity < 0)

        return positive_delta, negative_delta

    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.order_book.reset(order_depths)

        orders = []

        self.logger.print(f"position {position}")

        liquidated_orders = self.liquidate_mt_orders(position)
        orders += liquidated_orders
        buy_vol_l, sell_vol_l = self.calculate_delta_by_direction(orders)

        remaining_buy = self.pos_limit - (position + buy_vol_l)
        remaining_sell = self.pos_limit + (position + sell_vol_l)

        mt_orders = self.market_take(remaining_buy, remaining_sell)
        orders += mt_orders
        buy_vol_t, sell_vol_t = self.calculate_delta_by_direction(mt_orders)

        remaining_buy -= buy_vol_t
        remaining_sell -= sell_vol_t

        positions = {
            "remaining_buy": remaining_buy,
            "remaining_sell": remaining_sell,
            "position": position,
        }

        mm_orders = self.market_make(positions)
        orders += mm_orders

        self.print_orders(orders)
        self.print_product_end()

        return orders


class Kelp(Product):
    def __init__(self, config):
        super().__init__(config)

        # Kelp parameters
        self.name = "Kelp"
        self.symbol = "KELP"
        self.pos_limit = 50

        # Order book
        self.order_book = OrderBook()

        self.mm_default_vol = config.get("mm_default_vol")

    def market_take(self, remaining_buy, remaining_sell):
        # Check if there is an opportunity to market take in ask orders
        bid_orders = []
        fair_price = int(self.order_book.mm_fair_price)

        for depth_level in range(self.order_book.ask_orders_depth):
            ask_price, ask_volume = self.order_book.get_ask_order_at_depth(depth_level)
            if fair_price >= ask_price:
                bid_price = ask_price
                bid_volume = min(remaining_buy, ask_volume)
                bid_order = Order(self.symbol, bid_price, bid_volume)
                bid_orders.append(bid_order)
            else:
                break  # If even the best ask doesn't cross the mean, then no need to check further

        for bid_order in bid_orders:
            self.order_book.update(bid_order)

        # Check if there is an opportunity to market take in bid orders
        ask_orders = []
        for depth_level in range(self.order_book.bid_orders_depth):
            bid_price, bid_volume = self.order_book.get_bid_order_at_depth(depth_level)
            if bid_price >= fair_price:
                ask_price = bid_price
                ask_volume = min(remaining_sell, bid_volume)
                ask_order = Order(self.symbol, ask_price, -ask_volume)
                ask_orders.append(ask_order)
            else:
                break  # If even the best bid doesn't cross the mean, then no need to check further

        for ask_order in ask_orders:
            self.order_book.update(ask_order)

        return bid_orders + ask_orders

    def liquidate_mt_orders(self, position):
        close_long = []
        updated_position = position

        fair_price = int(self.order_book.mm_fair_price)
        for depth_level in range(self.order_book.bid_orders_depth):
            if updated_position > 0:
                bid_price, bid_volume = self.order_book.get_bid_order_at_depth(
                    depth_level
                )
                if bid_price >= fair_price:
                    qty = min(position, bid_volume)
                    ask_order = Order(self.symbol, bid_price, -qty)
                    close_long.append(ask_order)
                    updated_position -= qty
                else:
                    break
            else:
                break

        for ask_order in close_long:
            self.order_book.update(ask_order)

        close_short = []
        updated_position = position
        for depth_level in range(self.order_book.ask_orders_depth):
            if updated_position < 0:
                ask_price, ask_volume = self.order_book.get_ask_order_at_depth(
                    depth_level
                )
                if ask_price <= fair_price:
                    qty = min(abs(position), ask_volume)
                    bid_order = Order(self.symbol, ask_price, qty)
                    close_short.append(bid_order)
                    updated_position += qty
                else:
                    break
            else:
                break

        for bid_order in close_short:
            self.order_book.update(bid_order)

        return close_long + close_short

    def market_make(self, positions):
        orders = []

        # Get current market state
        spread = self.order_book.spread
        fair_price = self.order_book.mm_fair_price

        # Calculate our bid and ask prices
        half_spread = spread / 2
        bid_price = fair_price - half_spread
        ask_price = fair_price + half_spread

        bid_price = int(bid_price)
        ask_price = int(ask_price)

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

    def calculate_delta_by_direction(self, orders):
        positive_delta = sum(order.quantity for order in orders if order.quantity > 0)
        negative_delta = sum(order.quantity for order in orders if order.quantity < 0)

        return positive_delta, negative_delta

    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin(timestamp)
        self.order_book.reset(order_depths)

        orders = []

        self.logger.print(f"position {position}")

        liquidated_orders = self.liquidate_mt_orders(position)
        orders += liquidated_orders
        buy_vol_l, sell_vol_l = self.calculate_delta_by_direction(orders)

        remaining_buy = self.pos_limit - (position + buy_vol_l)
        remaining_sell = self.pos_limit + (position + sell_vol_l)

        mt_orders = self.market_take(remaining_buy, remaining_sell)
        orders += mt_orders
        buy_vol_t, sell_vol_t = self.calculate_delta_by_direction(mt_orders)

        remaining_buy -= buy_vol_t
        remaining_sell -= sell_vol_t

        positions = {
            "remaining_buy": remaining_buy,
            "remaining_sell": remaining_sell,
            "position": position,
        }

        mm_orders = self.market_make(positions)
        orders += mm_orders

        self.print_orders(orders)
        self.print_product_end()

        return orders
