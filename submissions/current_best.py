# Combined Python Files
# Files combined: market_utils.py, products.py, trader.py, utils.py

# Import statements
from abc import ABC, abstractmethod

from datamodel import Order

from datamodel import TradingState

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

    def reset(self, order_depths):
        sell_orders = order_depths.sell_orders
        buy_orders = order_depths.buy_orders

        sell_orders = list(sell_orders.items())
        self.ask_prices = [order[0] for order in sell_orders]
        self.ask_volumes = [abs(order[1]) for order in sell_orders]

        buy_orders = list(buy_orders.items())
        self.bid_prices = [order[0] for order in buy_orders]
        self.bid_volumes = [order[1] for order in buy_orders]

    def get_ask_order_at_depth(self, depth):
        assert depth < self.ask_orders_depth and depth >= 0
        return self.ask_prices[depth], self.ask_volumes[depth]

    def get_bid_order_at_depth(self, depth):
        assert depth < self.bid_orders_depth and depth >= 0
        return self.bid_prices[depth], self.bid_volumes[depth]

    @property
    def bid_orders_depth(self):
        return len(self.bid_prices)

    @property
    def ask_orders_depth(self):
        return len(self.ask_prices)

    @property
    def best_ask(self):
        return self.ask_prices[0], self.ask_volumes[0]

    @property
    def best_bid(self):
        return self.bid_prices[0], self.bid_volumes[0]

    @property
    def spread(self):
        return self.ask_prices[0] - self.bid_prices[0]

    @property
    def mid_price(self):
        return (self.ask_prices[0] + self.bid_prices[0]) / 2

    @property
    def vwap(self):
        bid_vwap = sum(
            [price * volume for price, volume in zip(self.bid_prices, self.bid_volumes)]
        ) / sum(self.bid_volumes)
        ask_vwap = sum(
            [price * volume for price, volume in zip(self.ask_prices, self.ask_volumes)]
        ) / sum(self.ask_volumes)

        vwap = (bid_vwap + ask_vwap) / 2

        return vwap

    def calculate_order_book_imbalance(self):
        """Calculate the order book imbalance ratio."""
        total_bid_volume = sum(self.bid_volumes)
        total_ask_volume = sum(self.ask_volumes)

        # Avoid division by zero
        if total_ask_volume == 0:
            return float("inf")  # Extreme buying pressure
        elif total_bid_volume == 0:
            return 0.0  # Extreme selling pressure

        return total_bid_volume / total_ask_volume

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
        imbalance = self.calculate_order_book_imbalance()
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

# Code from trader.py
config_rainforest = {
    # Market taking parameters
    "mt_bid_edge": 1,
    "mt_ask_edge": 1,
    "mt_short_pm": 0,
    "mt_long_pm": 0,
    # Market making parameters
    "mm_default_vol": 15,
    "max_position_factor": 0.8,  # Maximum position as a factor of position limit
}

config_kelp = {
    "mm_default_vol": 10,
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

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self):
        print(self.logs)
        self.logs = ""