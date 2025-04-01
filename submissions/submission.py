# Combined Python Files
# Files combined: market_utils.py, products.py, trader.py, utils.py

# Import statements
from abc import ABC, abstractmethod

from collections import deque

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


class PositionBook:
    def __init__(self):
        self.long_pos = {"price": 0, "quantity": 0}
        self.short_pos = {"price": 0, "quantity": 0}

    @property
    def tot_position(self):
        return self.long_pos["quantity"] - self.short_pos["quantity"]

    def add_pos(self, order):
        price = order.price
        qty = order.quantity

        # Update positions based on order side
        if qty > 0:  # Long position (buying)
            if self.long_pos["quantity"] == 0:
                new_price = price
            else:
                # Calculate weighted average price
                new_price = (
                    self.long_pos["price"] * self.long_pos["quantity"] + price * qty
                ) / (self.long_pos["quantity"] + qty)

            self.long_pos["price"] = new_price
            self.long_pos["quantity"] += qty

        else:  # Short position (selling)
            abs_qty = abs(qty)
            if self.short_pos["quantity"] == 0:
                new_price = price
            else:
                # Calculate weighted average price
                new_price = (
                    self.short_pos["price"] * self.short_pos["quantity"]
                    + price * abs_qty
                ) / (self.short_pos["quantity"] + abs_qty)

            self.short_pos["price"] = new_price
            self.short_pos["quantity"] += abs_qty

    def remove_pos(self, order):
        price = order.price
        qty = order.quantity

        # Update positions based on order side
        if qty > 0:  # Buying - reduces short position
            abs_qty = qty
            if self.short_pos["quantity"] > 0:
                # Calculate P&L for this liquidation
                pnl = (self.short_pos["price"] - price) * abs_qty

                # Update short position
                self.short_pos["quantity"] -= abs_qty

                # Reset price if position is fully liquidated
                if self.short_pos["quantity"] == 0:
                    self.short_pos["price"] = 0

                return pnl
            return 0  # No short position to liquidate
        else:  # Selling - reduces long position
            abs_qty = abs(qty)
            if self.long_pos["quantity"] > 0:
                # Calculate P&L for this liquidation
                pnl = (price - self.long_pos["price"]) * abs_qty

                # Update long position
                self.long_pos["quantity"] -= abs_qty

                # Reset price if position is fully liquidated
                if self.long_pos["quantity"] == 0:
                    self.long_pos["price"] = 0

                return pnl
            return 0  # No long position to liquidate

    def __repr__(self):
        lines = ["POSITION SUMMARY:\n"]

        if self.long_pos["quantity"] > 0:
            lines.append(
                f"LONG: {self.long_pos['quantity']} @ avg {self.long_pos['price']:.2f}\n"
            )

        if self.short_pos["quantity"] > 0:
            lines.append(
                f"SHORT: {self.short_pos['quantity']} @ avg {self.short_pos['price']:.2f}\n"
            )

        lines.append(
            f"NET POSITION: {self.long_pos['quantity'] - self.short_pos['quantity']}\n"
        )

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
            "position": position + delta,
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

# Code from trader.py
config_rainforest = {
    # Market taking parameters
    "mt_bid_edge": 1,
    "mt_ask_edge": 1,
    "mt_short_pm": 3,
    "mt_long_pm": 3,
    # Market making parameters
    "mm_default_vol": 15,
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

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self):
        print(self.logs)
        self.logs = ""