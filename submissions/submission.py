# Combined Python Files
# Files combined: datamodel.py, market_utils.py, products.py, trader.py, utils.py

# Import statements
from abc import ABC, abstractmethod

from datamodel import Order

from datamodel import TradingState

from json import JSONEncoder

from market_utils import OrderBook, PositionBook

from products import RainforestResin

from time import time

from typing import Any

from typing import Dict, List

from utils import CustomLogger

import json

import jsonpickle



# Code from datamodel.py
Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int


class Listing:
    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class ConversionObservation:
    def __init__(
        self,
        bidPrice: float,
        askPrice: float,
        transportFees: float,
        exportTariff: float,
        importTariff: float,
        sugarPrice: float,
        sunlightIndex: float,
    ):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sugarPrice = sugarPrice
        self.sunlightIndex = sunlightIndex


class Observation:
    def __init__(
        self,
        plainValueObservations: Dict[Product, ObservationValue],
        conversionObservations: Dict[Product, ConversionObservation],
    ) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations

    def __str__(self) -> str:
        return (
            "(plainValueObservations: "
            + jsonpickle.encode(self.plainValueObservations)
            + ", conversionObservations: "
            + jsonpickle.encode(self.conversionObservations)
            + ")"
        )


class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return (
            "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
        )

    def __repr__(self) -> str:
        return (
            "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
        )


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    def __init__(
        self,
        symbol: Symbol,
        price: int,
        quantity: int,
        buyer: UserId = None,
        seller: UserId = None,
        timestamp: int = 0,
    ) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return (
            "("
            + self.symbol
            + ", "
            + self.buyer
            + " << "
            + self.seller
            + ", "
            + str(self.price)
            + ", "
            + str(self.quantity)
            + ", "
            + str(self.timestamp)
            + ")"
        )

    def __repr__(self) -> str:
        return (
            "("
            + self.symbol
            + ", "
            + self.buyer
            + " << "
            + self.seller
            + ", "
            + str(self.price)
            + ", "
            + str(self.quantity)
            + ", "
            + str(self.timestamp)
            + ")"
        )


class TradingState(object):
    def __init__(
        self,
        traderData: str,
        timestamp: Time,
        listings: Dict[Symbol, Listing],
        order_depths: Dict[Symbol, OrderDepth],
        own_trades: Dict[Symbol, List[Trade]],
        market_trades: Dict[Symbol, List[Trade]],
        position: Dict[Product, Position],
        observations: Observation,
    ):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


class ProsperityEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

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

        vwap = sum(
            [price * volume for price, volume in zip(self.ask_prices, self.ask_volumes)]
        ) + sum(
            [price * volume for price, volume in zip(self.bid_prices, self.bid_volumes)]
        ) / (
            sum(self.ask_volumes) + sum(self.bid_volumes)
        )

        return (
            vwap,
            bid_vwap,
            ask_vwap,
        )

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
        lines.append(f"VWAP_BID: {vwap[0]}, VWAP_ASK: {vwap[1]}, VWAP_MID: {vwap[2]}\n")
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
        self.long_pos = 0
        self.long_price = 0
        self.short_pos = 0
        self.short_price = 0
        self.tot_position = 0

    def get_long_position(self):
        return self.long_price, self.long_pos

    def get_short_position(self):
        return self.short_price, self.short_pos

    def add_pos(self, order):
        price = order.price
        qty = order.quantity

        # Update positions based on order side
        if qty > 0:  # Long position (buying)
            if self.long_pos == 0:
                self.long_price = price
            else:
                # Calculate weighted average price
                self.long_price = (self.long_price * self.long_pos + price * qty) / (
                    self.long_pos + qty
                )
            self.long_pos += qty
        else:  # Short position (selling)
            abs_qty = abs(qty)
            if self.short_pos == 0:
                self.short_price = price
            else:
                # Calculate weighted average price
                self.short_price = (
                    self.short_price * self.short_pos + price * abs_qty
                ) / (self.short_pos + abs_qty)

            self.short_pos += abs_qty

        # Update total position
        self.tot_position = self.long_pos - self.short_pos

    def liquidate_pos(self, qty):
        if qty > 0:  # Buying to cover shorts
            self.short_pos -= qty

            # Reset average price if fully liquidated
            if self.short_pos == 0:
                self.short_price = 0
        else:  # Selling to close longs
            self.long_pos -= abs(qty)

            # Reset average price if fully liquidated
            if self.long_pos == 0:
                self.long_price = 0

        # Update total position
        self.tot_position = self.long_pos - self.short_pos

    def __repr__(self):
        lines = ["POSITION SUMMARY:\n"]

        if self.long_pos > 0:
            lines.append(f"LONG: {self.long_pos} @ avg {self.long_price:.2f}\n")

        if self.short_pos > 0:
            lines.append(f"SHORT: {self.short_pos} @ avg {self.short_price:.2f}\n")

        lines.append(f"NET POSITION: {self.tot_position}\n")

        return "".join(lines)

# Code from products.py
class Product(ABC):
    name: str = None
    symbol: str = None
    pos_limit: int = None

    def __init__(self, config):
        self.logger = CustomLogger()

    def print_product_begin(self, timestamp):
        self.logger.print(f"PRODUCT_BEGIN {self.symbol}")
        self.logger.print(f"timestamp {timestamp}")

    def print_product_end(self):
        self.logger.print(f"PRODUCT_END {self.symbol}")
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
        self.mt_positions = PositionBook()
        self.mt_pos_limit = config.get("mt_pos_limit")
        self.mt_hl_target = config.get("mt_hl_target")
        self.mt_bid_edge = config.get("mt_bid_edge")
        self.mt_ask_edge = config.get("mt_ask_edge")
        self.mt_short_profit_margin = config.get("mt_short_pm")
        self.mt_long_profit_margin = config.get("mt_long_pm")

        # Market making parameters
        self.mm_default_vol = config.get("mm_default_vol")

    def market_take(self):
        # Check if there is an opportunity to market take in ask orders
        position_delta = 0
        mt_bid_orders = []
        for depth_level in range(self.order_book.ask_orders_depth):
            ask_price, ask_volume = self.order_book.get_ask_order_at_depth(depth_level)
            if ask_price <= self.mean - self.mt_ask_edge:
                bid_price = ask_price
                bid_volume = ask_volume
                bid_order = Order(self.symbol, bid_price, bid_volume)
                mt_bid_orders.append(bid_order)
                position_delta += bid_volume
            else:
                break  # If even the best ask doesn't cross the mean, then no need to check further

        for bid_order in mt_bid_orders:
            self.order_book.update(bid_order)
            self.mt_positions.add_pos(bid_order)

        # Check if there is an opportunity to market take in bid orders
        mt_ask_orders = []
        for depth_level in range(self.order_book.bid_orders_depth):
            bid_price, bid_volume = self.order_book.get_bid_order_at_depth(depth_level)
            if bid_price >= self.mean + self.mt_bid_edge:
                ask_price = bid_price
                ask_volume = bid_volume
                ask_order = Order(self.symbol, ask_price, -ask_volume)
                mt_ask_orders.append(ask_order)
                position_delta -= ask_volume
            else:
                break  # If even the best bid doesn't cross the mean, then no need to check further

        for ask_order in mt_ask_orders:
            self.order_book.update(ask_order)
            self.mt_positions.add_pos(ask_order)

        return mt_ask_orders + mt_bid_orders, position_delta

    def liquidate_mt_orders(self):
        position_delta = 0

        close_long = []
        for depth_level in range(self.order_book.bid_orders_depth):
            if self.mt_positions.long_pos > 0:
                long_price, long_pos = self.mt_positions.get_long_position()
                bid_price, bid_volume = self.order_book.get_bid_order_at_depth(
                    depth_level
                )
                if bid_price - long_price >= self.mt_long_profit_margin:
                    qty = min(long_pos, bid_volume)
                    ask_order = Order(self.symbol, bid_price, -qty)
                    close_long.append(ask_order)
                    self.mt_positions.liquidate_pos(-qty)
                    position_delta -= qty
                else:
                    break
            else:
                break

        for ask_order in close_long:
            self.order_book.update(ask_order)

        close_short = []
        for depth_level in range(self.order_book.ask_orders_depth):
            if self.mt_positions.short_pos > 0:
                short_price, short_pos = self.mt_positions.get_short_position()
                ask_price, ask_volume = self.order_book.get_ask_order_at_depth(
                    depth_level
                )
                if short_price - ask_price >= self.mt_short_profit_margin:
                    qty = min(short_pos, ask_volume)
                    bid_order = Order(self.symbol, ask_price, qty)
                    close_short.append(bid_order)
                    self.mt_positions.liquidate_pos(qty)
                    position_delta += qty
                else:
                    break
            else:
                break

        for bid_order in close_short:
            self.order_book.update(bid_order)

        return close_long + close_short, position_delta

    def hard_liquidate(self):
        position_delta = 0

        liquidate_long = []

        # For long positions over the threshold
        if self.mt_positions.long_pos > 0:
            target_long = self.mt_hl_target
            sold_long = 0

            # Find the best available price to liquidate
            for depth_level in range(self.order_book.bid_orders_depth):
                bid_price, bid_volume = self.order_book.get_bid_order_at_depth(
                    depth_level
                )

                # Calculate how many shares to liquidate
                target_reduction = max(
                    0,
                    target_long - sold_long,
                )

                qty = min(bid_volume, target_reduction)

                if qty > 0:
                    ask_order = Order(self.symbol, bid_price, -qty)
                    liquidate_long.append(ask_order)
                    self.mt_positions.liquidate_pos(-qty)
                    position_delta -= qty

                    sold_long += qty

                    # If we've reduced enough, stop
                    if sold_long >= target_long:
                        break

        for sell_order in liquidate_long:
            self.order_book.update(sell_order)

        liquidate_short = []
        # For short positions over the threshold
        if self.mt_positions.short_pos > 0:
            target_short = self.mt_hl_target
            sold_short = 0
            # Find the best available price to liquidate
            for depth_level in range(self.order_book.ask_orders_depth):
                ask_price, ask_volume = self.order_book.get_ask_order_at_depth(
                    depth_level
                )

                # Calculate how many shares to liquidate
                target_reduction = max(
                    0,
                    target_short - sold_short,
                )
                qty = min(ask_volume, target_reduction)

                if qty > 0:
                    bid_order = Order(self.symbol, ask_price, qty)
                    liquidate_short.append(bid_order)
                    self.mt_positions.liquidate_pos(qty)
                    position_delta += qty

                    sold_short += qty

                    # If we've reduced enough, stop
                    if sold_short >= target_short:
                        break

        # Update order book
        for order in liquidate_short:
            self.order_book.update(order)

        return liquidate_long + liquidate_short, position_delta

    def market_make(self, positions):
        orders = []

        # Get current market state
        spread = self.order_book.spread
        mid_price = self.order_book.mid_price
        imbalance = self.order_book.calculate_order_book_imbalance()

        # Adjust based on order book imbalance
        spread_adjustment = 0
        price_skew = 0

        # If we have strong buying pressure, shift our quotes higher
        if imbalance > 1.5:
            spread_adjustment = 1
            price_skew = 1
        # If we have strong selling pressure, shift our quotes lower
        elif imbalance < 0.5:
            spread_adjustment = 1
            price_skew = -1

        # Further adjust based on our current position
        if positions["mm_pos"] > 22:  # We're long, so prefer to sell
            price_skew -= 1
            spread_adjustment -= 1
        elif positions["mm_pos"] < -22:  # We're short, so prefer to buy
            price_skew += 1
            spread_adjustment -= 1

        # Calculate our bid and ask prices
        half_spread = (spread + spread_adjustment) // 2
        bid_price = int(mid_price) - half_spread + price_skew
        ask_price = int(mid_price) + half_spread + price_skew

        # Make sure our prices are sensible relative to the mean value
        if bid_price > self.mean:
            bid_price = int(self.mean)
        if ask_price < self.mean:
            ask_price = int(self.mean)

        # Calculate appropriate volumes based on position limits
        remaining_long_capacity = self.pos_limit - positions["max_long"]

        remaining_short_capacity = self.pos_limit + positions["max_short"]
        # Note: current_position could be negative

        # Scale our order sizes based on how far we are from position limits
        bid_volume = min(self.mm_default_vol, remaining_long_capacity)
        ask_volume = min(self.mm_default_vol, remaining_short_capacity)

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

        mt_position = self.mt_positions.tot_position
        mm_position = position - mt_position

        self.logger.print(f"position {position}")
        self.logger.print(f"mt_position {mt_position}")
        self.logger.print(f"mm_position {mm_position}")

        liquidated_orders, delta_pos_l = self.liquidate_mt_orders()
        orders += liquidated_orders

        mt_orders, delta_pos_mt = self.market_take()
        orders += mt_orders

        mt_orders = liquidated_orders + mt_orders

        pos_delta, neg_delta = self.calculate_delta_by_direction(mt_orders)

        max_long_position = position + pos_delta
        max_max_short_position = position + neg_delta

        positions = {
            "current": position,
            "max_long": max_long_position,
            "max_short": max_max_short_position,
            "mm_pos": mm_position,
        }

        mm_orders = self.market_make(positions)
        orders += mm_orders

        self.print_orders(orders)

        self.print_product_end()

        return orders

# Code from trader.py
config_rainforest = {
    # Market taking parameters
    "mt_pos_limit": 30,
    "mt_hl_target": 5,
    "mt_bid_edge": 0,
    "mt_ask_edge": 0,
    "mt_short_pm": 2,
    "mt_long_pm": 2,
    # Market making parameters
    "mm_default_vol": 15,
}


class Trader:
    def __init__(self):
        self.logger = CustomLogger()

    def run(self, state: TradingState):
        t1 = time()

        self.logger.print("TRADER_BEGIN")
        self.logger.print(f"timestamp {state.timestamp}")

        result = {}
        if not state.traderData:
            products = {}
            products["RAINFOREST_RESIN"] = RainforestResin(config_rainforest)
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
                    order_depth, position, own_trades, state.timestamp
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
        self.logger.print("TRADER_END")
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