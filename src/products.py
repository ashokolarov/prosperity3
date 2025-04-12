from abc import ABC, abstractmethod
from collections import deque

from datamodel import Order
from market_utils import OrderBook
from utils import CustomLogger


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

    def print_product_begin(self):
        self.logger.print(f"PRODUCT_B {self.symbol}")
        self.logger.print_numeric("timestamp", self.timestamp)

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
        self.print_product_begin()
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
        # ------------------------------------------------
        self.print_orders(self.orders)
        self.print_product_end()

        return self.orders


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
        self.print_product_begin()
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

        self.print_orders(self.orders)
        self.print_product_end()

        return self.orders


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
        self.print_product_begin()
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
        self.directional_trade()

        self.print_orders(self.orders)
        self.print_product_end()

        return self.orders


# -----------------Croissant-----------------#
class Croissants(Product):
    def __init__(self, config):
        super().__init__(config)

        # Croissant parameters
        self.name = "Croissants"
        self.symbol = "CROISSANTS"
        self.pos_limit = 250

    def update_product(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin()
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

    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        self.print_orders(self.orders)
        self.print_product_end()

        return self.orders


# -----------------Jam-----------------#
class Jams(Product):
    def __init__(self, config):
        super().__init__(config)

        # Jam parameters
        self.name = "Jams"
        self.symbol = "JAMS"
        self.pos_limit = 350

    def update_product(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin()
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

    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        self.print_orders(self.orders)
        self.print_product_end()

        return self.orders


# -----------------Djembe-----------------#
class Djembes(Product):
    def __init__(self, config):
        super().__init__(config)

        # Djembe parameters
        self.name = "Djembes"
        self.symbol = "DJEMBES"
        self.pos_limit = 60

    def update_product(self, order_depths, position, own_trades, timestamp):
        self.print_product_begin()
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

    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        self.print_orders(self.orders)
        self.print_product_end()

        return self.orders


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
        self.print_product_begin()
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
        self.market_make()

        self.print_orders(self.orders)
        self.print_product_end()

        return self.orders


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
        self.print_product_begin()
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
        self.market_make()

        self.print_orders(self.orders)
        self.print_product_end()

        return self.orders


# Synthetic Basket 1
class SyntheticBasket1(Product):
    def __init__(self, config):
        super().__init__(config)

        # Synthetic Basket 1 parameters
        self.name = "Synthetic Basket 1"
        self.symbol = "SYNTHETIC_BASKET1"
