from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from algo_tools import WelfordStatsWithPriors
from autils import CustomLogger
from datamodel import Order
from market_utils import OrderBook


class Product(ABC):
    name: str = None
    symbol: str = None
    pos_limit: int = None
    order_book: OrderBook = None
    logger: CustomLogger = None
    orders: list = None
    position: int = None
    remaining_buy: int = None
    remaining_sell: int = None
    timestamp: int = None

    def __init__(self):
        self.logger = CustomLogger()
        self.order_book = OrderBook()

    def __getstate__(self):
        state = self.__dict__.copy()
        if "logger" in state:
            del state["logger"]
        if "order_book" in state:
            del state["order_book"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = CustomLogger()
        self.order_book = OrderBook()

    def print_product_begin(self, timestamp):
        self.logger.print(f"PRODUCT_B {self.symbol}")
        self.logger.print_numeric("timestamp", timestamp)

    def print_product_end(self):
        self.logger.print(f"PRODUCT_E {self.symbol}")
        self.logger.flush()

    def print_order(self, order):
        self.logger.print(f"order {order.quantity}@{order.price}")

    def place_order(
        self,
        price,
        quantity,
        update_order_book=True,
    ):
        if update_order_book:
            self.order_book.update(price, quantity)

        if quantity > 0:  # Buy order
            self.remaining_buy -= quantity
            self.position += quantity
        elif quantity < 0:  # Sell order (negative quantity)
            self.remaining_sell += quantity
            self.position -= quantity

        order = Order(self.symbol, price, quantity)
        self.print_order(order)
        self.orders.append(order)

    def on_timestep_end(self):
        self.print_product_end()
        return self.orders

    def update_product(self, order_depths, position, own_trades, timestamp, obs):
        self.print_product_begin(timestamp)
        self.logger.print_numeric("position", position)

        # Reset order book
        self.order_book.reset(order_depths)

        # Reset orders
        self.orders = []

        self.timestamp = timestamp

        # Update position
        self.position = position
        self.remaining_buy = self.pos_limit - position
        self.remaining_sell = self.pos_limit + position

    def get_positions(self):
        return self.position, self.remaining_buy, self.remaining_sell

    @abstractmethod
    def calculate_orders(self, manager):
        pass


# ------------------RAINFOREST_RESIN-------------------#
class RainforestResin(Product):
    def __init__(self, config):
        super().__init__()

        # Rainforest Resin parameters
        self.name = "Rainforest Resin"
        self.symbol = "RAINFOREST_RESIN"
        self.pos_limit = 50

        # Price estimation
        self.fair_value = 10000

        # Market taking parameters
        self.mt_take_edge = config.get("mt_take_edge")
        self.mt_profit_margin = config.get("mt_profit_margin")

        # Market making parameters
        self.mm_default_vol = config.get("mm_default_vol")
        self.mm_default_edge = config.get("mm_default_edge")
        self.mm_disregard_edge = config.get("mm_disregard_edge")
        self.mm_join_edge = config.get("mm_join_edge")
        self.mm_join_volume = config.get("mm_join_volume")
        self.mm_join_edge_2 = config.get("mm_join_edge_2")
        self.mm_join_volume_2 = config.get("mm_join_volume_2")
        self.mm_constrain_below_fair = config.get("mm_constrain_below_fair")
        self.mm_manage_position = config.get("mm_manage_position")

    def market_take(self):
        ask_prices = self.order_book.get_ask_prices()
        ask_volumes = self.order_book.get_ask_volumes()
        ask_orders_depth = self.order_book.ask_orders_depth
        # Check if there is an opportunity to market take in ask orders
        for depth_level in range(ask_orders_depth):
            ask_price = ask_prices[depth_level]
            ask_volume = ask_volumes[depth_level]
            if self.fair_value - ask_price >= self.mt_take_edge:
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
            if bid_price - self.fair_value >= self.mt_take_edge:
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
                if bid_price - self.fair_value >= self.mt_profit_margin:
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
                if self.fair_value - ask_price >= self.mt_profit_margin:
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

        if self.mm_manage_position:
            if self.position > 40:
                ask_price -= 1
            elif self.position < 40:
                bid_price += 1

        if self.mm_constrain_below_fair:
            if ask_price <= self.fair_value:
                ask_price = self.fair_value + 1

            if bid_price >= self.fair_value:
                bid_price = self.fair_value - 1

        bid_price = round(bid_price)
        ask_price = round(ask_price)

        bid_volume = min(self.mm_default_vol, self.remaining_buy)
        ask_volume = min(self.mm_default_vol, self.remaining_sell)

        # Create the orders if they make sense
        if bid_volume > 0:
            self.place_order(bid_price, bid_volume)

        if ask_volume > 0:
            self.place_order(ask_price, -ask_volume)

    def calculate_orders(self):
        # Liquidation
        self.liquidate_position()

        # Market taking
        self.market_take()

        # Market making
        self.market_make()


# ------------------KELP-------------------#
class Kelp(Product):
    def __init__(self, config):
        super().__init__()

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
        self.mt_take_edge = config.get("mt_take_edge")
        self.mt_profit_margin = config.get("mt_profit_margin")
        self.mt_adverse_volume = config.get("mt_adverse_volume")

        # Market making parameters
        self.mm_default_vol = config.get("mm_default_vol")
        self.mm_disregard_edge = config.get("mm_disregard_edge")
        self.mm_default_edge = config.get("mm_default_edge")
        self.mm_join_edge = config.get("mm_join_edge")
        self.mm_join_volume = config.get("mm_join_volume")
        self.mm_constrain_below_fair = config.get("mm_constrain_below_fair")
        self.mm_manage_position = config.get("mm_manage_position")

    def update_product(self, order_depths, position, own_trades, timestamp, obs):
        # Update order book, reset orders and recalculate positions
        super().update_product(order_depths, position, own_trades, timestamp, obs)

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

        fair_value = self.estimate_fair_value(current_price)
        self.fair_value = current_price if fair_value is None else fair_value
        self.logger.print_numeric("fair_value", self.fair_value)

    def estimate_fair_value(self, observed_price):
        # Initialize Kalman filter state if not exists
        if not hasattr(self, "kf_price"):
            self.kf_price = None  # Estimated state
            self.kf_variance = 1.0  # Uncertainty in the estimate
            self.process_variance = 0.4  # How quickly the true price changes
            self.measurement_variance = 0.0  # Noise in price observations

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

        ask_price = round(self.fair_value + self.mm_default_edge)
        if best_ask_above_fair is not None:
            baaf_idx = self.order_book.ask_prices.index(best_ask_above_fair)
            best_ask_volume = self.order_book.ask_volumes[baaf_idx]
            if (
                abs(best_ask_above_fair - self.fair_value) <= self.mm_join_edge
                and best_ask_volume <= self.mm_join_volume
            ):
                ask_price = best_ask_above_fair
            else:
                ask_price = best_ask_above_fair - 1

        bid_price = round(self.fair_value - self.mm_default_edge)
        if best_bid_below_fair is not None:
            bbbf_idx = self.order_book.bid_prices.index(best_bid_below_fair)
            best_bid_volume = self.order_book.bid_volumes[bbbf_idx]
            if (
                abs(self.fair_value - best_bid_below_fair) <= self.mm_join_edge
                and best_bid_volume <= self.mm_join_volume
            ):
                bid_price = best_bid_below_fair

            else:
                bid_price = best_bid_below_fair + 1

        if self.mm_manage_position:
            if self.position > 40:
                ask_price -= 1
            elif self.position < 40:
                bid_price += 1

        if self.mm_constrain_below_fair:
            if ask_price <= self.fair_value:
                ask_price = self.fair_value + 1

            if bid_price >= self.fair_value:
                bid_price = self.fair_value - 1

        bid_price = round(bid_price)
        ask_price = round(ask_price)

        bid_volume = min(self.mm_default_vol, self.remaining_buy)
        ask_volume = min(self.mm_default_vol, self.remaining_sell)

        # Create the orders if they make sense
        if bid_volume > 0:
            self.place_order(bid_price, bid_volume)

        if ask_volume > 0:
            self.place_order(ask_price, -ask_volume)

    def market_take(self):
        ask_prices = self.order_book.get_ask_prices()
        ask_volumes = self.order_book.get_ask_volumes()
        ask_orders_depth = self.order_book.ask_orders_depth
        # Check if there is an opportunity to market take in ask orders
        for depth_level in range(ask_orders_depth):
            ask_price = ask_prices[depth_level]
            ask_volume = ask_volumes[depth_level]
            if ask_volume <= self.mt_adverse_volume:
                if self.fair_value - ask_price >= self.mt_take_edge:
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
                if bid_price - self.fair_value >= self.mt_take_edge:
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
                    if bid_price - self.fair_value >= self.mt_profit_margin:
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
                    if self.fair_value - ask_price >= self.mt_profit_margin:
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


# ------------------Squid Ink-------------------#
class Squid(Product):
    def __init__(self, config):
        super().__init__()

        # Squid parameters
        self.name = "Squid Ink"
        self.symbol = "SQUID_INK"
        self.pos_limit = 50

        # Price estimation
        self.detect_mm_volume = config.get("detect_mm_volume")
        self.short_window = config.get("short_window")
        self.long_window = config.get("long_window")
        self.std_window = config.get("std_window")

        self.window_size = max(self.short_window, self.long_window, self.std_window)
        self.history = deque(maxlen=self.window_size)

        # Directional trading
        self.dt_default_vol = config.get("dt_default_vol")
        self.dt_signal_strength = config.get("dt_signal_strength")
        self.dt_threshold_z = config.get("dt_threshold_z")
        self.z_close_threshold = config.get("z_close_threshold")

        # Price drop protection
        self.price_drop_threshold = config.get(
            "price_drop_threshold", 3.0
        )  # Z-score threshold for drop detection
        self.recovery_wait_period = config.get(
            "recovery_wait_period", 10
        )  # Number of iterations to wait
        self.recovery_counter = 0  # Count iterations after drop detected
        self.in_recovery_mode = False  # Flag to indicate we're in recovery mode
        self.recovery_position_type = (
            None  # Will be "long" or "short" depending on position during drop
        )
        self.recent_price_changes = deque(maxlen=5)  # Track recent price changes
        self.prev_price = None  # Store previous price for change calculation

    def update_product(self, order_depths, position, own_trades, timestamp, obs):
        super().update_product(order_depths, position, own_trades, timestamp, obs)

        # --------------Price estimation------------------
        mm_price = self.order_book.get_mm_fair(self.detect_mm_volume)
        self.logger.print_numeric("mm_price", mm_price)
        vwap = self.order_book.vwap
        self.logger.print_numeric("vwap", vwap)
        mid_price = self.order_book.mid_price
        self.logger.print_numeric("mid_price", mid_price)

        # Calculate price change if we have history
        if hasattr(self, "prev_price") and self.prev_price is not None:
            if self.order_book.check_if_no_orders():
                return
            price_change = mid_price - self.prev_price
            self.recent_price_changes.append(price_change)

            # Detect sudden price movements if not already in recovery mode
            if not self.in_recovery_mode and len(self.recent_price_changes) >= 3:
                # Calculate standard deviation of recent changes
                std_changes = np.std(list(self.recent_price_changes))
                if std_changes > 0:
                    # Calculate z-score of current price change
                    current_change_z = price_change / std_changes

                    # If large negative z-score while holding long positions
                    # Or large positive z-score while holding short positions
                    if (
                        current_change_z < -self.price_drop_threshold
                        and self.position > 0
                    ) or (
                        current_change_z > self.price_drop_threshold
                        and self.position < 0
                    ):
                        self.in_recovery_mode = True
                        self.recovery_counter = 0
                        self.recovery_position_type = (
                            "long" if self.position > 0 else "short"
                        )
        # Update recovery counter if in recovery mode
        if self.in_recovery_mode:
            self.recovery_counter += 1

            # Check if recovery period is over
            if self.recovery_counter >= self.recovery_wait_period:
                self.in_recovery_mode = False
                self.recovery_position_type = None
                self.recovery_counter = 0

        # Store current price for next update
        self.prev_price = mid_price

        if mm_price is None:
            self.fair_value = mid_price
        else:
            self.fair_value = mm_price
        self.logger.print_numeric("fair_value", self.fair_value)

        # Update history with the latest price
        self.history.append(self.fair_value)

    def directional_trade(self):
        # Check if we have enough data points for all three moving averages
        if len(self.history) >= self.long_window:
            price_history = list(self.history)

            long_mean = np.mean(price_history[-self.long_window :])
            short_mean = np.mean(price_history[-self.short_window :])
            std = np.std(price_history[-self.std_window :])

            self.logger.print_numeric("long_mean", long_mean)
            self.logger.print_numeric("short_mean", short_mean)
            self.logger.print_numeric("std", std)

            z_score = abs(short_mean - long_mean) / std
            self.logger.print_numeric("z_score", z_score)

            short_below_long = short_mean < long_mean

            # Check if we should close existing positions based on z_close_threshold
            # Only close positions if we're not in recovery mode OR
            # if the position is opposite to the type of position we're protecting
            should_close_position = (
                abs(z_score) < self.z_close_threshold
                and self.position != 0
                and not (
                    self.in_recovery_mode
                    and (
                        (
                            self.recovery_position_type == "long" and self.position > 0
                        )  # Protecting long positions
                        or (
                            self.recovery_position_type == "short" and self.position < 0
                        )  # Protecting short positions
                    )
                )
            )

            if should_close_position:
                # Close position logic
                if self.position > 0:
                    # We have a long position to close
                    best_ask_price, best_ask_volume = self.order_book.get_best_bid()
                    ask_volume = min(abs(self.position), best_ask_volume)
                    self.place_order(best_ask_price, -ask_volume)
                elif self.position < 0:
                    # We have a short position to close
                    best_bid_price, best_bid_volume = self.order_book.get_best_ask()
                    bid_volume = min(abs(self.position), best_bid_volume)
                    self.place_order(best_bid_price, bid_volume)
                return  # Exit after closing position

            # If we're in recovery mode for a specific position type,
            # don't initiate new positions of the same type
            if self.in_recovery_mode:
                if (self.recovery_position_type == "long" and short_below_long) or (
                    self.recovery_position_type == "short" and not short_below_long
                ):
                    return

            if short_below_long:
                # Long signal
                if self.position >= 0:
                    z_score_threshold = self.dt_threshold_z
                    bid_volume = min(
                        self.dt_default_vol,
                        self.remaining_buy,
                    )
                else:
                    z_score_threshold = 0
                    bid_volume = min(self.remaining_buy, abs(self.position))

                if z_score > z_score_threshold and self.remaining_buy > 0:
                    best_bid_price, best_bid_volume = self.order_book.get_best_ask()
                    bid_price = best_bid_price
                    bid_volume = min(bid_volume, best_bid_volume)
                    self.place_order(bid_price, bid_volume)

            elif not short_below_long:
                # Short signal
                if self.position <= 0:
                    z_score_threshold = self.dt_threshold_z
                    ask_volume = min(
                        self.dt_default_vol,
                        self.remaining_sell,
                    )
                else:
                    z_score_threshold = 0
                    ask_volume = min(self.remaining_sell, self.position)

                if self.remaining_sell > 0 and z_score > z_score_threshold:
                    best_ask_price, best_ask_volume = self.order_book.get_best_bid()
                    ask_price = best_ask_price
                    ask_volume = min(ask_volume, best_ask_volume)
                    self.place_order(ask_price, -ask_volume)

    def calculate_orders(self):
        # Directional trading
        self.directional_trade()


# -----------------Croissant-----------------#
class Croissants(Product):
    def __init__(self, config):
        super().__init__()

        # Croissant parameters
        self.name = "Croissants"
        self.symbol = "CROISSANTS"
        self.pos_limit = 250

    def update_product(self, order_depths, position, own_trades, timestamp, obs):
        super().update_product(order_depths, position, own_trades, timestamp, obs)

    def calculate_orders(self):
        pass


# -----------------Jam-----------------#
class Jams(Product):
    def __init__(self, config):
        super().__init__()

        # Jam parameters
        self.name = "Jams"
        self.symbol = "JAMS"
        self.pos_limit = 350

    def update_product(self, order_depths, position, own_trades, timestamp, obs):
        super().update_product(order_depths, position, own_trades, timestamp, obs)

    def calculate_orders(self):
        pass


# -----------------Djembe-----------------#
class Djembes(Product):
    def __init__(self, config):
        super().__init__()

        # Djembe parameters
        self.name = "Djembes"
        self.symbol = "DJEMBES"
        self.pos_limit = 60

    def update_product(self, order_depths, position, own_trades, timestamp, obs):
        super().update_product(order_depths, position, own_trades, timestamp, obs)

    def calculate_orders(self):
        pass


# -------------Picnic Basket 1 ----------------#
class PicnicBasket1(Product):
    def __init__(self, config):
        super().__init__()

        # Picnic Basket 1 parameters
        self.name = "Picnic Basket 1"
        self.symbol = "PICNIC_BASKET1"
        self.pos_limit = 60

        # Price estimation
        self.detect_mm_volume = config.get("detect_mm_volume")

        # Market making parameters
        self.enable_market_making = config.get("market_making")
        self.mm_default_vol = config.get("mm_default_vol")
        self.mm_default_edge = config.get("mm_default_edge")
        self.mm_disregard_edge = config.get("mm_disregard_edge")
        self.mm_join_edge = config.get("mm_join_edge")
        self.mm_join_volume = config.get("mm_join_volume")
        self.mm_constrain_below_fair = config.get("mm_constrain_below_fair")
        self.mm_manage_position = config.get("mm_manage_position")

    def update_product(self, order_depths, position, own_trades, timestamp, obs):
        super().update_product(order_depths, position, own_trades, timestamp, obs)

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

        ask_price = round(self.fair_value + self.mm_default_edge)
        if baaf is not None:
            baaf_idx = self.order_book.ask_prices.index(baaf)
            best_ask_volume = self.order_book.ask_volumes[baaf_idx]
            if (
                baaf - self.fair_value <= self.mm_join_edge
                and best_ask_volume <= self.mm_join_volume
            ):
                ask_price = baaf - 1
            else:
                ask_price = baaf - 2

        bid_price = round(self.fair_value - self.mm_default_edge)
        if bbbf is not None:
            bbbf_idx = self.order_book.bid_prices.index(bbbf)
            best_bid_volume = self.order_book.bid_volumes[bbbf_idx]
            if (
                abs(self.fair_value - bbbf) <= self.mm_join_edge
                and best_bid_volume <= self.mm_join_volume
            ):
                bid_price = bbbf + 1

            else:
                bid_price = bbbf + 2

        if self.mm_manage_position:
            if self.position > 0:
                ask_price -= 1
            elif self.position < 0:
                bid_price += 1

        if self.mm_constrain_below_fair:
            if ask_price <= self.fair_value:
                ask_price = self.fair_value + 1
            if bid_price >= self.fair_value:
                bid_price = self.fair_value - 1

        bid_price = round(bid_price)
        ask_price = round(ask_price)

        ob_best_bid, _ = self.order_book.get_best_bid()
        ob_best_ask, _ = self.order_book.get_best_ask()

        if bid_price >= ob_best_ask:
            bid_price = ob_best_ask - 1

        if ask_price <= ob_best_bid:
            ask_price = ob_best_bid + 1

        bid_volume = min(self.mm_default_vol, self.remaining_buy)
        ask_volume = min(self.mm_default_vol, self.remaining_sell)

        # Create the orders if they make sense
        if bid_volume > 0:
            self.place_order(bid_price, bid_volume, update_order_book=False)

        if ask_volume > 0:
            self.place_order(ask_price, -ask_volume, update_order_book=False)

    def calculate_orders(self):
        # Market making
        if self.enable_market_making:
            self.market_make()


# -------------Picnic Basket 2 ----------------#
class PicnicBasket2(Product):
    def __init__(self, config):
        super().__init__()

        # Picnic Basket 2 parameters
        self.name = "Picnic Basket 2"
        self.symbol = "PICNIC_BASKET2"
        self.pos_limit = 100

        # Price estimation
        self.detect_mm_volume = config.get("detect_mm_volume")

        # Market making parameters
        self.enable_market_making = config.get("market_making")
        self.mm_default_vol = config.get("mm_default_vol")
        self.mm_default_edge = config.get("mm_default_edge")
        self.mm_disregard_edge = config.get("mm_disregard_edge")
        self.mm_join_edge = config.get("mm_join_edge")
        self.mm_join_volume = config.get("mm_join_volume")
        self.mm_constrain_below_fair = config.get("mm_constrain_below_fair")
        self.mm_manage_position = config.get("mm_manage_position")

    def update_product(self, order_depths, position, own_trades, timestamp, obs):
        super().update_product(order_depths, position, own_trades, timestamp, obs)

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

        ask_price = round(self.fair_value + self.mm_default_edge)
        if baaf is not None:
            baaf_idx = self.order_book.ask_prices.index(baaf)
            best_ask_volume = self.order_book.ask_volumes[baaf_idx]
            if (
                baaf - self.fair_value <= self.mm_join_edge
                and best_ask_volume <= self.mm_join_volume
            ):
                ask_price = baaf
            else:
                ask_price = baaf - 1

        bid_price = round(self.fair_value - self.mm_default_edge)
        if bbbf is not None:
            bbbf_idx = self.order_book.bid_prices.index(bbbf)
            best_bid_volume = self.order_book.bid_volumes[bbbf_idx]
            if (
                abs(self.fair_value - bbbf) <= self.mm_join_edge
                and best_bid_volume <= self.mm_join_volume
            ):
                bid_price = bbbf

            else:
                bid_price = bbbf + 1

        if self.mm_manage_position:
            if self.position > 0:
                ask_price -= 1
            elif self.position < 0:
                bid_price += 1

        if self.mm_constrain_below_fair:
            if ask_price <= self.fair_value:
                ask_price = self.fair_value + 1
            if bid_price >= self.fair_value:
                bid_price = self.fair_value - 1

        bid_price = round(bid_price)
        ask_price = round(ask_price)

        ob_best_bid, _ = self.order_book.get_best_bid()
        ob_best_ask, _ = self.order_book.get_best_ask()

        if bid_price >= ob_best_ask:
            bid_price = ob_best_ask - 1

        if ask_price <= ob_best_bid:
            ask_price = ob_best_bid + 1

        bid_volume = min(self.mm_default_vol, self.remaining_buy)
        ask_volume = min(self.mm_default_vol, self.remaining_sell)

        # Create the orders if they make sense
        if bid_volume > 0:
            self.place_order(bid_price, bid_volume, update_order_book=False)

        if ask_volume > 0:
            self.place_order(ask_price, -ask_volume, update_order_book=False)

    def calculate_orders(self):
        # Market making
        if self.enable_market_making:
            self.market_make()


class SyntheticProduct:
    def __init__(self):
        self.logger = CustomLogger()

    def print_product_begin(self, timestamp):
        self.logger.print(f"PRODUCT_B {self.symbol}")
        self.logger.print_numeric("timestamp", timestamp)

    def print_product_end(self):
        self.logger.print(f"PRODUCT_E {self.symbol}")
        self.logger.flush()

    def on_timestep_end(self):
        self.print_product_end()

    def __getstate__(self):
        state = self.__dict__.copy()
        if "logger" in state:
            del state["logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = CustomLogger()

    @abstractmethod
    def calculate_orders(self, products, timestamp):
        pass


class SyntheticBasket1(SyntheticProduct):
    def __init__(self, config):
        super().__init__()

        # Synthetic Basket 1 parameters
        self.name = "Synthetic Basket 1"
        self.symbol = "SYNTHETIC_BASKET1"

        # Constituent products
        self.constituents = ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]
        self.pb1_ratio = 1
        self.crois_ratio = 6
        self.jams_ratio = 3
        self.djembe_ratio = 1

        # Open and close position thresholds
        self.buy_entry = config.get("buy_entry")
        self.buy_exit = config.get("buy_exit")
        self.sell_entry = config.get("sell_entry")
        self.sell_exit = config.get("sell_exit")

        # Price tracking
        self.N = config.get("N")
        self.BUY_SPREAD_MEAN = 16.54
        self.BUY_SPREAD_VAR = 12766.11
        self.SELL_SPREAD_MEAN = -6.032
        self.SELL_SPREAD_VAR = 12766.588

        self.buy_spread_stats = WelfordStatsWithPriors(
            self.BUY_SPREAD_MEAN, self.BUY_SPREAD_VAR, self.N
        )
        self.sell_spread_stats = WelfordStatsWithPriors(
            self.SELL_SPREAD_MEAN, self.SELL_SPREAD_VAR, self.N
        )

        # Theoretical max is 41
        self.max_basket_position = 41
        self.baskets_long = 0
        self.baskets_short = 0

        self.converge_window = 25
        self.iter = 0

    def calculate_orders(self, products, timestamp):
        self.print_product_begin(timestamp)

        # Set timestamp
        self.timestamp = timestamp
        self.iter += 1

        for constituent in self.constituents:
            if constituent not in products.keys():
                return
            if products[constituent].order_book.check_if_no_orders():
                return

        pb1 = products["PICNIC_BASKET1"]
        pb1_ask_price, pb1_ask_volume = pb1.order_book.get_best_ask()
        pb1_bid_price, pb1_bid_volume = pb1.order_book.get_best_bid()
        pb1_pos, pb1_remaining_buy, pb1_remaining_sell = pb1.get_positions()

        crois = products["CROISSANTS"]
        crois_ask_price, crois_ask_volume = crois.order_book.get_best_ask()
        crois_bid_price, crois_bid_volume = crois.order_book.get_best_bid()
        crois_pos, crois_remaining_buy, crois_remaining_sell = crois.get_positions()

        jams = products["JAMS"]
        jams_ask_price, jams_ask_volume = jams.order_book.get_best_ask()
        jams_bid_price, jams_bid_volume = jams.order_book.get_best_bid()
        jams_pos, jams_remaining_buy, jams_remaining_sell = jams.get_positions()

        djembes = products["DJEMBES"]
        djembes_ask_price, djembes_ask_volume = djembes.order_book.get_best_ask()
        djembes_bid_price, djembes_bid_volume = djembes.order_book.get_best_bid()
        djembes_pos, djembes_remaining_buy, djembes_remaining_sell = (
            djembes.get_positions()
        )

        buy_spread = self.pb1_ratio * pb1_ask_price - (
            self.crois_ratio * crois_bid_price
            + self.jams_ratio * jams_bid_price
            + self.djembe_ratio * djembes_bid_price
        )
        sell_spread = self.pb1_ratio * pb1_bid_price - (
            self.crois_ratio * crois_ask_price
            + self.jams_ratio * jams_ask_price
            + self.djembe_ratio * djembes_ask_price
        )

        self.buy_spread_stats.update(buy_spread)
        self.sell_spread_stats.update(sell_spread)

        self.logger.print_numeric("buy_spread", buy_spread)
        self.logger.print_numeric("sell_spread", sell_spread)

        if self.iter < self.converge_window:
            self.on_timestep_end()
            return

        # BASKET BUY STRATEGY (Long PB1, Short Components)
        # Calculate max basket units based on position limits
        basket_buy_limits = [
            pb1_remaining_buy // self.pb1_ratio,
            crois_remaining_sell // self.crois_ratio,
            jams_remaining_sell // self.jams_ratio,
            djembes_remaining_sell // self.djembe_ratio,
        ]

        # Calculate max basket units based on available market liquidity
        liquidity_buy_limits = [
            pb1_ask_volume // self.pb1_ratio,
            crois_bid_volume // self.crois_ratio,
            jams_bid_volume // self.jams_ratio,
            djembes_bid_volume // self.djembe_ratio,
        ]
        max_baskets_buy = min(min(basket_buy_limits), min(liquidity_buy_limits))

        # BASKET SELL STRATEGY (Short PB1, Long Components)
        # Calculate max basket units based on position limits
        basket_sell_limits = [
            pb1_remaining_sell // self.pb1_ratio,
            crois_remaining_buy // self.crois_ratio,
            jams_remaining_buy // self.jams_ratio,
            djembes_remaining_buy // self.djembe_ratio,
        ]

        # Calculate max basket units based on available market liquidity
        liquidity_sell_limits = [
            pb1_bid_volume // self.pb1_ratio,
            crois_ask_volume // self.crois_ratio,
            jams_ask_volume // self.jams_ratio,
            djembes_ask_volume // self.djembe_ratio,
        ]

        # The limiting factor is the minimum of both constraints
        max_baskets_sell = min(min(basket_sell_limits), min(liquidity_sell_limits))

        # buy_std = self.BUY_SPREAD_VAR**0.5
        buy_std = self.buy_spread_stats.get_std()
        buy_mean = self.BUY_SPREAD_MEAN
        # buy_mean = self.buy_spread_stats.get_mean()
        z_score_buy = (buy_spread - buy_mean) / buy_std
        self.logger.print_numeric("z_score_buy", z_score_buy)

        sell_std = self.sell_spread_stats.get_std()
        # sell_std = self.SELL_SPREAD_VAR**0.5  # self.sell_spread_stats.get_std()
        # sell_mean = self.sell_spread_stats.get_mean()
        sell_mean = self.SELL_SPREAD_MEAN  # self.sell_spread_stats.get_mean()
        z_score_sell = (sell_spread - sell_mean) / sell_std
        self.logger.print_numeric("z_score_sell", z_score_sell)

        if (
            z_score_buy <= -self.buy_entry
            and max_baskets_buy > 0
            and self.baskets_long < self.max_basket_position
        ):
            available_for_new_positions = self.max_basket_position - self.baskets_long
            new_baskets = min(max_baskets_buy, available_for_new_positions)

            # Calculate exact volumes while respecting the basket ratio
            if new_baskets > 0:
                pb1_buy_volume = new_baskets * self.pb1_ratio
                croissants_sell_volume = new_baskets * self.crois_ratio
                jams_sell_volume = new_baskets * self.jams_ratio
                djembes_sell_volume = new_baskets * self.djembe_ratio

                # Place orders
                pb1.place_order(pb1_ask_price, pb1_buy_volume)
                crois.place_order(crois_bid_price, -croissants_sell_volume)
                jams.place_order(jams_bid_price, -jams_sell_volume)
                djembes.place_order(djembes_bid_price, -djembes_sell_volume)

                self.baskets_long += new_baskets
                self.logger.print_numeric("open_buy_spread", buy_spread)

        elif (
            z_score_buy >= -self.buy_exit
            and self.baskets_long > 0
            and max_baskets_sell > 0
        ):
            # Determine how many baskets to unwind
            baskets_to_unwind = min(max_baskets_sell, self.baskets_long)

            if baskets_to_unwind > 0:
                pb1_sell_volume = baskets_to_unwind * self.pb1_ratio
                croissants_buy_volume = baskets_to_unwind * self.crois_ratio
                jams_buy_volume = baskets_to_unwind * self.jams_ratio
                djembes_buy_volume = baskets_to_unwind * self.djembe_ratio

                # Place orders
                pb1.place_order(pb1_bid_price, -pb1_sell_volume)
                crois.place_order(crois_ask_price, croissants_buy_volume)
                jams.place_order(jams_ask_price, jams_buy_volume)
                djembes.place_order(djembes_ask_price, djembes_buy_volume)

                self.baskets_long -= baskets_to_unwind
                self.logger.print_numeric("close_buy_spread", buy_spread)

        if (
            z_score_sell >= self.sell_entry
            and self.baskets_short < self.max_basket_position
            and max_baskets_sell > 0
        ):
            available_for_new_positions = self.max_basket_position - self.baskets_short
            new_baskets = min(max_baskets_sell, available_for_new_positions)

            if new_baskets > 0:
                pb1_sell_volume = new_baskets * self.pb1_ratio
                croissants_buy_volume = new_baskets * self.crois_ratio
                jams_buy_volume = new_baskets * self.jams_ratio
                djembes_buy_volume = new_baskets * self.djembe_ratio

                # Place orders
                pb1.place_order(pb1_bid_price, -pb1_sell_volume)
                crois.place_order(crois_ask_price, croissants_buy_volume)
                jams.place_order(jams_ask_price, jams_buy_volume)
                djembes.place_order(djembes_ask_price, djembes_buy_volume)

                self.baskets_short += new_baskets
                self.logger.print_numeric("open_sell_spread", sell_spread)
        elif (
            z_score_sell <= self.sell_exit
            and self.baskets_short > 0
            and max_baskets_buy > 0
        ):
            # Determine how many baskets to unwind
            baskets_to_unwind = min(max_baskets_buy, self.baskets_short)

            if baskets_to_unwind:
                pb1_buy_volume = baskets_to_unwind * self.pb1_ratio
                croissants_sell_volume = baskets_to_unwind * self.crois_ratio
                jams_sell_volume = baskets_to_unwind * self.jams_ratio
                djembes_sell_volume = baskets_to_unwind * self.djembe_ratio

                # Place orders
                pb1.place_order(pb1_ask_price, pb1_buy_volume)
                crois.place_order(crois_bid_price, -croissants_sell_volume)
                jams.place_order(jams_bid_price, -jams_sell_volume)
                djembes.place_order(djembes_bid_price, -djembes_sell_volume)

                self.baskets_short -= baskets_to_unwind
                self.logger.print_numeric("close_sell_spread", sell_spread)

        self.on_timestep_end()


# Synthetic Basket 1
class SyntheticBasket2(SyntheticProduct):
    def __init__(self, config):
        super().__init__()

        # Synthetic Basket 1 parameters
        self.name = "Synthetic Basket 2"
        self.symbol = "SYNTHETIC_BASKET2"

        # Constituent products
        self.composition = [
            "PICNIC_BASKET2",
            "CROISSANTS",
            "JAMS",
        ]
        self.pb2_ratio = 1
        self.crois_ratio = 4
        self.jams_ratio = 2

        self.disable_pairs = config.get("disable_pairs")

        # Open and close position thresholds
        self.N = config.get("N")
        self.buy_entry = config.get("buy_entry")
        self.buy_exit = config.get("buy_exit")
        self.sell_entry = config.get("sell_entry")
        self.sell_exit = config.get("sell_exit")

        # Price tracking
        self.converge_window = 25
        self.BUY_SPREAD_MEAN = 59.77
        self.BUY_SPREAD_VAR = 4364.21
        self.SELL_SPREAD_MEAN = 46.49
        self.SELL_SPREAD_VAR = 4363.81

        self.buy_spread_stats = WelfordStatsWithPriors(
            self.BUY_SPREAD_MEAN, self.BUY_SPREAD_VAR, self.N
        )
        self.sell_spread_stats = WelfordStatsWithPriors(
            self.SELL_SPREAD_MEAN, self.SELL_SPREAD_VAR, self.N
        )

        # Theoretical max is 62
        self.max_basket_position = 62
        self.baskets_long = 0
        self.baskets_short = 0

        self.converge_window = 25
        self.iter = 0

    def calculate_orders(self, products, timestamp):
        self.print_product_begin(timestamp)

        self.timestamp = timestamp
        self.iter += 1

        for constituent in self.composition:
            if constituent not in products.keys():
                return
            if products[constituent].order_book.check_if_no_orders():
                return

        pb2 = products["PICNIC_BASKET2"]
        pb2_ask_price, pb2_ask_volume = pb2.order_book.get_best_ask()
        pb2_bid_price, pb2_bid_volume = pb2.order_book.get_best_bid()
        pb2_pos, pb2_remaining_buy, pb2_remaining_sell = pb2.get_positions()

        crois = products["CROISSANTS"]
        crois_ask_price, crois_ask_volume = crois.order_book.get_best_ask()
        crois_bid_price, crois_bid_volume = crois.order_book.get_best_bid()
        crois_pos, crois_remaining_buy, crois_remaining_sell = crois.get_positions()

        jams = products["JAMS"]
        jams_ask_price, jams_ask_volume = jams.order_book.get_best_ask()
        jams_bid_price, jams_bid_volume = jams.order_book.get_best_bid()
        jams_pos, jams_remaining_buy, jams_remaining_sell = jams.get_positions()

        buy_spread = self.pb2_ratio * pb2_ask_price - (
            self.crois_ratio * crois_bid_price + self.jams_ratio * jams_bid_price
        )
        sell_spread = self.pb2_ratio * pb2_bid_price - (
            self.crois_ratio * crois_ask_price + self.jams_ratio * jams_ask_price
        )

        self.buy_spread_stats.update(buy_spread)
        self.sell_spread_stats.update(sell_spread)

        self.logger.print_numeric("buy_spread", buy_spread)
        self.logger.print_numeric("sell_spread", sell_spread)

        if self.iter < self.converge_window or self.disable_pairs:
            self.on_timestep_end()
            return

        # BASKET BUY STRATEGY (Long PB2, Short Components)
        # Calculate max basket units based on position limits
        basket_buy_limits = [
            pb2_remaining_buy // self.pb2_ratio,
            crois_remaining_sell // self.crois_ratio,
            jams_remaining_sell // self.jams_ratio,
        ]

        # Calculate max basket units based on available market liquidity
        liquidity_buy_limits = [
            pb2_ask_volume // self.pb2_ratio,
            crois_bid_volume // self.crois_ratio,
            jams_bid_volume // self.jams_ratio,
        ]
        max_baskets_buy = min(min(basket_buy_limits), min(liquidity_buy_limits))

        # BASKET SELL STRATEGY (Short PB2, Long Components)
        # Calculate max basket units based on position limits
        basket_sell_limits = [
            pb2_remaining_sell // self.pb2_ratio,
            crois_remaining_buy // self.crois_ratio,
            jams_remaining_buy // self.jams_ratio,
        ]

        # Calculate max basket units based on available market liquidity
        liquidity_sell_limits = [
            pb2_bid_volume // self.pb2_ratio,
            crois_ask_volume // self.crois_ratio,
            jams_ask_volume // self.jams_ratio,
        ]

        # The limiting factor is the minimum of both constraints
        max_baskets_sell = min(min(basket_sell_limits), min(liquidity_sell_limits))

        # buy_std = self.BUY_SPREAD_VAR**0.5
        buy_std = self.buy_spread_stats.get_std()
        buy_mean = self.BUY_SPREAD_MEAN
        # buy_mean = self.buy_spread_stats.get_mean()
        z_score_buy = (buy_spread - buy_mean) / buy_std
        self.logger.print_numeric("z_score_buy", z_score_buy)

        sell_std = self.sell_spread_stats.get_std()
        # sell_std = self.SELL_SPREAD_VAR**0.5  # self.sell_spread_stats.get_std()
        # sell_mean = self.sell_spread_stats.get_mean()
        sell_mean = self.SELL_SPREAD_MEAN  # self.sell_spread_stats.get_mean()
        z_score_sell = (sell_spread - sell_mean) / sell_std
        self.logger.print_numeric("z_score_sell", z_score_sell)

        if (
            z_score_buy <= -self.buy_entry
            and max_baskets_buy > 0
            and self.baskets_long < self.max_basket_position
        ):
            available_for_new_positions = self.max_basket_position - self.baskets_long
            new_baskets = min(max_baskets_buy, available_for_new_positions)

            # Calculate exact volumes while respecting the basket ratio
            if new_baskets > 0:
                pb2_buy_volume = new_baskets * self.pb2_ratio
                croissants_sell_volume = new_baskets * self.crois_ratio
                jams_sell_volume = new_baskets * self.jams_ratio

                # Place orders
                pb2.place_order(pb2_ask_price, pb2_buy_volume)
                crois.place_order(crois_bid_price, -croissants_sell_volume)
                jams.place_order(jams_bid_price, -jams_sell_volume)

                self.baskets_long += new_baskets

                self.logger.print_numeric("buy_spread_open", buy_spread)

        elif (
            z_score_buy >= -self.buy_exit
            and self.baskets_long > 0
            and max_baskets_sell > 0
        ):
            # Determine how many baskets to unwind
            baskets_to_unwind = min(max_baskets_sell, self.baskets_long)

            if baskets_to_unwind > 0:
                pb2_sell_volume = baskets_to_unwind * self.pb2_ratio
                croissants_buy_volume = baskets_to_unwind * self.crois_ratio
                jams_buy_volume = baskets_to_unwind * self.jams_ratio

                # Place orders
                pb2.place_order(pb2_bid_price, -pb2_sell_volume)
                crois.place_order(crois_ask_price, croissants_buy_volume)
                jams.place_order(jams_ask_price, jams_buy_volume)

                self.baskets_long -= baskets_to_unwind

                self.logger.print_numeric("buy_spread_close", buy_spread)

        if (
            z_score_sell >= self.sell_entry
            and self.baskets_short < self.max_basket_position
            and max_baskets_sell > 0
        ):
            available_for_new_positions = self.max_basket_position - self.baskets_short
            new_baskets = min(max_baskets_sell, available_for_new_positions)

            if new_baskets > 0:
                pb2_sell_volume = new_baskets * self.pb2_ratio
                croissants_buy_volume = new_baskets * self.crois_ratio
                jams_buy_volume = new_baskets * self.jams_ratio

                # Place orders
                pb2.place_order(pb2_bid_price, -pb2_sell_volume)
                crois.place_order(crois_ask_price, croissants_buy_volume)
                jams.place_order(jams_ask_price, jams_buy_volume)

                self.baskets_short += new_baskets
                self.logger.print_numeric("sell_spread_open", sell_spread)
        elif (
            z_score_sell <= self.sell_exit
            and self.baskets_short > 0
            and max_baskets_buy > 0
        ):
            # Determine how many baskets to unwind
            baskets_to_unwind = min(max_baskets_buy, self.baskets_short)

            if baskets_to_unwind:
                pb2_buy_volume = baskets_to_unwind * self.pb2_ratio
                croissants_sell_volume = baskets_to_unwind * self.crois_ratio
                jams_sell_volume = baskets_to_unwind * self.jams_ratio

                # Place orders
                pb2.place_order(pb2_ask_price, pb2_buy_volume)
                crois.place_order(crois_bid_price, -croissants_sell_volume)
                jams.place_order(jams_bid_price, -jams_sell_volume)

                self.baskets_short -= baskets_to_unwind
                self.logger.print_numeric("sell_spread_close", sell_spread)

        self.on_timestep_end()


class VolcanicRock(Product):
    def __init__(self, config):
        super().__init__()

        # Squid parameters
        self.name = "Volcanic Rock"
        self.symbol = "VOLCANIC_ROCK"
        self.pos_limit = 400

        # Price estimation
        self.short_window = config.get("short_window")
        self.long_window = config.get("long_window")
        self.std_window = config.get("std_window")

        self.window_size = max(self.short_window, self.long_window, self.std_window)
        self.history = deque(maxlen=self.window_size)

        # Directional trading
        self.dt_default_vol = config.get("dt_default_vol")
        self.dt_signal_strength = config.get("dt_signal_strength")
        self.dt_threshold_z = config.get("dt_threshold_z")
        self.z_close_threshold = config.get("z_close_threshold")

        # Price drop protection
        self.price_drop_threshold = config.get("price_drop_threshold")
        self.recovery_wait_period = config.get("recovery_wait_period")
        self.recovery_counter = 0  # Count iterations after drop detected
        self.in_recovery_mode = False  # Flag to indicate we're in recovery mode
        self.recovery_position_type = (
            None  # Will be "long" or "short" depending on position during drop
        )
        self.recent_price_changes = deque(maxlen=5)  # Track recent price changes
        self.prev_price = None  # Store previous price for change calculation

        self.no_orders = False

    def update_product(self, order_depths, position, own_trades, timestamp, obs):
        super().update_product(order_depths, position, own_trades, timestamp, obs)

        if self.order_book.check_if_no_orders():
            self.no_orders = True
            return
        else:
            self.no_orders = False

        # --------------Price estimation------------------
        vwap = self.order_book.vwap
        self.logger.print_numeric("vwap", vwap)
        mid_price = self.order_book.mid_price
        self.logger.print_numeric("mid_price", mid_price)

        # Calculate price change if we have history
        if hasattr(self, "prev_price") and self.prev_price is not None:
            price_change = mid_price - self.prev_price
            self.recent_price_changes.append(price_change)

            # Detect sudden price movements if not already in recovery mode
            if not self.in_recovery_mode and len(self.recent_price_changes) >= 3:
                # Calculate standard deviation of recent changes
                std_changes = np.std(list(self.recent_price_changes))
                if std_changes > 0:
                    # Calculate z-score of current price change
                    current_change_z = price_change / std_changes

                    # If large negative z-score while holding long positions
                    # Or large positive z-score while holding short positions
                    if (
                        current_change_z < -self.price_drop_threshold
                        and self.position > 0
                    ) or (
                        current_change_z > self.price_drop_threshold
                        and self.position < 0
                    ):
                        self.in_recovery_mode = True
                        self.recovery_counter = 0
                        self.recovery_position_type = (
                            "long" if self.position > 0 else "short"
                        )
        # Update recovery counter if in recovery mode
        if self.in_recovery_mode:
            self.recovery_counter += 1

            # Check if recovery period is over
            if self.recovery_counter >= self.recovery_wait_period:
                self.in_recovery_mode = False
                self.recovery_position_type = None
                self.recovery_counter = 0

        # Store current price for next update
        self.prev_price = mid_price

        self.fair_value = vwap
        self.logger.print_numeric("fair_value", self.fair_value)

        # Update history with the latest price
        self.history.append(self.fair_value)

    def directional_trade(self):
        # Check if we have enough data points for all three moving averages
        if len(self.history) >= self.long_window:
            price_history = list(self.history)

            long_mean = np.mean(price_history[-self.long_window :])
            short_mean = np.mean(price_history[-self.short_window :])
            std = np.std(price_history[-self.std_window :])

            self.logger.print_numeric("long_mean", long_mean)
            self.logger.print_numeric("short_mean", short_mean)
            self.logger.print_numeric("std", std)

            z_score = abs(short_mean - long_mean) / std
            self.logger.print_numeric("z_score", z_score)

            short_below_long = short_mean < long_mean

            # Check if we should close existing positions based on z_close_threshold
            # Only close positions if we're not in recovery mode OR
            # if the position is opposite to the type of position we're protecting
            should_close_position = (
                abs(z_score) < self.z_close_threshold
                and self.position != 0
                and not (
                    self.in_recovery_mode
                    and (
                        (
                            self.recovery_position_type == "long" and self.position > 0
                        )  # Protecting long positions
                        or (
                            self.recovery_position_type == "short" and self.position < 0
                        )  # Protecting short positions
                    )
                )
            )

            if should_close_position:
                # Close position logic
                if self.position > 0:
                    # We have a long position to close
                    best_ask_price, best_ask_volume = self.order_book.get_best_bid()
                    ask_volume = min(abs(self.position), best_ask_volume)
                    self.place_order(best_ask_price, -ask_volume)
                elif self.position < 0:
                    # We have a short position to close
                    best_bid_price, best_bid_volume = self.order_book.get_best_ask()
                    bid_volume = min(abs(self.position), best_bid_volume)
                    self.place_order(best_bid_price, bid_volume)
                return  # Exit after closing position

            # If we're in recovery mode for a specific position type,
            # don't initiate new positions of the same type
            if self.in_recovery_mode:
                if (self.recovery_position_type == "long" and short_below_long) or (
                    self.recovery_position_type == "short" and not short_below_long
                ):
                    return

            if short_below_long:
                # Long signal
                if self.position >= 0:
                    z_score_threshold = self.dt_threshold_z
                    bid_volume = min(
                        self.dt_default_vol,
                        self.remaining_buy,
                    )
                else:
                    z_score_threshold = 0
                    bid_volume = min(self.remaining_buy, abs(self.position))

                if z_score > z_score_threshold and self.remaining_buy > 0:
                    best_bid_price, best_bid_volume = self.order_book.get_best_ask()
                    bid_price = best_bid_price
                    bid_volume = min(bid_volume, best_bid_volume)
                    self.place_order(bid_price, bid_volume)

            elif not short_below_long:
                # Short signal
                if self.position <= 0:
                    z_score_threshold = self.dt_threshold_z
                    ask_volume = min(
                        self.dt_default_vol,
                        self.remaining_sell,
                    )
                else:
                    z_score_threshold = 0
                    ask_volume = min(self.remaining_sell, self.position)

                if self.remaining_sell > 0 and z_score > z_score_threshold:
                    best_ask_price, best_ask_volume = self.order_book.get_best_bid()
                    ask_price = best_ask_price
                    ask_volume = min(ask_volume, best_ask_volume)
                    self.place_order(ask_price, -ask_volume)

    def calculate_orders(self):
        # Directional trading
        if not self.no_orders:
            self.directional_trade()


class Volcanic9500(VolcanicRock):
    def __init__(self, config):
        super().__init__(config)
        self.name = "Volcanic 9500"
        self.symbol = "VOLCANIC_ROCK_VOUCHER_9500"
        self.pos_limit = 200


class Volcanic9750(VolcanicRock):
    def __init__(self, config):
        super().__init__(config)
        self.name = "Volcanic 9750"
        self.symbol = "VOLCANIC_ROCK_VOUCHER_9750"
        self.pos_limit = 200


class Volcanic10000(VolcanicRock):
    def __init__(self, config):
        super().__init__(config)
        self.name = "Volcanic 10000"
        self.symbol = "VOLCANIC_ROCK_VOUCHER_10000"
        self.pos_limit = 200


class Volcanic10250(Product):
    def __init__(self, config):
        super().__init__()
        self.name = "Volcanic 10250"
        self.symbol = "VOLCANIC_ROCK_VOUCHER_10250"
        self.pos_limit = 200

        self.total_bought = 0  # Total volume bought
        self.total_cost = 0  # Total cost of all purchases
        self.avg_price = 0  # Weighted average price

        self.saturated = False

    def calculate_orders(self):
        if self.remaining_buy > 0 and not self.saturated:
            if self.order_book.check_if_no_orders():
                return
            best_ask_price, best_ask_volume = self.order_book.get_best_ask()
            bid_price = best_ask_price
            bid_volume = min(best_ask_volume, self.remaining_buy)
            self.place_order(bid_price, bid_volume)

            # Update the total volume and cost
            self.total_cost += bid_price * bid_volume
            self.total_bought += bid_volume

            # Calculate the new weighted average price
            if self.total_bought > 0:
                self.avg_price = self.total_cost / self.total_bought

        else:
            self.saturated = True

        if self.saturated and self.remaining_sell > 0:
            if self.order_book.check_if_no_orders():
                return
            # Check if the average price is above the max profit factor
            best_bid_price, best_bid_volume = self.order_book.get_best_bid()

            if self.timestamp < 500000:
                max_profit_factor = 5
            else:
                max_profit_factor = 2.5

            if self.avg_price * max_profit_factor < best_bid_price:
                ask_volume = min(
                    best_bid_volume, self.remaining_sell, abs(self.position)
                )
                self.place_order(best_bid_price, -ask_volume)


class Volcanic10500(Product):
    def __init__(self, config):
        super().__init__()
        self.name = "Volcanic 10500"
        self.symbol = "VOLCANIC_ROCK_VOUCHER_10500"
        self.pos_limit = 200

        self.total_bought = 0  # Total volume bought
        self.total_cost = 0  # Total cost of all purchases
        self.avg_price = 0  # Weighted average price

        self.saturated = False

    def calculate_orders(self):
        if self.remaining_buy > 0 and not self.saturated:
            if self.order_book.check_if_no_orders():
                return
            best_ask_price, best_ask_volume = self.order_book.get_best_ask()
            bid_price = best_ask_price
            bid_volume = min(best_ask_volume, self.remaining_buy)
            self.place_order(bid_price, bid_volume)

            # Update the total volume and cost
            self.total_cost += bid_price * bid_volume
            self.total_bought += bid_volume

            # Calculate the new weighted average price
            if self.total_bought > 0:
                self.avg_price = self.total_cost / self.total_bought

        else:
            self.saturated = True

        if self.saturated and self.remaining_sell > 0:
            # Check if the average price is above the max profit factor
            if self.order_book.check_if_no_orders():
                return

            best_bid_price, best_bid_volume = self.order_book.get_best_bid()

            if self.timestamp < 500000:
                max_profit_factor = 10
            else:
                max_profit_factor = 5

            if self.avg_price * max_profit_factor < best_bid_price:
                ask_volume = min(
                    best_bid_volume, self.remaining_sell, abs(self.position)
                )
                self.place_order(best_bid_price, -ask_volume)


class MagnificentMacarons(Product):
    def __init__(self, config):
        super().__init__()
        self.name = "Magnificent Macarons"
        self.symbol = "MAGNIFICENT_MACARONS"
        self.pos_limit = 75

        self.obs = None

    def update_product(self, order_depths, position, own_trades, timestamp, obs):
        super().update_product(order_depths, position, own_trades, timestamp, obs)

        self.obs = obs
        self.conversions = 0

    def calculate_orders(self):
        self.conversions -= self.position

        if self.obs is None:
            return

        buy_price = (
            self.obs.askPrice + self.obs.transportFees + self.obs.importTariff + 1
        )
        bid_price = self.obs.bidPrice - 0.5

        sell_price = max(round(buy_price), round(bid_price))

        vol = self.pos_limit
        self.place_order(sell_price, -vol)
