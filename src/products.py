from abc import ABC, abstractmethod

from datamodel import Order
from market_utils import MMPositionBook, OrderBook, PositionBook


class Product(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calculate_orders(self, order_depths, position, own_trades, timestamp):
        pass


class RainforestResin(Product):
    def __init__(self, config):
        self.name = "Rainforest Resin"
        self.symbol = "RAINFOREST_RESIN"
        self.pos_limit = 50

        self.mean = config.get("mean")
        self.std = config.get("std")
        self._x, self._y = (2.0, 1.0)

        self.order_book = OrderBook()

        self.mm_positions = MMPositionBook()
        self.mm_order_volume = config.get("mm_order_volume")

        self.mt_positions = PositionBook(self.pos_limit)
        self.hard_mt_pos_limit = config.get("mt_hard_limit") * self.pos_limit
        self.hard_liquidate_target_percentage = config.get(
            "hard_liquidate_target_percentage"
        )

    def market_take(self):
        # Check if there is an opportunity to market take in ask orders
        position_delta = 0
        mt_bid_orders = []
        for depth_level in range(self.order_book.ask_orders_depth):
            ask_price, ask_volume = self.order_book.get_ask_order_at_depth(depth_level)
            if ask_price <= self.mean:
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
            if bid_price >= self.mean:
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

    def liquidate_mt_orders(self, profit_margin):
        position_delta = 0

        close_long = []
        for depth_level in range(self.order_book.bid_orders_depth):
            if self.mt_positions.long_pos > 0:
                long_price, long_pos = self.mt_positions.get_long_position()
                bid_price, bid_volume = self.order_book.get_bid_order_at_depth(
                    depth_level
                )
                if bid_price - long_price >= profit_margin:
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
                if short_price - ask_price >= profit_margin:
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
        liquidation_orders = []

        original_long_pos = self.mt_positions.long_pos
        target_long = int(self.hard_liquidate_target_percentage * original_long_pos)
        sold_long = 0

        # For long positions over the threshold
        if self.mt_positions.long_pos > 0:
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
                    liquidation_orders.append(ask_order)
                    self.mt_positions.liquidate_pos(-qty)
                    position_delta -= qty

                    sold_long += qty

                    # If we've reduced enough, stop
                    if sold_long >= target_long:
                        break

        original_short_pos = self.mt_positions.short_pos
        target_short = int(self.hard_liquidate_target_percentage * original_short_pos)
        sold_short = 0

        # For short positions over the threshold
        if self.mt_positions.short_pos > 0:
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
                    liquidation_orders.append(bid_order)
                    self.mt_positions.liquidate_pos(qty)
                    position_delta += qty

                    sold_short += qty

                    # If we've reduced enough, stop
                    if sold_short >= target_short:
                        break

        # Update order book
        for order in liquidation_orders:
            self.order_book.update(order)

        return liquidation_orders, position_delta

    def market_make(self, positions):
        orders = []

        # Get current market state
        spread = self.order_book.spread
        mid_price = self.order_book.mid_price
        imbalance = self.order_book.calculate_order_book_imbalance()

        # Determine our bid and ask prices
        # Base spread is typically 2-6 ticks based on market conditions
        base_spread = max(2, min(6, int(spread * 0.8)))

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
        if positions["mm_pos"] > 10:  # We're long, so prefer to sell
            price_skew -= 1
        elif positions["mm_pos"] < -10:  # We're short, so prefer to buy
            price_skew += 1

        # Calculate our bid and ask prices
        half_spread = (base_spread + spread_adjustment) // 2
        bid_price = int(mid_price) - half_spread + price_skew
        ask_price = int(mid_price) + half_spread + price_skew

        # Make sure our prices are sensible relative to the mean value
        if bid_price > self.mean:
            bid_price = int(self.mean)
        if ask_price < self.mean:
            ask_price = int(self.mean)

        # Calculate appropriate volumes based on position limits
        remaining_long_capacity = min(
            self.mm_order_volume, self.pos_limit - positions["max_long"]
        )
        remaining_short_capacity = min(
            self.mm_order_volume, self.pos_limit + positions["max_short"]
        )  # Note: current_position could be negative

        # Scale our order sizes based on how far we are from position limits
        bid_volume = remaining_long_capacity
        ask_volume = remaining_short_capacity

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
        # self.order_book.reset(order_depths)

        # orders = []

        # mt_position = self.mt_positions.tot_position
        # mm_position = position - mt_position

        # if abs(mt_position) < self.hard_mt_pos_limit:
        #     mt_profit_margin = 2
        #     liquidated_orders, delta_pos_l = self.liquidate_mt_orders(mt_profit_margin)
        # else:
        #     mt_profit_margin = 0
        #     liquidated_orders, delta_pos_l = self.liquidate_mt_orders(mt_profit_margin)

        #     hard_liquidated, delta_pos_hl = self.hard_liquidate()
        #     liquidated_orders += hard_liquidated

        # orders += liquidated_orders

        # mt_orders, delta_pos_mt = self.market_take()
        # orders += mt_orders

        # mt_orders = liquidated_orders + mt_orders

        # pos_delta, neg_delta = self.calculate_delta_by_direction(mt_orders)

        # max_long_position = position + pos_delta
        # max_max_short_position = position + neg_delta

        # positions = {
        #     "current": position,
        #     "max_long": max_long_position,
        #     "max_short": max_max_short_position,
        #     "mm_pos": mm_position,
        # }

        # mm_orders = self.market_make(positions)
        # orders += mm_orders

        # return orders

        # ----------------------------
        self.order_book.reset(order_depths)
        moves = []

        mt_volume = 30

        price = self.order_book.mid_price
        mm_volume = self.pos_limit - abs(position)

        if price > int(self.mean + self._x * self.std):
            ask_price = self.order_book.best_bid[0]
            if position >= 0:
                ask_volume = mt_volume
                mm_volume -= min(mt_volume, self.pos_limit - abs(position))
            else:
                ask_volume = min(mt_volume, self.pos_limit - abs(position))
                mm_volume = max(0, mm_volume - ask_volume)

            bid_order = Order(self.symbol, ask_price, -ask_volume)
            moves.append(bid_order)
            self.order_book.update(bid_order)

        elif price < int(self.mean - 0.5 * self._x * self.std):
            bid_price = self.order_book.best_ask[0]
            if position <= 0:
                bid_volume = mt_volume
                mm_volume -= min(mt_volume, self.pos_limit - abs(position))
            else:
                bid_volume = min(mt_volume, self.pos_limit - abs(position))
                mm_volume = max(0, mm_volume - bid_volume)

            ask_order = Order(self.symbol, bid_price, bid_volume)
            moves.append(ask_order)
            self.order_book.update(ask_order)

        price = self.order_book.mid_price
        ask_price = int(price) + 3
        ask_volume = mm_volume
        bid_order = Order(self.symbol, ask_price, -ask_volume)
        moves.append(bid_order)
        self.order_book.update(bid_order)

        bid_price = int(price) - 3
        bid_volume = mm_volume
        ask_order = Order(self.symbol, bid_price, bid_volume)
        moves.append(ask_order)
        self.order_book.update(ask_order)

        return moves
