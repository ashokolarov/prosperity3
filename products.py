from abc import ABC, abstractmethod

from datamodel import Order
from market_utils import OrderBook, PositionBook


class Product(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calculate_orders(self, order_depth):
        pass


class RainforestResin(Product):
    def __init__(self):
        self.name = "Rainforest Resin"
        self.symbol = "RAINFOREST_RESIN"
        self.pos_limit = 50

        self.mean = 1e4
        self.std = 1.48
        self._x, self._y = (2.0, 1.0)

        self.order_book = OrderBook()
        self.mt_positions = PositionBook(self.pos_limit)
        self.mt_pos_limit = 20
        self.liquidation_profit_margin = 1

    def market_take(self, timestamp):
        # Check if there is an opportunity to market take in ask orders
        position_delta = 0
        mt_bid_orders = []
        for depth_level in range(self.order_book.ask_orders_depth):
            ask_price, ask_volume = self.order_book.get_ask_order_at_depth(depth_level)
            if ask_price < self.mean:
                bid_price = ask_price
                bid_volume = ask_volume
                bid_order = Order(self.symbol, bid_price, bid_volume)
                mt_bid_orders.append(bid_order)
                position_delta += bid_volume
            else:
                continue  # If even the best ask doesn't cross the mean, then no need to check further

        for bid_order in mt_bid_orders:
            self.order_book.update(bid_order)
            self.mt_positions.add_pos(bid_order, timestamp)

        # Check if there is an opportunity to market take in bid orders
        mt_ask_orders = []
        for depth_level in range(self.order_book.bid_orders_depth):
            bid_price, bid_volume = self.order_book.get_bid_order_at_depth(depth_level)
            if bid_price > self.mean:
                ask_price = bid_price
                ask_volume = bid_volume
                ask_order = Order(self.symbol, ask_price, -ask_volume)
                mt_ask_orders.append(ask_order)
                position_delta -= ask_volume
            else:
                continue  # If even the best bid doesn't cross the mean, then no need to check further

        for ask_order in mt_ask_orders:
            self.order_book.update(ask_order)
            self.mt_positions.add_pos(ask_order, timestamp)

        return mt_ask_orders + mt_bid_orders, position_delta

    def liquidate_mt_orders(self):
        orders = []
        position_delta = 0
        for pos_price, pos_volume, position_timestamp in self.mt_positions.positions:
            best_ask, best_ask_volume = self.order_book.best_ask
            best_bid, best_bid_volume = self.order_book.best_bid

            if pos_volume > 0:
                if best_bid - pos_price >= self.liquidation_profit_margin:
                    qty = min(pos_volume, best_bid_volume)
                    ask_order = Order(self.symbol, best_bid, -qty)
                    orders.append(ask_order)
                    self.order_book.update(ask_order)
                    self.mt_positions.liquidate_pos(qty, position_timestamp)
                    position_delta -= qty
            elif pos_volume < 0:
                if pos_price - best_ask >= self.liquidation_profit_margin:
                    qty = min(-pos_volume, best_ask_volume)
                    bid_order = Order(self.symbol, best_ask, qty)
                    orders.append(bid_order)
                    self.order_book.update(bid_order)
                    self.mt_positions.liquidate_pos(qty, position_timestamp)
                    position_delta += qty
        return orders, position_delta

    def market_make(self, position):
        orders = []

        price = self.order_book.mid_price

        ask_price = int(price) + 3
        ask_volume = self.pos_limit - abs(position)
        bid_order = Order(self.symbol, ask_price, -ask_volume)
        orders.append(bid_order)
        self.order_book.update(bid_order)

        bid_price = int(price) - 3
        bid_volume = self.pos_limit - abs(position)
        ask_order = Order(self.symbol, bid_price, bid_volume)
        orders.append(ask_order)
        self.order_book.update(ask_order)

        return orders

    def calculate_orders(self, order_depths, position, timestamp):
        self.order_book.reset(order_depths)

        liquidated_orders, delta_pos_l = self.liquidate_mt_orders()
        mt_orders, delta_pos_mt = self.market_take(timestamp)

        # current_position = position + delta_pos_l + delta_pos_mt
        # mm_orders = self.market_make(current_position)

        # mm_ask_price = updated_order_book.ask_prices[0] - 1
        # mm_ask_volume = 5
        # mm_sell_order = Order(self.symbol, mm_ask_price, -mm_ask_volume)
        # moves.append(mm_sell_order)
        # updated_order_book.update(mm_sell_order)

        # mm_bid_price = updated_order_book.bid_prices[0] + 1
        # mm_bid_volume = 5
        # mm_buy_order = Order(self.symbol, mm_bid_price, mm_bid_volume)
        # moves.append(mm_buy_order)
        # updated_order_book.update(mm_buy_order)

        # best_ask = order_book.ask_prices[0]
        # best_bid = order_book.bid_prices[0]

        # price = (best_ask + best_bid) / 2
        # if price >= int(self._RESIN_MEAN + self._x * self._RESIN_STD):
        #     ask_price = best_bid
        #     ask_volume = 50 + position
        #     bid_order = Order(self.symbol, ask_price, -ask_volume)
        #     moves.append(bid_order)

        # elif price <= int(self._RESIN_MEAN - 0.5 * self._x * self._RESIN_STD):
        #     bid_price = best_ask
        #     bid_volume = 50 - position
        #     ask_order = Order(self.symbol, bid_price, bid_volume)
        #     moves.append(ask_order)

        # else:
        #     ask_price = int(price) + 3
        #     ask_volume = 50 - abs(position)
        #     bid_order = Order(self.symbol, ask_price, -ask_volume)
        #     moves.append(bid_order)

        #     bid_price = int(price) - 3
        #     bid_volume = 50 - abs(position)
        #     ask_order = Order(self.symbol, bid_price, bid_volume)
        #     moves.append(ask_order)

        moves = liquidated_orders + mt_orders
        return moves
