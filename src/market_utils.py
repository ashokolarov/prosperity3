from copy import deepcopy


class OrderBook:
    def __init__(self):
        self.ask_prices = []
        self.ask_volumes = []
        self.bid_prices = []
        self.bid_volumes = []

        self.previous_ask_prices = []
        self.previous_ask_volumes = []
        self.previous_bid_prices = []
        self.previous_bid_volumes = []

    def reset(self, order_depths):
        sell_orders = order_depths.sell_orders
        buy_orders = order_depths.buy_orders

        # Save previous state
        self.previous_ask_prices = deepcopy(self.ask_prices)
        self.previous_ask_volumes = deepcopy(self.ask_volumes)
        self.previous_bid_prices = deepcopy(self.bid_prices)
        self.previous_bid_volumes = deepcopy(self.bid_volumes)

        # Reset order book
        sell_orders = list(sell_orders.items())
        self.ask_prices = [order[0] for order in sell_orders]
        self.ask_volumes = [abs(order[1]) for order in sell_orders]

        buy_orders = list(buy_orders.items())
        self.bid_prices = [order[0] for order in buy_orders]
        self.bid_volumes = [order[1] for order in buy_orders]

    def reset_to_previous(self):
        self.ask_prices = deepcopy(self.previous_ask_prices)
        self.ask_volumes = deepcopy(self.previous_ask_volumes)
        self.bid_prices = deepcopy(self.previous_bid_prices)
        self.bid_volumes = deepcopy(self.previous_bid_volumes)

    def get_best_bid(self):
        if len(self.bid_prices) == 0:
            return None
        else:
            return self.bid_prices[0], self.bid_volumes[0]

    def get_best_ask(self):
        if len(self.ask_prices) == 0:
            return None
        else:
            return self.ask_prices[0], self.ask_volumes[0]

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

    def get_mm_fair(self, adverse_volume):
        if self.ask_orders_depth == 0 or self.bid_orders_depth == 0:
            return None
        else:
            if (
                max(self.ask_volumes) >= adverse_volume
                and max(self.bid_volumes) >= adverse_volume
            ):
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

    def calculate_ofi(self):
        if len(self.previous_ask_prices) == 0 or len(self.previous_bid_prices) == 0:
            return 0
        else:
            total_bid_volume = sum(self.bid_volumes)
            total_ask_volume = sum(self.ask_volumes)
            total_bid_volume_prev = sum(self.previous_bid_volumes)
            total_ask_volume_prev = sum(self.previous_ask_volumes)

            delta_bid = total_bid_volume - total_bid_volume_prev
            delta_ask = total_ask_volume - total_ask_volume_prev

            return delta_bid - delta_ask

    def update(self, order):
        if order.quantity > 0:  # Buy order
            if order.price in self.ask_prices:
                index = self.ask_prices.index(order.price)
                if self.ask_volumes[index] > order.quantity:
                    self.ask_volumes[index] -= order.quantity
                elif self.ask_volumes[index] < order.quantity:
                    volumes = deepcopy(self.ask_volumes)
                    self.bid_volumes.append(order.quantity - volumes[index])
                    self.bid_prices.append(order.price)
                    self.ask_prices.pop(index)
                    self.ask_volumes.pop(index)
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
                    volumes = deepcopy(self.bid_volumes)
                    self.ask_volumes.append(abs(order.quantity) - volumes[index])
                    self.ask_prices.append(order.price)
                    self.bid_prices.pop(index)
                    self.bid_volumes.pop(index)
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
