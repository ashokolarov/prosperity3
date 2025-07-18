from copy import deepcopy


class OrderBook:
    def __init__(self):
        self.ask_prices = []
        self.ask_volumes = []
        self.bid_prices = []
        self.bid_volumes = []

    def reset(self, order_depths):
        sell_orders = order_depths.sell_orders
        buy_orders = order_depths.buy_orders

        # Reset order book
        sell_orders = list(sell_orders.items())
        self.ask_prices = [order[0] for order in sell_orders]
        self.ask_volumes = [abs(order[1]) for order in sell_orders]

        buy_orders = list(buy_orders.items())
        self.bid_prices = [order[0] for order in buy_orders]
        self.bid_volumes = [order[1] for order in buy_orders]

    def check_if_no_orders(self):
        return (
            len(self.bid_prices) == 0
            or len(self.ask_prices) == 0
            or self.bid_volumes[0] == 0
            or self.ask_volumes[0] == 0
        )

    def get_best_bid(self):
        if len(self.bid_prices) == 0:
            return None, None
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

    def get_bid_prices(self):
        return deepcopy(self.bid_prices)

    def get_ask_prices(self):
        return deepcopy(self.ask_prices)

    def get_bid_volumes(self):
        return deepcopy(self.bid_volumes)

    def get_ask_volumes(self):
        return deepcopy(self.ask_volumes)

    @property
    def bid_orders_depth(self):
        return len(self.bid_prices)

    @property
    def ask_orders_depth(self):
        return len(self.ask_prices)

    @property
    def spread(self):
        if self.check_if_no_orders():
            return None
        else:
            return self.ask_prices[0] - self.bid_prices[0]

    @property
    def mid_price(self):
        if self.check_if_no_orders():
            return None
        else:
            return (self.ask_prices[0] + self.bid_prices[0]) / 2

    @property
    def vwap(self):
        if self.check_if_no_orders():
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

    def get_mm_fair(self, adverse_volume, with_spread=False):
        if (
            self.ask_orders_depth == 0
            or self.bid_orders_depth == 0
            or max(self.ask_volumes) < adverse_volume
            or max(self.bid_volumes) < adverse_volume
        ):
            return None
        else:
            filtered_ask = [
                self.ask_prices[idx]
                for idx in range(self.ask_orders_depth)
                if self.ask_volumes[idx] >= adverse_volume
            ]

            filtered_bid = [
                self.bid_prices[idx]
                for idx in range(self.bid_orders_depth)
                if self.bid_volumes[idx] >= adverse_volume
            ]

            mm_ask = max(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = min(filtered_bid) if len(filtered_bid) > 0 else None

            if mm_ask is None or mm_bid is None:
                return None

            fair_price = (mm_ask + mm_bid) / 2
            spread = mm_ask - mm_bid

            if with_spread:
                return fair_price, spread
            else:
                return fair_price

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

    def update(self, price, quantity):
        if quantity > 0:  # Buy order
            if price in self.ask_prices:
                index = self.ask_prices.index(price)
                if self.ask_volumes[index] > quantity:
                    self.ask_volumes[index] -= quantity
                elif self.ask_volumes[index] < quantity:
                    volumes = deepcopy(self.ask_volumes)
                    self.bid_volumes.append(quantity - volumes[index])
                    self.bid_prices.append(price)
                    self.ask_prices.pop(index)
                    self.ask_volumes.pop(index)
                else:
                    self.ask_prices.pop(index)
                    self.ask_volumes.pop(index)
            else:
                if price in self.bid_prices:
                    index = self.bid_prices.index(price)
                    self.bid_volumes[index] += quantity
                else:
                    self.bid_prices.append(price)
                    self.bid_volumes.append(quantity)
        else:  # Sell order
            if price in self.bid_prices:
                index = self.bid_prices.index(price)
                if self.bid_volumes[index] > abs(quantity):
                    self.bid_volumes[index] -= abs(quantity)
                elif self.bid_volumes[index] < abs(quantity):
                    volumes = deepcopy(self.bid_volumes)
                    self.ask_volumes.append(abs(quantity) - volumes[index])
                    self.ask_prices.append(price)
                    self.bid_prices.pop(index)
                    self.bid_volumes.pop(index)
                else:
                    self.bid_prices.pop(index)
                    self.bid_volumes.pop(index)
            else:
                if price in self.ask_prices:
                    index = self.ask_prices.index(price)
                    self.ask_volumes[index] += abs(quantity)
                else:
                    self.ask_prices.append(price)
                    self.ask_volumes.append(abs(quantity))

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
