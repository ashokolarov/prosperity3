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
