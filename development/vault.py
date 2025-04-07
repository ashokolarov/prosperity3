def get_mm_trades(self, own_trades):
    """
    Extract market-making trades from executed trades by filtering out known market-taking trades.

    Args:
        own_trades: List of executed trades

    Returns:
        List of market-making trades
    """
    if len(own_trades) == 0 or not self.prev_mt_orders:
        return []

    mm_trades = []
    prev_timestamp = self.prev_mt_orders["timestamp"]

    # Filter trades that match our timestamp
    current_timestamp_trades = [
        trade
        for trade in own_trades
        if trade.symbol == self.symbol and trade.timestamp == prev_timestamp
    ]

    for trade in current_timestamp_trades:
        # Assume it's a market-making trade unless proven otherwise
        is_mt_trade = False

        for mt_order in self.prev_mt_orders["orders"]:
            # Both price AND quantity must match to identify a market-taking trade
            if trade.price == mt_order.price and trade.quantity == mt_order.quantity:
                is_mt_trade = True
                break

        # If it's not identified as a market-taking trade, add it to mm_trades
        if not is_mt_trade:
            mm_trades.append(trade)

    return mm_trades


  def directional(self, position):
        orders = []
        if self.check_history():
            # Calculate moving averages
            short_ma = sum(self.short_history) / self.short_window
            long_ma = sum(self.long_history) / self.long_window

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