import argparse
import os

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output

from development.log_processing import from_csv, process_log
from development.visualizer.layout import get_layout

POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "PICNIC_BASKET1": 60,
    "SYNTHETIC_BASKET1": 100,
}


def get_visualizer(log_file=None, prosperity_round=None, day=None):
    # Extract the lines
    if log_file is not None:
        trader_data, products_data, activities, trades = process_log(log_file)
        file_name = os.path.basename(log_file)
    else:
        activities, trades = from_csv(prosperity_round, day)
        trader_data = None
        products_data = None
        file_name = f"round{prosperity_round}_day{day}.csv"

    products = activities["product"].unique()
    product_data = activities[activities["product"] == products[0]].reset_index()
    timestamps = len(product_data["timestamp"])
    timestamps = np.arange(0, timestamps * 100, 100)

    # Initialize Dash app
    app = dash.Dash(__name__)

    app.layout = get_layout(products, timestamps, file_name)

    @app.callback(
        Output("timestamp-input", "value"),
        [Input("timestamp-slider", "value")],
    )
    def update_input_from_slider(slider_value):
        return slider_value

    @app.callback(
        Output("timestamp-slider", "value"),
        [Input("timestamp-button", "n_clicks")],
        [dash.dependencies.State("timestamp-input", "value")],
    )
    def update_slider_from_input(n_clicks, input_value):
        if n_clicks is None or input_value is None:
            # No clicks yet or empty input, return initial value
            return min(timestamps)

        # Round to nearest multiple of 100
        rounded_value = round(input_value / 100) * 100

        # Ensure the value is within the valid range
        rounded_value = max(min(timestamps), min(rounded_value, max(timestamps)))
        return rounded_value

    @app.callback(
        [
            Output("position-chart", "figure"),
            Output("pnl-chart", "figure"),
            Output("mid-price-chart", "figure"),
            Output("order-book-table", "data"),
            Output("ob-stats-table", "data"),
            Output("orders-table", "data"),
            Output("trades-table", "data"),
            Output("positions-table", "data"),
        ],
        [
            Input("product-dropdown", "value"),
            Input("timestamp-slider", "value"),
            # No need to add the button here since we're already updating the slider value
        ],
    )
    def update_graphs(selected_product, timestamp_value):
        # Filter data for the selected product
        if products_data is not None:
            if selected_product in products_data.keys():
                product_data = products_data[selected_product].reset_index()
            else:
                product_data = None
        else:
            product_data = None

        trade = trades[trades["symbol"] == selected_product].reset_index()
        activity = activities[activities["product"] == selected_product].reset_index()

        if product_data is not None:
            t_idx_prod = product_data[timestamps == timestamp_value].index[0]
        t_idx_act = activity[activity["timestamp"] == timestamp_value].index[0]

        pos_limit = POSITION_LIMITS[selected_product]

        # Position Chart
        position_fig = go.Figure()
        if product_data is not None:
            position_fig.add_trace(
                go.Scatter(
                    x=product_data["timestamp"],
                    y=product_data["position"],
                    mode="lines",
                    name="Position",
                    line=dict(color="green", width=4),
                    hovertemplate="Timestamp: %{x:,.0f}<br>Position: %{y}<extra></extra>",
                )
            )

            # Add a horizontal line at y=0
            position_fig.add_shape(
                type="line",
                x0=min(product_data["timestamp"]),
                y0=pos_limit,
                x1=max(product_data["timestamp"]),
                y1=pos_limit,
                line=dict(
                    color="black",
                    width=4,
                    dash="dash",
                ),
            )
            position_fig.add_shape(
                type="line",
                x0=min(product_data["timestamp"]),
                y0=-pos_limit,
                x1=max(product_data["timestamp"]),
                y1=-pos_limit,
                line=dict(
                    color="gray",
                    width=3,
                    dash="dash",
                ),
            )
        else:
            position_fig.add_trace(
                go.Scatter(
                    x=activity["timestamp"],
                    y=[0 for i in range(len(activity))],
                    mode="lines",
                    name="Position",
                    line=dict(color="green", width=4),
                    hovertemplate="Timestamp: %{x:,.0f}<br>Position: %{y}<extra></extra>",
                )
            )

        position_fig.update_layout(
            title="Position Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Position",
            title_x=0.5,
            hovermode="x unified",
            xaxis=dict(
                tickformat=",d",  # Format tick labels with comma separators and no decimal places
            ),
            margin=dict(
                t=30, b=20, l=20, r=20
            ),  # Reduce top margin to bring title closer
        )

        # PnL Chart
        pnl_fig = go.Figure(
            go.Scatter(
                x=activity["timestamp"],
                y=activity["profit_and_loss"],
                mode="lines",
                name="PnL",
                line=dict(width=3),
                hovertemplate="Timestamp: %{x:,.0f}<br>PnL: %{y:,.0f}<extra></extra>",
            )
        )
        pnl_fig.update_layout(
            title="Profit and Loss Over Time",
            xaxis_title="Timestamp",
            yaxis_title="PnL",
            title_x=0.5,
            xaxis=dict(
                tickformat=",d",  # Format tick labels with comma separators and no decimal places
            ),
            margin=dict(
                t=30, b=20, l=20, r=20
            ),  # Reduce top margin to bring title closer
        )

        # Mid Price Chart
        mid_price_fig = go.Figure()
        mid_price_fig.add_trace(
            go.Scatter(
                x=activity["timestamp"],
                y=activity["mid_price"],
                mode="lines",
                name="Mid Price",
                line=dict(color="blue"),
                hovertemplate="Timestamp: %{x:,.0f}<br>Mid Price: %{y}<extra></extra>",
            )
        )
        mid_price_fig.add_trace(
            go.Scatter(
                x=activity["timestamp"],
                y=activity["bid_price_1"],
                mode="lines",
                name="Best Bid",
                line=dict(color="red"),
                hovertemplate="Timestamp: %{x:,.0f}<br>Best Bid: %{y}<extra></extra>",
            )
        )

        mid_price_fig.add_trace(
            go.Scatter(
                x=activity["timestamp"],
                y=activity["ask_price_1"],
                mode="lines",
                name="Best Ask",
                line=dict(color="orange"),
                hovertemplate="Timestamp: %{x:,.0f}<br>Best Ask: %{y}<extra></extra>",
            )
        )
        mid_price_fig.update_layout(
            title="Mid Price Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Price",
            title_x=0.5,
            hovermode="x unified",
            xaxis=dict(
                tickformat=",d",  # Format tick labels with comma separators and no decimal places
            ),
            yaxis=dict(
                tickformat=",d",  # Format y-axis tick labels with comma separators and no decimal places
                range=[min(activity["mid_price"] - 4), max(activity["mid_price"]) + 4],
            ),
            margin=dict(
                t=60, b=20, l=20, r=20
            ),  # Reduce top margin to bring title closer
        )

        # Create order book table data for the selected timestamp
        order_book_data = []
        if not activity.empty:
            bid_data = [
                (
                    activity[f"bid_price_{i}"].iloc[t_idx_act],
                    activity[f"bid_volume_{i}"].iloc[t_idx_act],
                )
                for i in range(1, 4)
            ]
            ask_data = [
                (
                    activity[f"ask_price_{i}"].iloc[t_idx_act],
                    activity[f"ask_volume_{i}"].iloc[t_idx_act],
                )
                for i in range(1, 4)
            ]

            all_prices = sorted(
                set(
                    [b[0] for b in bid_data if b[1] > 0]
                    + [a[0] for a in ask_data if a[1] > 0]
                ),
                reverse=True,
            )

            for price in all_prices:
                bid_volume = next((b[1] for b in bid_data if b[0] == price), None)
                ask_volume = next((a[1] for a in ask_data if a[0] == price), None)
                order_book_data.append(
                    {"bid_volume": bid_volume, "price": price, "ask_volume": ask_volume}
                )

        order_stats_data = []
        if product_data is not None:
            if not activity.empty:
                names = [
                    "mid_price",
                    "vwap",
                    "fair_value",
                    "volatility",
                    "mm_price",
                    "mm_spread",
                ]
                for name in names:
                    if name in product_data.columns:
                        order_stats_data.append(
                            {
                                "Name": name,
                                "Value": product_data[name].iloc[t_idx_prod],
                            }
                        )

        # Orders Table
        orders_data = []
        if product_data is not None:
            if not activity.empty:
                orders = product_data["orders"].iloc[t_idx_prod]
                for order in orders:
                    orders_data.append(
                        {
                            "Type": "Buy" if order[0] > 0 else "Sell",
                            "Quantity": order[0],
                            "Price": order[1],
                        }
                    )

        # Trades Table
        trades_data = []
        if not trade.empty:
            trade_data = trade[trade["timestamp"] == timestamp_value]
            for _, trade_cur in trade_data.iterrows():
                trades_data.append(
                    {
                        "Buyer": trade_cur["buyer"],
                        "Seller": trade_cur["seller"],
                        "Price": trade_cur["price"],
                        "Quantity": trade_cur["quantity"],
                    }
                )

        # Positions Table
        position_data = []
        if product_data is not None:
            if not product_data.empty:
                position_data.append(
                    {
                        "Position_Type": "Position",
                        "Value": product_data["position"].iloc[t_idx_prod],
                    }
                )
                position_data.append(
                    {
                        "Position_Type": "MT Position",
                        "Value": (
                            product_data["mt_position"].iloc[t_idx_prod]
                            if "mt_position" in product_data.columns
                            else 0
                        ),
                    }
                )
                position_data.append(
                    {
                        "Position_Type": "MM Position",
                        "Value": (
                            product_data["mm_position"].iloc[t_idx_prod]
                            if "mm_position" in product_data.columns
                            else 0
                        ),
                    }
                )

        return (
            position_fig,
            pnl_fig,
            mid_price_fig,
            order_book_data,
            order_stats_data,
            orders_data,
            trades_data,
            position_data,
        )

    return app


# Setup command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize trading data from a log file."
    )
    parser.add_argument(
        "--log_file",
        help="Path to the log file to visualize (default: backtests/prosperity.log)",
    )
    parser.add_argument(
        "--round",
        type=int,
    )
    parser.add_argument(
        "--day",
        type=int,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the Dash server on (default: 8050)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run the server in debug mode"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.log_file is None:
        if args.round is None and args.day is None:
            raise ("Please provide a log file or specify a round and day.")
        else:
            print(args.round, args.day)
            app = get_visualizer(prosperity_round=args.round, day=args.day)
    else:
        app = get_visualizer(log_file=args.log_file)

    debug_mode = args.debug
    port = args.port
    print(f"Starting visualization server on http://127.0.0.1:{port}/")
    app.run(debug=debug_mode, port=port)
