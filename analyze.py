import argparse

import dash
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html

from log_processing import process_log


def get_visualizer(file_path):
    # Extract the lines
    states, activities, trades = process_log(file_path)

    products = activities["product"].unique()
    timestamps = sorted(activities["timestamp"].unique())

    # Initialize Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.Div(
                [
                    # Dropdown on the left
                    html.Div(
                        [
                            html.Label("Select Product:", style={"marginRight": "5px"}),
                            dcc.Dropdown(
                                id="product-dropdown",
                                options=[{"label": p, "value": p} for p in products],
                                value=products[1],
                                clearable=False,
                                style={
                                    "width": "200px",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                        },
                    ),
                    # Title on the left with flex to take up more space
                    html.H1(
                        "Restarted Quants Visualization",
                        style={"textAlign": "center", "margin": "0", "flex": "1"},
                    ),
                    html.Div(
                        style={"width": "200px"}
                    ),  # Empty div on the right for balance
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "marginBottom": "20px",
                    "marginTop": "10px",
                    "paddingLeft": "20px",
                    "paddingRight": "20px",
                },
            ),
            html.Div(
                [
                    dcc.Graph(id="position-chart", style={"flex": "1"}),
                    dcc.Graph(id="pnl-chart", style={"flex": "1"}),
                ],
                style={"display": "flex"},
            ),
            dcc.Graph(
                id="mid-price-chart",
                style={
                    "flex": "1",
                },
            ),
            # Slider with full width
            html.Div(
                [
                    html.Label(
                        "Select Timestamp:",
                        style={
                            "marginBottom": "5px",
                            "display": "block",
                            "marginLeft": "20px",
                        },
                    ),
                    dcc.Slider(
                        id="timestamp-slider",
                        min=min(timestamps),
                        max=max(timestamps),
                        value=min(timestamps),
                        marks={
                            ts: {
                                "label": str(ts),
                                "style": {
                                    "transform": "rotate(45deg)",
                                    "white-space": "nowrap",
                                    "margin-top": "10px",
                                },
                            }
                            for ts in range(
                                min(timestamps),
                                max(timestamps) + 1,
                                (max(timestamps) - min(timestamps)) // 10,
                            )
                        },
                        step=100,
                        tooltip={
                            "placement": "bottom",
                            "always_visible": True,
                            "style": {
                                "fontSize": "16px",  # Increase font size
                                "fontWeight": "bold",  # Make it bold for better visibility
                                "padding": "8px",  # Add more padding
                            },
                        },
                    ),
                ],
                style={
                    "width": "98%",
                    "marginBottom": "30px",  # Extra space for rotated timestamp labels
                    "paddingRight": "20px",
                    "paddingLeft": "20px",
                },
            ),
            html.Div(
                [
                    # Order Book Table
                    html.Div(
                        [
                            html.H3("Order Book", style={"textAlign": "center"}),
                            dash_table.DataTable(
                                id="order-book-table",
                                columns=[
                                    {
                                        "name": "Bid volume",
                                        "id": "bid_volume",
                                        "type": "numeric",
                                    },
                                    {"name": "Price", "id": "price", "type": "numeric"},
                                    {
                                        "name": "Ask volume",
                                        "id": "ask_volume",
                                        "type": "numeric",
                                    },
                                ],
                                style_data_conditional=[
                                    {
                                        "if": {
                                            "column_id": "bid_volume",
                                            "filter_query": "{bid_volume} > 0",
                                        },
                                        "backgroundColor": "#eaf7ea",
                                    },
                                    {
                                        "if": {
                                            "column_id": "ask_volume",
                                            "filter_query": "{ask_volume} > 0",
                                        },
                                        "backgroundColor": "#fdebea",
                                    },
                                ],
                                style_table={
                                    "width": "350px",
                                    "margin": "0 auto",
                                },
                            ),
                        ],
                        style={
                            "width": "25%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "marginRight": "1%",
                        },
                    ),
                    # Orders Table
                    html.Div(
                        [
                            html.H3("Orders", style={"textAlign": "center"}),
                            html.Div(
                                dash_table.DataTable(
                                    id="orders-table",
                                    columns=[
                                        {"name": "Type", "id": "Type"},
                                        {
                                            "name": "Price",
                                            "id": "Price",
                                            "type": "numeric",
                                            "format": {"specifier": ",.0f"},
                                        },
                                        {
                                            "name": "Quantity",
                                            "id": "Quantity",
                                            "type": "numeric",
                                        },
                                    ],
                                    style_table={"width": "350px"},
                                    style_data_conditional=[
                                        {
                                            "if": {"filter_query": '{Type} = "Sell"'},
                                            "backgroundColor": "#fde9e9",
                                            "color": "#d32f2f",
                                            "fontWeight": "bold",
                                        },
                                        {
                                            "if": {"filter_query": '{Type} = "Buy"'},
                                            "backgroundColor": "#e9fdee",
                                            "color": "#2e7d32",
                                            "fontWeight": "bold",
                                        },
                                    ],
                                ),
                                style={
                                    "display": "flex",
                                    "justifyContent": "center",
                                },
                            ),
                        ],
                        style={
                            "width": "25%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "marginRight": "1%",
                        },
                    ),
                    # Trades Table
                    html.Div(
                        [
                            html.H3("Recent Trades", style={"textAlign": "center"}),
                            html.Div(
                                dash_table.DataTable(
                                    id="trades-table",
                                    columns=[
                                        {"name": "Buyer", "id": "Buyer"},
                                        {"name": "Seller", "id": "Seller"},
                                        {
                                            "name": "Price",
                                            "id": "Price",
                                            "type": "numeric",
                                        },
                                        {
                                            "name": "Quantity",
                                            "id": "Quantity",
                                            "type": "numeric",
                                        },
                                    ],
                                    style_table={"width": "350px"},
                                    style_data_conditional=[
                                        {
                                            "if": {"column_id": "Buyer"},
                                            "fontWeight": "bold",
                                            "color": "#2e7d32",
                                        },
                                        {
                                            "if": {"column_id": "Seller"},
                                            "fontWeight": "bold",
                                            "color": "#d32f2f",
                                        },
                                    ],
                                ),
                                style={
                                    "display": "flex",
                                    "justifyContent": "center",
                                },
                            ),
                        ],
                        style={
                            "width": "25%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                        },
                    ),
                    html.Div(
                        [
                            html.H3("Current Positions", style={"textAlign": "center"}),
                            html.Div(
                                dash_table.DataTable(
                                    id="positions-table",
                                    columns=[
                                        {
                                            "name": "Position Type",
                                            "id": "Position_Type",
                                        },
                                        {
                                            "name": "Value",
                                            "id": "Value",
                                            "type": "numeric",
                                        },
                                    ],
                                    style_table={"width": "100%"},
                                    style_data_conditional=[
                                        {
                                            "if": {"filter_query": "{Value} > 0"},
                                            "color": "#2e7d32",  # Green for positive values
                                            "fontWeight": "bold",
                                        },
                                        {
                                            "if": {"filter_query": "{Value} < 0"},
                                            "color": "#d32f2f",  # Red for negative values
                                            "fontWeight": "bold",
                                        },
                                    ],
                                    style_cell={
                                        "textAlign": "center",
                                        "fontSize": "16px",
                                        "padding": "10px 15px",
                                    },
                                    style_header={
                                        "fontWeight": "bold",
                                        "backgroundColor": "#f8f9fa",
                                    },
                                ),
                                style={
                                    "display": "flex",
                                    "justifyContent": "center",
                                },
                            ),
                        ],
                        style={
                            "width": "25%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                        },
                    ),
                ],
                style={
                    "width": "100%",
                    "display": "flex",
                    "justifyContent": "center",
                    "marginTop": "30px",
                    "marginBottom": "100px",
                },
            ),
        ]
    )

    @app.callback(
        [
            Output("position-chart", "figure"),
            Output("pnl-chart", "figure"),
            Output("mid-price-chart", "figure"),
            Output("order-book-table", "data"),
            Output("orders-table", "data"),
            Output("trades-table", "data"),
            Output("positions-table", "data"),
        ],
        [Input("product-dropdown", "value"), Input("timestamp-slider", "value")],
    )
    def update_graphs(selected_product, timestamp_value):
        # Filter data for the selected product
        state = states[states["product"] == selected_product]
        activity = activities[activities["product"] == selected_product]
        trade = trades[trades["symbol"] == selected_product]

        t_idx_state = state[state["timestamp"] == timestamp_value].index[0]
        t_idx_act = activity[activity["timestamp"] == timestamp_value].index[0]

        # Position Chart
        position_fig = go.Figure()
        position_fig.add_trace(
            go.Scatter(
                x=state["timestamp"],
                y=state["position"],
                mode="lines",
                name="Position",
                line=dict(color="green"),
                hovertemplate="Timestamp: %{x:,.0f}<br>Position: %{y}<extra></extra>",
            )
        )
        position_fig.add_trace(
            go.Scatter(
                x=state["timestamp"],
                y=state["mt_position"],
                mode="lines",
                name="MT Position",
                line=dict(color="red", dash="dot"),
                hovertemplate="Timestamp: %{x:,.0f}<br>Position: %{y}<extra></extra>",
            )
        )
        position_fig.add_trace(
            go.Scatter(
                x=state["timestamp"],
                y=state["mm_position"],
                mode="lines",
                name="MM Position",
                line=dict(color="blue", dash="dot"),
                hovertemplate="Timestamp: %{x:,.0f}<br>Position: %{y}<extra></extra>",
            )
        )

        # Add a horizontal line at y=0
        position_fig.add_shape(
            type="line",
            x0=min(state["timestamp"]),
            y0=50,
            x1=max(state["timestamp"]),
            y1=50,
            line=dict(
                color="gray",
                width=3,
                dash="dash",
            ),
        )
        position_fig.add_shape(
            type="line",
            x0=min(state["timestamp"]),
            y0=-50,
            x1=max(state["timestamp"]),
            y1=-50,
            line=dict(
                color="gray",
                width=3,
                dash="dash",
            ),
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

        orders_data = []
        if not activity.empty:
            orders = state["orders"].iloc[t_idx_state]
            for order in orders:
                orders_data.append(
                    {
                        "Type": "Buy" if order[0] > 0 else "Sell",
                        "Price": order[0],
                        "Quantity": abs(order[1]),
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
        if not state.empty:
            position_data.append(
                {
                    "Position_Type": "Position",
                    "Value": state["position"].iloc[t_idx_state],
                }
            )
            position_data.append(
                {
                    "Position_Type": "MT Position",
                    "Value": state["mt_position"].iloc[t_idx_state],
                }
            )
            position_data.append(
                {
                    "Position_Type": "MM Position",
                    "Value": state["mm_position"].iloc[t_idx_state],
                }
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
            )
        )
        mid_price_fig.update_layout(
            title="Mid Price Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Price",
            title_x=0.5,
            xaxis=dict(
                tickformat=",d",  # Format tick labels with comma separators and no decimal places
            ),
            margin=dict(
                t=60, b=20, l=20, r=20
            ),  # Reduce top margin to bring title closer
        )

        return (
            position_fig,
            pnl_fig,
            mid_price_fig,
            order_book_data,
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
        "log_file",
        nargs="?",
        default="backtests/prosperity.log",
        help="Path to the log file to visualize (default: backtests/prosperity.log)",
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
    app = get_visualizer(args.log_file)

    debug_mode = args.debug
    port = args.port
    print(f"Starting visualization server on http://127.0.0.1:{port}/")
    app.run(debug=debug_mode, port=port)
