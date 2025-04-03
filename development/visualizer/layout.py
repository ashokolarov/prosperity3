from dash import dash_table, dcc, html


def get_layout(products, timestamps, log_name):
    layout = html.Div(
        [
            html.Div(
                [
                    # Dropdown on the left
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="product-dropdown",
                                options=[{"label": p, "value": p} for p in products],
                                value=products[0],
                                clearable=False,
                                style={
                                    "marginTop": "5px",
                                    "marginLeft": "30px",
                                    "width": "250px",
                                },
                            ),
                        ],
                    ),
                    # Title on the left with flex to take up more space
                    html.H1(
                        f"Restarted Quants Dashboard - {log_name}",
                        style={"textAlign": "center", "margin": "0", "flex": "1"},
                    ),
                    html.Div(
                        style={"width": "200px"}
                    ),  # Empty div on the right for balance
                ],
                style={
                    "display": "flex",
                    "marginTop": "20px",
                    "marginBottom": "20px",
                },
            ),
            html.Div(
                [
                    dcc.Graph(id="position-chart", style={"flex": "1.2"}),
                    dcc.Graph(id="pnl-chart", style={"flex": "1"}),
                ],
                style={"display": "flex", "gap": "10px", "marginRight": "20px"},
            ),
            dcc.Graph(
                id="mid-price-chart",
                style={
                    "flex": "1",
                    "marginRight": "20px",
                },
            ),
            html.Div(
                [
                    # The main slider container
                    # Flex container for slider and input
                    # Slider takes most of the space
                    html.Div(
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
                                        "margin-top": "12px",
                                        "fontSize": "16px",
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
                                    "fontSize": "16px",
                                    "fontWeight": "bold",
                                    "padding": "8px",
                                },
                            },
                        ),
                        style={
                            "flex": "1",
                            "marginRight": "40px",
                            "marginLeft": "20px",
                            "marginBottom": "20px",
                        },
                    ),
                    # Input and button in a vertical arrangement
                    html.Div(
                        [
                            dcc.Input(
                                id="timestamp-input",
                                type="number",
                                placeholder="Enter timestamp...",
                                min=min(timestamps),
                                max=max(timestamps),
                                step=100,  # Set step to 100
                                style={
                                    "marginRight": "10px",
                                    "padding": "8px",
                                    "width": "125px",
                                },
                            ),
                            html.Button(
                                "Go",
                                id="timestamp-button",
                                style={
                                    "padding": "8px 15px",
                                    "backgroundColor": "#4CAF50",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "6px",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "marginBottom": "30px",
                    "marginRight": "30px",
                    "marginTop": "25px",
                },
            ),
            html.Div(
                [
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
                            html.H3(
                                "Order Book Statistics", style={"textAlign": "center"}
                            ),
                            html.Div(
                                dash_table.DataTable(
                                    id="ob-stats-table",
                                    columns=[
                                        {"name": "Name", "id": "Name"},
                                        {
                                            "name": "Value",
                                            "id": "Value",
                                            "type": "numeric",
                                        },
                                    ],
                                    style_table={"width": "200px"},
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
                            html.H3("Trades", style={"textAlign": "center"}),
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
                            "marginRight": "10px",
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
                                    style_table={"width": "200px"},
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
                            "marginRight": "10px",
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

    return layout
