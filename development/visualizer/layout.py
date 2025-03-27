from dash import dash_table, dcc, html


def get_layout(products, timestamps):
    layout = html.Div(
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
            html.Div(
                [
                    # The main slider container
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
                            # Flex container for slider and input
                            html.Div(
                                [
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
                                                        "white-space": "nowrap",
                                                        "margin-top": "10px",
                                                    },
                                                }
                                                for ts in range(
                                                    min(timestamps),
                                                    max(timestamps) + 1,
                                                    (max(timestamps) - min(timestamps))
                                                    // 10,
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
                                        style={"flex": "1", "marginRight": "20px"},
                                    ),
                                    # Input and button in a vertical arrangement
                                    html.Div(
                                        [
                                            html.Label(
                                                "Exact Timestamp:",
                                                style={
                                                    "marginBottom": "5px",
                                                    "display": "block",
                                                },
                                            ),
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
                                                            "width": "150px",
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
                                                            "borderRadius": "4px",
                                                            "cursor": "pointer",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                },
                                            ),
                                        ],
                                        style={"width": "220px"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "flex-end",
                                    "justifyContent": "space-between",
                                    "marginBottom": "30px",
                                },
                            ),
                        ],
                    ),
                ],
                style={
                    "width": "98%",
                    "marginBottom": "30px",
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

    return layout
