from io import StringIO

import pandas as pd


def extract_trader_state(file_path: str) -> pd.DataFrame:
    columns = [
        "timestamp",
        "product",
        "position",
        "mt_position",
        "mm_position",
        "orders",
    ]
    data = pd.DataFrame(columns=columns)

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith('"lambdaLog"'):
                entities = line[14:-2].split("\\n")

                # Initialize variables for each iteration
                timestamp = None
                product_name = None
                position = None
                mt_position = None
                mm_position = None
                orders = []

                for entity in entities:
                    if entity.startswith("timestamp "):
                        timestamp = int(entity[len("timestamp ") :].strip())
                    elif entity.startswith("product "):
                        product_name = entity[len("product ") :].strip()
                    elif entity.startswith("position "):
                        position = int(entity[len("position ") :].strip())
                    elif entity.startswith("mt_position "):
                        mt_position = int(entity[len("mt_position ") :].strip())
                    elif entity.startswith("mm_position "):
                        mm_position = int(entity[len("mm_position ") :].strip())
                    elif entity.startswith("order "):
                        order = entity[len("order ") :].strip()
                        price, volume = order.split(" ")
                        orders.append([int(price), int(volume)])

                # Append the row to the DataFrame
                new_row = {
                    "timestamp": timestamp,
                    "product": product_name,
                    "position": position,
                    "mt_position": mt_position,
                    "mm_position": mm_position,
                    "orders": orders,
                }

                # Method 1: Using loc with the length of the DataFrame
                data.loc[len(data)] = new_row
    data = data.astype(
        {
            "timestamp": "Int64",
            "product": str,
            "position": "Int64",
            "mt_position": "Int64",
            "mm_position": "Int64",
        }
    )
    return data


def extract_book_and_trades(file_path: str) -> list:
    """
    Extract all lines between 'Activities logs:' and 'Trader history:' from the file.
    Returns a list of lines.
    """
    activities_lines = ""
    capture_activities = False

    trades_lines = ""
    capture_trades = False

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace

            # Start capturing lines after 'Activities logs:'
            if line == "Activities log:":
                capture_activities = True
                continue  # Skip the delimiter line itself

            # Start capturing lines after 'Trade history:'
            if line == "Trade History:":
                capture_trades = True
                capture_activities = False
                continue  # Skip the delimiter line itself

            # Add the line to the list if we're in the capture mode
            if capture_activities and line:  # Only add non-empty lines
                activities_lines += line + "\n"

            # Add the line to the list if we're in the capture mode
            if capture_trades and line:  # Only add non-empty lines
                trades_lines += line

    activities = pd.read_csv(StringIO(activities_lines), sep=";", na_filter=True)
    trades = pd.read_json(StringIO(trades_lines))

    return activities, trades


def process_log(file_path: str) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trader_state = extract_trader_state(file_path)
    activities, trades = extract_book_and_trades(file_path)
    return trader_state, activities, trades
