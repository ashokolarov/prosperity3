from io import StringIO
import json

import pandas as pd


def is_integer(string):
    if string.startswith("-"):
        return string[1:].isdigit()
    else:
        return string.isdigit()


def parse_sandbox_logs(file_path: str):
    """Parse the 'Sandbox logs' section from the file into a list of JSON objects."""
    sandbox_logs = []
    current_obj = ""
    in_sandbox_section = False

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "Sandbox logs:":
                in_sandbox_section = True
                continue
            elif line in ["Activities log:", "Trade History:"]:
                in_sandbox_section = False
                if current_obj:
                    sandbox_logs.append(json.loads(current_obj))
                current_obj = ""
                break

            if in_sandbox_section:
                if line.startswith("{"):
                    if current_obj:
                        sandbox_logs.append(json.loads(current_obj))
                    current_obj = line
                elif line.startswith("}"):
                    current_obj += line
                elif line:
                    current_obj += line

    if current_obj:  # Append the last object
        sandbox_logs.append(json.loads(current_obj))

    return sandbox_logs


def extract_trader_data(sandbox_logs):
    """Extract trader-level data from sandbox logs."""
    trader_data = []

    for entry in sandbox_logs:
        lines = entry["lambdaLog"].split("\n")
        current_trader = {}
        record = False

        for line in lines:
            line = line.strip()
            if line == "TRADER_BEGIN":
                record = True
                continue
            elif line == "TRADER_END":
                record = False
                trader_data.append(current_trader)
                continue

            if record:
                key, value = line.split(" ")
                current_trader[key] = int(value) if is_integer(value) else float(value)

    return pd.DataFrame(trader_data)


def extract_product_data(sandbox_logs):
    products_data = {}
    for entry in sandbox_logs:
        lines = entry["lambdaLog"].split("\n")
        current_product = {}
        record = False

        for line in lines:
            line = line.strip()
            if line.startswith("PRODUCT_BEGIN"):
                record = True
                current_product["product"] = line.split(" ")[1]
                current_product["orders"] = []
                continue
            elif line.startswith("PRODUCT_END"):
                record = False
                product = current_product["product"]
                if product in products_data:
                    products_data[product].append(current_product)
                else:
                    products_data[product] = [current_product]
                continue

            if record:
                key, value = line.split(" ")

                if key == "order":
                    price, volume = value.split("@")
                    current_product["orders"].append([int(price), int(volume)])
                else:
                    current_product[key] = (
                        int(value) if is_integer(value) else float(value)
                    )

    for product in products_data:
        products_data[product] = pd.DataFrame(products_data[product])

    return products_data


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
    sanbox_logs = parse_sandbox_logs(file_path)
    trader_data = extract_trader_data(sanbox_logs)
    products_data = extract_product_data(sanbox_logs)
    activities, trades = extract_book_and_trades(file_path)
    return trader_data, products_data, activities, trades


if __name__ == "__main__":
    log_file = "backtests/example.log"

    trader_data, products_data, activities, trades = process_log(log_file)

    print("AAA")
