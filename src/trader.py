from time import time

import jsonpickle

from autils import CustomLogger
from datamodel import TradingState
from products import (
    Croissants,
    Djembes,
    Jams,
    Kelp,
    PicnicBasket1,
    PicnicBasket2,
    RainforestResin,
    Squid,
    SyntheticBasket1,
    SyntheticBasket2,
    Volcanic9500,
    Volcanic9750,
    Volcanic10000,
    Volcanic10250,
    Volcanic10500,
    VolcanicRock,
)

config_rainforest = {
    # Market taking parameters
    "mt_take_edge": 1,
    "mt_profit_margin": 1,
    # Market making parameters
    "mm_default_vol": 15,
    "mm_default_edge": 4,
    "mm_disregard_edge": 1,
    "mm_join_edge": 2,
    "mm_join_volume": 1,
    "mm_join_edge_2": 3,
    "mm_join_volume_2": 1,
    "mm_constrain_below_fair": True,
    "mm_manage_position": False,
}

config_kelp = {
    # General
    "detect_mm_volume": 15,  # Volume to detect market maker
    # Market taking parameters
    "mt_take_edge": 1,
    "mt_profit_margin": 0.5,
    "mt_adverse_volume": 15,  # Maximum mt volume
    # Market making parameters
    "mm_default_vol": 20,
    "mm_default_edge": 1,
    "mm_disregard_edge": 1,
    "mm_join_edge": 2,
    "mm_join_volume": 3,
    "mm_constrain_below_fair": True,
    "mm_manage_position": True,
}

config_squid = {
    # General
    "detect_mm_volume": 15,  # Volume to detect market maker
    # Price estimation
    "short_window": 70,
    "long_window": 510,
    "std_window": 290,
    # Directional parameters
    "dt_default_vol": 10,
    "dt_threshold_z": 0.9,
    "z_close_threshold": 0.15,
    "price_drop_threshold": 3,
    "recovery_wait_period": 5,
}

config_croissants = {}

config_jams = {}

config_djembes = {}

config_picnic_basket_1 = {
    "detect_mm_volume": 20,  # Volume to detect market maker
    # Market making parameters
    "market_making": False,
    "mm_default_vol": 10,
    "mm_default_edge": 4,
    "mm_disregard_edge": 2,
    "mm_join_edge": 6,
    "mm_join_volume": 5,
    "mm_constrain_below_fair": True,
    "mm_manage_position": False,
}

config_picnic_basket_2 = {
    "detect_mm_volume": 20,  # Volume to detect market maker
    # Market making parameters
    "market_making": False,
    "mm_default_vol": 15,
    "mm_default_edge": 4,
    "mm_disregard_edge": 1,
    "mm_join_edge": 6,
    "mm_join_volume": 5,
    "mm_constrain_below_fair": True,
    "mm_manage_position": True,
}

config_synthetic_basket_1 = {
    "N": 10,
    "buy_entry": 1.6,
    "buy_exit": 0.25,
    "sell_entry": 1.6,
    "sell_exit": 0.25,
}

config_synthetic_basket_2 = {
    "N": 100,
    "buy_entry": 1.6,
    "buy_exit": 0.2,
    "sell_entry": 1.6,
    "sell_exit": 0.2,
}

config_volcanic = {
    # Price estimation
    "short_window": 100,
    "long_window": 500,
    "std_window": 140,
    # Directional parameters
    "dt_default_vol": 100,
    "dt_threshold_z": 0.7,
    "z_close_threshold": 0.1,
    "price_drop_threshold": 2,
    "recovery_wait_period": 5,
}

config_volcanic_9500 = {
    # Price estimation
    "short_window": 90,
    "long_window": 510,
    "std_window": 140,
    # Directional parameters
    "dt_default_vol": 100,
    "dt_threshold_z": 0.7,
    "z_close_threshold": 0.2,
    "price_drop_threshold": 2.0,
    "recovery_wait_period": 5,
}

config_volcanic_9750 = {
    # Price estimation
    "short_window": 90,
    "long_window": 510,
    "std_window": 140,
    # Directional parameters
    "dt_default_vol": 100,
    "dt_threshold_z": 0.8,
    "z_close_threshold": 0.2,
    "price_drop_threshold": 2.0,
    "recovery_wait_period": 5,
}

config_volcanic_10000 = {
    # Price estimation
    "short_window": 90,
    "long_window": 510,
    "std_window": 140,
    # Directional parameters
    "dt_default_vol": 100,
    "dt_threshold_z": 0.8,
    "z_close_threshold": 0.2,
    "price_drop_threshold": 2.0,
    "recovery_wait_period": 5,
}

config_volcanic_10250 = {
    # Price estimation
    "short_window": 90,
    "long_window": 500,
    "std_window": 140,
    # Directional parameters
    "dt_default_vol": 100,
    "dt_threshold_z": 0.8,
    "z_close_threshold": 0.2,
    "price_drop_threshold": 2.0,
    "recovery_wait_period": 5,
}

config_volcanic_10500 = {
    # Price estimation
    "short_window": 90,
    "long_window": 500,
    "std_window": 140,
    # Directional parameters
    "dt_default_vol": 100,
    "dt_threshold_z": 0.8,
    "z_close_threshold": 0.2,
    "price_drop_threshold": 2.0,
    "recovery_wait_period": 5,
}


class Trader:
    def run(self, state: TradingState):
        logger = CustomLogger()

        t1 = time()

        logger.print("TRADER_B")
        timestamp = state.timestamp
        logger.print_numeric("timestamp", timestamp)

        PAIRS_PRODUCTS = [
            "CROISSANTS",
            "JAMS",
            "DJEMBES",
            "PICNIC_BASKET1",
            "PICNIC_BASKET2",
        ]

        result = {}
        if not state.traderData:
            # -------------------Initialize Products-------------------
            products = {}
            products["RAINFOREST_RESIN"] = RainforestResin(config_rainforest)
            products["KELP"] = Kelp(config_kelp)
            products["SQUID_INK"] = Squid(config_squid)
            products["CROISSANTS"] = Croissants(config_croissants)
            products["JAMS"] = Jams(config_jams)
            products["DJEMBES"] = Djembes(config_djembes)
            products["PICNIC_BASKET1"] = PicnicBasket1(config_picnic_basket_1)
            products["PICNIC_BASKET2"] = PicnicBasket2(config_picnic_basket_2)
            products["VOLCANIC_ROCK"] = VolcanicRock(config_volcanic)
            products["VOLCANIC_ROCK_VOUCHER_9500"] = Volcanic9500(config_volcanic_9500)
            products["VOLCANIC_ROCK_VOUCHER_9750"] = Volcanic9750(config_volcanic_9750)
            products["VOLCANIC_ROCK_VOUCHER_10000"] = Volcanic10000(
                config_volcanic_10000
            )
            products["VOLCANIC_ROCK_VOUCHER_10250"] = Volcanic10250(
                config_volcanic_10250
            )
            products["VOLCANIC_ROCK_VOUCHER_10500"] = Volcanic10500(
                config_volcanic_10500
            )
            # ------------------Synthetic Products-------------------
            synthetic = {}
            synthetic["SYNTHETIC_BASKET1"] = SyntheticBasket1(config_synthetic_basket_1)
            synthetic["SYNTHETIC_BASKET2"] = SyntheticBasket2(config_synthetic_basket_2)
        else:
            traderData = jsonpickle.decode(state.traderData)
            products = traderData["products"]
            synthetic = traderData["synthetic"]

        for product in state.order_depths:
            if product in products.keys():
                order_depth = state.order_depths[product]
                if product in state.position:
                    position = state.position[product]
                else:
                    position = 0

                if product in state.own_trades:
                    own_trades = state.own_trades[product]
                else:
                    own_trades = []

                products[product].update_product(
                    order_depth, position, own_trades, timestamp
                )

                if product not in PAIRS_PRODUCTS:
                    products[product].calculate_orders()

        for product in synthetic.keys():
            synthetic[product].calculate_orders(products, timestamp)

        for product in PAIRS_PRODUCTS:
            if product in products.keys():
                products[product].calculate_orders()

        for product in state.order_depths:
            if product in products.keys():
                result[product] = products[product].on_timestep_end()

        traderData = dict()
        traderData["products"] = products
        traderData["synthetic"] = synthetic
        traderData = jsonpickle.encode(traderData)

        conversions = 1

        t2 = time()

        logger.print_numeric("runtime", t2 - t1)
        logger.print("TRADER_E")
        logger.flush()

        return result, conversions, traderData
