# Wolves of Muurstraat

This repository contains all the research and algorithms developed by our team for the 3rd edition of IMC's Prosperity algorithmic trading competitions. Due to personal circumstances within the team, we could not participate during all five rounds of the competition, but nevertheless, enjoyed it thoroughly and ranked 13th out of more than 12000 teams during the first three rounds. 

## Prosperity 3
IMC Prosperity 3 was a global algorithmic trading competition that brought together over 12,000 teams from around the world. The competition ran over five rounds and fifteen days, simulating a dynamic multi-product marketplace where teams designed and submitted trading algorithms to maximize profit and loss (PnL).

Each round introduced new synthetic products, each with its own behavior and market structure. Some assets encouraged market making with predictable price movements and steady spreads, while others required aggressive market taking to capture short-term opportunities. A few products were specifically designed to enable pairs trading, cross-product arbitrage, or ETF-style decomposition, rewarding strategies that could identify and exploit statistical relationships across assets.

In addition to algorithmic trading, each round featured a small-scale manual trading challenge, typically focused on game theory, optimization under uncertainty, or decision-making based on limited information. While these contributed less to overall PnL, they added a fun and intellectually engaging element to the competition.

For full details on the product mechanics, simulation environment, and round structure, see the Prosperity 3 Wiki. see the [Prosperity 3 Wiki](https://imc-prosperity.notion.site/Prosperity-3-Wiki-19ee8453a09380529731c4e6fb697ea4)

## Organization
We include all of work including a trading dashboard developed for visualizing positions, PnL, and market activity; a comprehensive set of research notebooks and analysis tools in the development folder that document our strategy development, data exploration, and backtesting results; and the core trading algorithms in the src directory, which implement our trading and risk management logic across multiple products.

Instead of relying heavily on open-source tools, which many successful teams did, we chose to build our own infrastructure from scratch. This allowed us to tailor everything to our specific workflow and needs, but helped us learn a lot in the process. One of the core components we developed was a custom visualization dashboard, which helped us analyze price movements, order book dynamics, and bot behavior more effectively throughout the competition.

### Dashboard

The dashboard was one of the first things we built, which allowed us to visualize and analyze the performance of our trading algorithms. It provides dynamic charts and tables for tracking positions, profit and loss (PnL), and order book data across all traded products. Users can select specific products, navigate through different timestamps, and view detailed metrics such as executed trades, order flow, and position limits. It was mainly used to quickly identify trends, monitor risk, and evaluate the effectiveness of your trading strategies during backtesting sessions.

![Dashboard](figures/dashboard.png)
