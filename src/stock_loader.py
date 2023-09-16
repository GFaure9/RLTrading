import pandas as pd
import yfinance as yf

from typing import Dict, List


def load_stock(ticker_name: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
    ticker = yf.Ticker(ticker_name)
    df = ticker.history(start=start_date, end=end_date, **kwargs)
    return df


def get_data_stocks(
        list_ticker_names: List[str],
        start_date: str,
        end_date: str,
        **kwargs
) -> Dict[str, pd.DataFrame]:
    data_stocks = {}
    for name in list_ticker_names:
        data_stocks[name] = load_stock(
            ticker_name=name, start_date=start_date, end_date=end_date, **kwargs
        )
    return data_stocks


if __name__ == "__main__":
    stocks = get_data_stocks(
        ["TSLA", "AAPL", "GOOGL"],
        "2020-01-10",
        "2023-01-09",
    )
    print(stocks["TSLA"].columns)
