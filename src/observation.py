# observations functions
import pandas as pd
import numpy as np


def low(df_current: pd.DataFrame) -> np.ndarray:
    return np.array(df_current["Low"])


def high(df_current: pd.DataFrame) -> np.ndarray:
    return np.array(df_current["High"])


def close(df_current: pd.DataFrame) -> np.ndarray:
    return np.array(df_current["Close"])


def volume(df_current: pd.DataFrame) -> np.ndarray:
    return np.array(df_current["Volume"])


def dividends(df_current: pd.DataFrame) -> np.ndarray:
    return np.array(df_current["Dividends"])


def stock_splits(df_current: pd.DataFrame) -> np.ndarray:
    return np.array(df_current["Stock Splits"])


def sma(df_current: pd.DataFrame) -> np.ndarray:
    return np.array(df_current["SMA"])


def ema(df_current: pd.DataFrame) -> np.ndarray:
    return np.array(df_current["EMA"])
