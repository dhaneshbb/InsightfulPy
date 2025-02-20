import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate
import textwrap

def calc_stats(data):
    return {
        'Count': data.count(),
        'Mean': data.mean(),
        'Trimmed Mean': iqr_trimmed_mean(data),
        'MAD': mad(data),
        'Std': data.std(),
        'Min': data.min(),
        '25%': data.quantile(0.25),
        '50%': data.median(),
        '75%': data.quantile(0.75),
        'Max': data.max(),
        'Mode': data.mode()[0] if not data.mode().empty else 'N/A',
        'Range': data.max() - data.min(),
        'IQR': data.quantile(0.75) - data.quantile(0.25),
        'Variance': data.var(),
        'Skewness': data.skew(),
        'Kurtosis': data.kurt()
    }


def iqr_trimmed_mean(data):
    q1, q3 = np.percentile(data.dropna(), [25, 75])
    iqr = q3 - q1
    return data[(data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)].mean()


def mad(data):
    return np.mean(np.abs(data - data.mean()))

