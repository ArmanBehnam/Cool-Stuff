# -*- coding: utf-8 -*-

import numpy as np


def moving_average(x, n, type='simple'):
    """
    compute an n period moving average.
    type is 'simple' | 'exponential'
    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a


def relative_strength(prices, n=14):
    """
    compute the n period relative strength indicator
    http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http://www.investopedia.com/terms/r/rsi.asp
    """

    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100. / (1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi


def moving_average_convergence(x, nslow=26, nfast=12, nsign=9, simple=False):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and
    slow exponential moving avg'
    """

    macd_dict = {}
    macd_dict['fast'] = moving_average(x, nfast, type='exponential')
    macd_dict['slow'] = moving_average(x, nslow, type='exponential')
    macd_dict['macd'] = map(lambda f, s: round(f - s, 5), macd_dict['fast'],
                            macd_dict['slow'])
    macd_dict['sign'] = moving_average(macd_dict['macd'], nsign)
    if not simple:
        return macd_dict
    else:
        return macd_dict['macd']
