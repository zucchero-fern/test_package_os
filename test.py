import numpy as np
import pandas as pd

def rank(df):
    return df.rank(axis=1, pct=True)

def ts_std(df, window):
    return df.rolling(window).std()

def ts_argmax(df, window):
    return df.rolling(window).apply(np.argmax) + 1

def ts_delta(df, period):
    return df.diff(period)

def ts_corr(x, y, window):
    return x.rolling(window).corr(y)

def ts_sum(df, window):
    return df.rolling(window).sum()

def ts_mean(df, window):
    return df.rolling(window).mean()

def ts_min(df, window):
    return df.rolling(window).min()

def ts_max(df, window):
    return df.rolling(window).max()

def ts_rank(df, window):
    return df.rolling(window).apply(lambda x: x.rank().iloc[-1])

def ts_weighted_mean(df, window):
    weights = np.arange(1, window + 1)
    return df.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def ts_lag(df, lag):
    return df.shift(lag)

def ts_product(df, window):
    return df.rolling(window).apply(np.prod)

def log(df):
    return np.log1p(df)

def sign(df):
    return np.sign(df)

def power(df, exp):
    return df.pow(exp)

def alpha001(close, returns):
    transformed = np.where(returns < 0, ts_std(returns, 20), close)
    return rank(ts_argmax(np.power(transformed, 2), 5))

def alpha002(open_, close, volume):
    s1 = rank(ts_delta(log(volume), 2))
    s2 = rank((close - open_) / open_)
    return -ts_corr(s1, s2, 6)

def alpha003(open_, volume):
    return -ts_corr(rank(open_), rank(volume), 10)

def alpha004(low):
    return -ts_rank(rank(low), 9)

def alpha005(open_, vwap, close):
    return rank(open_ - ts_mean(vwap, 10)) * -abs(rank(close - vwap))

def alpha006(open_, volume):
    return -ts_corr(open_, volume, 10)

def alpha007(close, volume, adv20):
    delta7 = ts_delta(close, 7)
    return (-ts_rank(abs(delta7), 60) * sign(delta7)).where(adv20 < volume, -1)

def alpha008(open_, returns):
    return -rank((ts_sum(open_, 5) * ts_sum(returns, 5)) - ts_lag((ts_sum(open_, 5) * ts_sum(returns, 5)), 10))

def alpha009(close):
    close_diff = ts_delta(close, 1)
    return close_diff.where(ts_min(close_diff, 5) > 0, close_diff.where(ts_max(close_diff, 5) < 0, -close_diff))

def alpha010(close):
    close_diff = ts_delta(close, 1)
    return rank(close_diff.where(ts_min(close_diff, 4) > 0, close_diff.where(ts_max(close_diff, 4) < 0, -close_diff)))

def alpha012(volume, close):
    return sign(ts_delta(volume, 1)) * -ts_delta(close, 1)

def alpha013(close, volume):
    return -rank(ts_corr(rank(close), rank(volume), 5))

def alpha014(open_, volume, returns):
    return -rank(ts_delta(returns, 3)) * ts_corr(open_, volume, 10)

def alpha015(high, volume):
    return -ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3)

def alpha016(high, volume):
    return -rank(ts_cov(rank(high), rank(volume), 5))

def alpha017(close, volume):
    adv20 = ts_mean(volume, 20)
    return (-rank(ts_rank(close, 10)) * rank(ts_delta(ts_delta(close, 1), 1)) * rank(ts_rank(volume / adv20, 5)))

def alpha018(open_, close):
    return -rank(ts_std(abs(close - open_), 5) + (close - open_) + ts_corr(close, open_, 10))

def alpha019(close, returns):
    return -sign((close - ts_lag(close, 7)) + ts_delta(close, 7)) * (1 + rank(1 + ts_sum(returns, 250)))

def alpha020(open_, high, low, close):
    return -rank(open_ - ts_lag(high, 1)) * rank(open_ - ts_lag(close, 1)) * rank(open_ - ts_lag(low, 1))

def alpha021(close, volume):
    sma2 = ts_mean(close, 2)
    sma8 = ts_mean(close, 8)
    std8 = ts_std(close, 8)
    cond_1 = sma8 + std8 < sma2
    cond_2 = sma8 - std8 > sma2
    cond_3 = volume / ts_mean(volume, 20) < 1
    return pd.DataFrame(np.select([cond_1, cond_2, cond_3], [-1, 1, -1], default=1), index=close.index, columns=close.columns)

def alpha022(high, close, volume):
    return -ts_delta(ts_corr(high, volume, 5), 5) * rank(ts_std(close, 20))

def alpha023(high, close):
    return ts_delta(high, 2).mul(-1).where(ts_mean(high, 20) < high, 0)

def alpha024(close):
    cond = ts_delta(ts_mean(close, 100), 100) / ts_lag(close, 100) <= 0.05
    return close.sub(ts_min(close, 100)).mul(-1).where(cond, -ts_delta(close, 3))

def alpha025(high, close, returns, vwap, adv20):
    return rank(-returns * adv20 * vwap * (high - close))

def alpha026(high, volume):
    return -ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

def alpha027(volume, vwap):
    cond = rank(ts_mean(ts_corr(rank(volume), rank(vwap), 6), 2))
    return cond.notnull().astype(float).where(cond <= 0.5, -1)

def alpha028(high, low, close, vwap, adv20):
    return scale(ts_corr(adv20, low, 5).replace([-np.inf, np.inf], 0) + (high + low) / 2 - close)

def alpha029(close, returns):
    return ts_min(rank(rank(scale(log(ts_sum(rank(rank(-rank(ts_delta(close - 1, 5)))), 2))))), 5) + ts_rank(ts_lag(-returns, 6), 5)

def alpha030(close, volume):
    close_diff = ts_delta(close, 1)
    return rank(sign(close_diff) + sign(ts_lag(close_diff, 1)) + sign(ts_lag(close_diff, 2))).mul(-1).add(1).mul(ts_sum(volume, 5)).div(ts_sum(volume, 20))
def alpha031(low, close, adv20):
    return (rank(rank(rank(ts_weighted_mean(rank(rank(ts_delta(close, 10))).mul(-1), 10)))) +
            rank(ts_delta(close, 3).mul(-1)) + sign(scale(ts_corr(adv20, low, 12))))

def alpha032(close, vwap):
    return scale(ts_mean(close, 7) - close) + (20 * scale(ts_corr(vwap, ts_lag(close, 5), 230)))

def alpha033(open_, close):
    return rank(-(1 - (open_ / close)))

def alpha034(close, returns):
    return rank(((1 - rank((ts_std(returns, 2) / ts_std(returns, 5)))) + (1 - rank(ts_delta(close, 1)))))

def alpha035(high, low, close, volume, returns):
    return (ts_rank(volume, 32) * (1 - ts_rank(((close + high) - low), 16)) * (1 - ts_rank(returns, 32)))

def alpha036(open_, close, volume, returns, adv20):
    return (rank(ts_corr((close - open_), ts_lag(volume, 1), 15)) * 2.21 +
            rank((open_ - close)) * 0.7 +
            rank(ts_rank(ts_lag(-returns, 6), 5)) * 0.73 +
            rank(abs(ts_corr(vwap, adv20, 6))) +
            rank(((ts_mean(close, 200) - open_) * (close - open_))) * 0.6)

def alpha037(open_, close):
    return (rank(ts_corr(ts_lag(open_ - close, 1), close, 200)) + rank(open_ - close))

def alpha038(open_, close):
    return (-1 * rank(ts_rank(close, 10)) * rank(close / open_))

def alpha039(close, volume, returns, adv20):
    return (-rank(ts_delta(close, 7) * (1 - rank(ts_weighted_mean(volume / adv20, 9)))) *
            (1 + rank(ts_sum(returns, 250))))

def alpha040(high, volume):
    return (-1 * rank(ts_std(high, 10)) * ts_corr(high, volume, 10))

def alpha041(high, low, vwap):
    return power(high * low, 0.5) - vwap

def alpha042(close, vwap):
    return rank(vwap - close) / rank(vwap + close)

def alpha043(close, volume, adv20):
    return ts_rank(volume / adv20, 20) * ts_rank(-ts_delta(close, 7), 8)

def alpha044(high, volume):
    return -ts_corr(high, rank(volume), 5)

def alpha045(close, volume):
    return -rank(ts_mean(ts_lag(close, 5), 20) * ts_corr(close, volume, 2) * rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2)))

def alpha046(close):
    cond = ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10
    return cond.where(cond > 0.25, -cond.where(cond < 0, -ts_delta(close, 1)))

def alpha047(high, close, volume, vwap, adv20):
    return rank(close.pow(-1)).mul(volume).div(adv20) * (high * rank(high - close) / ts_sum(high, 5) - rank(vwap - ts_lag(vwap, 5)))

def alpha048(close, industry):
    return ts_corr(ts_delta(close, 1), ts_delta(ts_lag(close, 1), 1), 250) * ts_delta(close, 1) / close

def alpha049(close):
    cond = ts_delta(ts_lag(close, 10), 10) / 10 - ts_delta(close, 10) / 10 >= -0.1 * close
    return -ts_delta(close, 1).where(cond, 1)

def alpha050(volume, vwap):
    return -ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)

def alpha051(close):
    cond = ts_delta(ts_lag(close, 10), 10) / 10 - ts_delta(close, 10) / 10 >= -0.05 * close
    return -ts_delta(close, 1).where(cond, 1)

def alpha052(low, volume, returns):
    return (ts_delta(ts_min(low, 5), 5) * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * ts_rank(volume, 5))

def alpha053(high, low, close):
    return -ts_delta(1 - (high - close) / (close - low), 9)

def alpha054(open_, high, low, close):
    return -((low - close) * power(open_, 5) / ((low - high) * power(close, 5)))

def alpha055(high, low, close, volume):
    return -ts_corr(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)

def alpha056(returns, cap):
    return -rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * rank(returns * cap)

def alpha057(close, vwap):
    return -(close - vwap) / ts_weighted_mean(rank(ts_argmax(close, 30)), 2)

def alpha058(volume, vwap, sector):
    return -ts_rank(ts_weighted_mean(ts_corr(indneutralize(vwap, sector), volume, 3), 7), 5)

def alpha059(volume, vwap, industry):
    return -ts_rank(ts_weighted_mean(ts_corr(indneutralize(vwap, industry), volume, 4), 16), 8)

def alpha060(low, high, close, volume):
    return -((2 * scale(rank(((close - low) - (high - close)) / (high - low) * volume))) - scale(rank(ts_argmax(close, 10))))

def alpha101(open_, high, low, close):
    return (close - open_) / ((high - low) + 0.001)

def alpha061(vwap, volume):
    return rank((vwap - ts_min(vwap, 16))) < rank(ts_corr(vwap, ts_mean(volume, 180), 18))

def alpha062(open_, high, low, vwap, adv20):
    return (rank(ts_corr(vwap, ts_sum(adv20, 22), 9)) < rank(rank(open_) + rank(open_) < rank((high + low) / 2) + rank(high))) * -1

def alpha063(vwap, adv180, industry):
    return (rank(ts_weighted_mean(ts_delta(indneutralize(vwap, industry), 2), 8)) - rank(ts_weighted_mean(ts_corr((vwap * 0.318108) + (open_ * (1 - 0.318108)), ts_sum(adv180, 37), 13), 12))) * -1

def alpha064(open_, high, low, vwap, adv120):
    return (rank(ts_corr(ts_sum((open_ * 0.178404) + (low * (1 - 0.178404)), 12), ts_sum(adv120, 12), 16)) < rank(ts_delta(((high + low) / 2) * 0.178404 + (vwap * (1 - 0.178404)), 3))) * -1

def alpha065(open_, vwap, adv60):
    return (rank(ts_corr(((open_ * 0.00817205) + (vwap * (1 - 0.00817205))), ts_sum(adv60, 9), 6)) < rank(open_ - ts_min(open_, 13))) * -1

def alpha066(low, high, vwap):
    return (rank(ts_weighted_mean(ts_delta(vwap, 4), 7)) + ts_rank(ts_weighted_mean(((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open_ - ((high + low) / 2)), 11), 7)) * -1

def alpha067(high, industry, subindustry):
    return power(rank(high - ts_min(high, 2)), rank(ts_corr(indneutralize(vwap, industry), indneutralize(adv20, subindustry), 6))) * -1

def alpha068(high, close, adv15):
    return (ts_rank(ts_corr(rank(high), rank(adv15), 9), 14) < rank(ts_delta((close * 0.518371) + (low * (1 - 0.518371)), 1))) * -1

def alpha069(close, vwap, industry, adv20):
    return (power(rank(ts_max(ts_delta(indneutralize(vwap, industry), 2), 5)), ts_rank(ts_corr((close * 0.490655) + (vwap * (1 - 0.490655)), adv20, 5), 9))) * -1

def alpha070(open_, close, industry):
    return (power(rank(ts_delta(vwap, 1)), ts_rank(ts_corr(indneutralize(close, industry), adv50, 18), 18))) * -1

def alpha070(open_, close, industry):
    return (power(rank(ts_delta(vwap, 1)), ts_rank(ts_corr(indneutralize(close, industry), adv50, 18), 18))) * -1

def alpha071(open_, close, volume, vwap):
    return max(ts_rank(ts_weighted_mean(ts_corr(ts_rank(close, 3), ts_rank(volume, 12), 18), 4), 16), ts_rank(ts_weighted_mean(rank(((low + open_) - (vwap + vwap))) ** 2, 16), 4))

def alpha072(high, low, volume, vwap):
    return rank(ts_weighted_mean(ts_corr((high + low) / 2, adv40, 9), 10)) / rank(ts_weighted_mean(ts_corr(ts_rank(vwap, 3), ts_rank(volume, 18), 6), 2))

def alpha073(low, vwap):
    w = 0.147155
    s1 = rank(ts_weighted_mean(ts_delta(vwap, 5), 3))
    s2 = ts_rank(ts_weighted_mean(ts_delta((open_ * w) + (low * (1 - w)), 2) / ((open_ * w) + (low * (1 - w))) * -1, 3), 16)
    return s1.where(s1 > s2, s2).mul(-1)

def alpha074(close, volume, vwap):
    w = 0.0261661
    return rank(ts_corr(close, ts_sum(adv30, 37), 15)) < rank(ts_corr(rank((high * w) + (vwap * (1 - w))), rank(volume), 11)) * -1

def alpha075(low, volume, vwap):
    return rank(ts_corr(vwap, volume, 4)) < rank(ts_corr(rank(low), rank(adv50), 12))

def alpha076(low, vwap, sector):
    return max(rank(ts_weighted_mean(ts_delta(vwap, 1.24383), 11)), ts_rank(ts_weighted_mean(ts_rank(ts_corr(indneutralize(low, sector), adv81, 8), 19), 17), 17))

def alpha077(high, low, vwap):
    s1 = rank(ts_weighted_mean(((((high + low) / 2) + high) - (vwap + high)), 20))
    s2 = rank(ts_weighted_mean(ts_corr((high + low) / 2, adv40, 3), 6))
    return s1.where(s1 < s2, s2)

def alpha078(low, volume, vwap):
    w = 0.352233
    return rank(ts_corr(ts_sum((low * w) + (vwap * (1 - w)), 19), ts_sum(adv40, 19), 6)) ** rank(ts_corr(rank(vwap), rank(volume), 5))

def alpha079(open_, volume, sector):
    return rank(ts_delta(indneutralize((close * 0.60733) + (open_ * (1 - 0.60733)), sector), 1)) < rank(ts_corr(ts_rank(vwap, 3), ts_rank(adv150, 9), 14))

def alpha080(high, industry):
    return power(rank(sign(ts_delta(indneutralize(((open_ * 0.868128) + (high * (1 - 0.868128))), industry), 4))), ts_rank(ts_corr(high, adv10, 5), 5)) * -1

def alpha081(v, vwap):
    return (rank(log(ts_product(rank(rank(ts_corr(vwap, ts_sum(ts_mean(v, 10), 50), 8)).pow(4)), 15)))
            .lt(rank(ts_corr(rank(vwap), rank(v), 5)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha083(h, l, c):
    s = h.sub(l).div(ts_mean(c, 5))
    return (rank(rank(ts_lag(s, 2))
                 .mul(rank(rank(v)))
                 .div(s).div(vwap.sub(c).add(1e-3)))
            .stack('ticker')
            .swaplevel()
            .replace((np.inf, -np.inf), np.nan))

def alpha084(c, vwap):
    return (rank(power(ts_rank(vwap.sub(ts_max(vwap, 15)), 20),
                       ts_delta(c, 6)))
            .stack('ticker')
            .swaplevel())

def alpha085(h, c, v):
    w = 0.876703
    return (rank(ts_corr(h.mul(w).add(c.mul(1 - w)), ts_mean(v, 30), 10))
            .pow(rank(ts_corr(ts_rank(h.add(l).div(2), 4),
                              ts_rank(v, 10), 7)))
            .stack('ticker')
            .swaplevel())

def alpha086(c, v, vwap):
    return (ts_rank(ts_corr(c, ts_mean(ts_mean(v, 20), 15), 6), 20)
            .lt(rank(c.sub(vwap)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha088(o, h, l, c, v):
    s1 = (rank(ts_weighted_mean(rank(o)
                                .add(rank(l))
                                .sub(rank(h))
                                .add(rank(c)), 8)))
    s2 = ts_rank(ts_weighted_mean(ts_corr(ts_rank(c, 8),
                                          ts_rank(ts_mean(v, 60), 20), 8), 6), 2)
    return (s1.where(s1 < s2, s2)
            .stack('ticker')
            .swaplevel())

def alpha092(o, l, c, v):
    p1 = ts_rank(ts_weighted_mean(h.add(l).div(2).add(c).lt(l.add(o)), 15), 18)
    p2 = ts_rank(ts_weighted_mean(ts_corr(rank(l), rank(ts_mean(v, 30)), 7), 6), 6)
    return (p1.where(p1 < p2, p2)
            .stack('ticker')
            .swaplevel())

def alpha094(v, vwap):
    return (rank(vwap.sub(ts_min(vwap, 11)))
            .pow(ts_rank(ts_corr(ts_rank(vwap, 20),
                                 ts_rank(ts_mean(v, 60), 4), 18), 2))
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha095(o, l, v):
    return (rank(o.sub(ts_min(o, 12)))
            .lt(ts_rank(rank(ts_corr(ts_mean(h.add(l).div(2), 19),
                                     ts_sum(ts_mean(v, 40), 19), 13).pow(5)), 12))
            .astype(int)
            .stack('ticker')
            .swaplevel())

def alpha096(c, v, vwap):
    s1 = ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(v), 10), 4), 8)
    s2 = ts_rank(ts_weighted_mean(ts_argmax(ts_corr(ts_rank(c, 7),
                                                    ts_rank(ts_mean(v, 60), 10), 10), 12), 14), 13)
    return (s1.where(s1 > s2, s2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())

def alpha098(o, v, vwap):
    adv5 = ts_mean(v, 5)
    adv15 = ts_mean(v, 15)
    return (rank(ts_weighted_mean(ts_corr(vwap, ts_mean(adv5, 26), 4), 7))
            .sub(rank(ts_weighted_mean(ts_rank(ts_argmin(ts_corr(rank(o),
                                                                 rank(adv15), 20), 8), 6))))
            .stack('ticker')
            .swaplevel())

def alpha099(l, v):
    return ((rank(ts_corr(ts_sum((h.add(l).div(2)), 19),
                          ts_sum(ts_mean(v, 60), 19), 8))
             .lt(rank(ts_corr(l, v, 6)))
             .mul(-1))
            .stack('ticker')
            .swaplevel())

def alpha100(r, cap):
    pass

def alpha100(open_, high, low, close):
    return (close - open_) / ((high - low) + 0.001)

def alpha101(open_, high, low, close):
    return (close - open_) / ((high - low) + 0.001)

# Define all variables at the end
o = data['open']
h = data['high']
l = data['low']
c = data['close']
v = data['volume']
vwap = (o + h + l + c) / 4
adv20 = v.rolling(20).mean()
r = data['returns']
cap = data['market_cap']

# Example usage:
# alphas = {f'alpha{i:03}': globals()[f'alpha{i:03}'](o, h, l, c, v, vwap, adv20, r, cap) for i in range(1, 102)}
