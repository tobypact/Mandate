from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import json, os, uuid
from datetime import datetime

app = Flask(__name__)
RECORDS_FILE = "records.json"

def load_records():
    if os.path.exists(RECORDS_FILE):
        with open(RECORDS_FILE) as f:
            return json.load(f)
    return []

def save_records(r):
    with open(RECORDS_FILE, "w") as f:
        json.dump(r, f)

def compute_rsi(series, period=14):
    d = series.diff()
    g = d.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    l = (-d.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - (100 / (1 + g / l))

def run_backtest(ticker, start, end, capital, rsi_buy, rsi_sell_enabled, rsi_sell,
                 profit_target, stop_loss, buy_metrics, sell_metrics,
                 buy_logic, sell_logic, benchmark, bm_mode, extra_params=None):
    extra_params = extra_params or {}
    bm = buy_metrics or {}
    sm = sell_metrics or {}

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return None, f"No data for {ticker}"

    closes = df["Close"].squeeze()
    opens  = df["Open"].squeeze()
    rsi    = compute_rsi(closes)

    has_buy_rsi = rsi_buy is not None
    has_sell    = profit_target or stop_loss or rsi_sell_enabled or any(v is not None for v in sm.values())
    if not has_sell:
        rsi_sell_enabled = True
        rsi_sell = 70.0

    def gate(checks, logic):
        active = [c for c in checks if c is not None]
        if not active: return True
        return all(active) if logic == "and" else any(active)

    def sell_checks(pnl_pct, eq_vals):
        checks, reasons = [], []
        dd = round(abs(pnl_pct)*100, 1) if pnl_pct < 0 else 0
        gain = pnl_pct * 100
        if sm.get("maxDrawdown") is not None:
            p = bool(dd >= float(sm["maxDrawdown"]))
            checks.append(p)
            if p: reasons.append(f"DD {dd}%>={sm['maxDrawdown']}%")
        for key, lbl in [("pe","P/E"),("pb","P/B"),("eps","EPS"),("divYield","DivY"),
                          ("roe","ROE"),("de","D/E"),("fcfYield","FCF"),("revGrowth","RevG")]:
            if sm.get(key) is not None:
                p = bool(gain >= float(sm[key]))
                checks.append(p)
                if p: reasons.append(f"{lbl}+{round(gain,1)}%>={sm[key]}%")
        return checks, reasons

    trades, equity = [], []
    cash = capital
    in_trade = just_exited = False
    ep = ed = sh = None

    for i in range(1, len(df)):
        date     = df.index[i]
        t_open   = float(opens.iloc[i])
        prev_rsi = float(rsi.iloc[i-1]) if not np.isnan(rsi.iloc[i-1]) else None

        if not in_trade:
            if just_exited:
                just_exited = False
            else:
                lk  = max(0, i-252)
                wh  = float(closes.iloc[lk:i].max()) if i > lk else t_open
                dd  = round((wh-t_open)/wh*100, 1) if wh > 0 else 0
                bc  = []
                if has_buy_rsi: bc.append(bool(prev_rsi is not None and prev_rsi < rsi_buy))
                if bm.get("maxDrawdown") is not None: bc.append(bool(dd >= float(bm["maxDrawdown"])))
                for k in ["pe","pb","eps","divYield","roe","de","fcfYield","revGrowth"]:
                    if bm.get(k) is not None: bc.append(True)
                if gate(bc, buy_logic) if bc else True:
                    sh = cash / t_open; ep = t_open; ed = date; in_trade = True
        else:
            pnl = (t_open - ep) / ep
            exit_reason = None
            if stop_loss and pnl <= -(stop_loss/100):
                exit_reason = f"-{stop_loss}% stop"
            if not exit_reason:
                sc, sr = [], []
                if rsi_sell_enabled and prev_rsi is not None:
                    p = bool(prev_rsi > rsi_sell)
                    sc.append(p)
                    if p: sr.append(f"RSI>{int(rsi_sell)}({round(prev_rsi,1)})")
                if profit_target:
                    p = bool(pnl >= profit_target/100)
                    sc.append(p)
                    if p: sr.append(f"+{profit_target}% target")
                if sm:
                    mc, mr = sell_checks(pnl, [e["value"] for e in equity])
                    sc.extend(mc); sr.extend(mr)
                if sc and gate(sc, sell_logic):
                    exit_reason = " & ".join(sr[:3]) or "Sell conditions met"
            if exit_reason:
                cash = sh * t_open
                trades.append({"entryDate":str(ed)[:10],"exitDate":str(date)[:10],
                    "entryPrice":round(ep,2),"exitPrice":round(t_open,2),
                    "returnPct":round(pnl*100,2),"pnl":round(cash-sh*ep,2),"reason":exit_reason})
                in_trade = False; just_exited = True; sh = None
        equity.append({"date":str(date)[:10],"value":round(sh*t_open if (in_trade and sh) else cash,2)})

    if in_trade:
        fp  = float(closes.iloc[-1]); pnl = (fp-ep)/ep; cash = sh*fp
        trades.append({"entryDate":str(ed)[:10],"exitDate":str(df.index[-1])[:10],
            "entryPrice":round(ep,2),"exitPrice":round(fp,2),
            "returnPct":round(pnl*100,2),"pnl":round(cash-sh*ep,2),"reason":"End of period"})

    # Benchmark equity
    bm_name = benchmark or "SPY"
    bm_lbl  = f"{bm_name} Buy & Hold"
    bm_df   = yf.download(bm_name, start=start, end=end, progress=False, auto_adjust=True)
    if not bm_df.empty:
        bp = bm_df["Close"].squeeze()
        b0 = float(bp.iloc[0])
        bm_eq = [{"date":str(bm_df.index[i])[:10],"value":round(float(bp.iloc[i])/b0*capital,2)} for i in range(1,len(bm_df))]
        bm_ret = round((float(bp.iloc[-1])/b0-1)*100,1)
    else:
        c0 = float(closes.iloc[0])
        bm_eq  = [{"date":str(df.index[i])[:10],"value":round(float(closes.iloc[i])/c0*capital,2)} for i in range(1,len(df))]
        bm_ret = round((float(closes.iloc[-1])/c0-1)*100,1)

    ev = [e["value"] for e in equity]
    rets = pd.Series(ev).pct_change().dropna()
    sharpe = round((rets.mean()/rets.std())*np.sqrt(252),2) if rets.std()>0 else 0
    rm = pd.Series(ev).cummax()
    maxdd = round(((pd.Series(ev)-rm)/rm).min()*100,1)
    tr = [t["returnPct"] for t in trades]
    wins = [r for r in tr if r>0]; losses = [r for r in tr if r<=0]

    # Days spanned & annualised return
    date_start = pd.Timestamp(start); date_end = pd.Timestamp(end)
    days_spanned = (date_end - date_start).days
    years_spanned = days_spanned / 365.25
    total_ret = (ev[-1]/capital - 1)
    annualised_ret = round(((1 + total_ret) ** (1/years_spanned) - 1) * 100, 2) if years_spanned > 0 else 0

    # Time-Weighted Return (TWA) — chain-link sub-period returns
    if trades:
        sub_rets = [(1 + t["returnPct"]/100) for t in trades]
        twa = round((np.prod(sub_rets) - 1) * 100, 2)
    else:
        twa = 0.0

    # IRR-based Money-Weighted Return using daily cash flows
    # Build a daily cash flow array over the full period
    mwa = None
    irr_annual = None
    hypo_final = None
    try:
        all_dates = [str(df.index[i])[:10] for i in range(len(df))]
        date_idx  = {d:i for i,d in enumerate(all_dates)}
        n_days    = len(all_dates)
        cf_arr    = np.zeros(n_days)
        cf_arr[0] = -capital  # initial outlay

        for t in trades:
            # Cash in (buy) at entry
            entry_i = date_idx.get(t["entryDate"])
            exit_i  = date_idx.get(t["exitDate"])
            if entry_i is not None:
                shares_bought = capital / t["entryPrice"] if t["entryPrice"] > 0 else 0
                cf_arr[entry_i] -= 0  # already accounted in initial capital
            if exit_i is not None:
                cf_arr[exit_i] += t["pnl"]  # realised profit/loss at exit

        # Final portfolio value returned at end
        cf_arr[-1] += ev[-1]

        # Solve for daily IRR using Newton's method
        def npv(r, cfs):
            return sum(cf / (1+r)**i for i, cf in enumerate(cfs))
        def dnpv(r, cfs):
            return sum(-i*cf / (1+r)**(i+1) for i, cf in enumerate(cfs))

        # Bisection fallback for robustness
        r = 0.001  # initial guess: 0.1% daily
        for _ in range(200):
            n = npv(r, cf_arr)
            dn = dnpv(r, cf_arr)
            if abs(dn) < 1e-12: break
            r_new = r - n/dn
            if r_new < -0.999: r_new = -0.5
            if abs(r_new - r) < 1e-10: break
            r = r_new

        if not np.isnan(r) and r > -0.999:
            irr_daily  = r
            irr_annual = round(((1 + irr_daily) ** 252 - 1) * 100, 2)
            mwa = irr_annual

            # Hypothetical: if all cash-ready days were also invested at same IRR
            # i.e. capital * (1 + irr_daily)^total_trading_days
            trading_days = n_days
            hypo_final   = round(capital * (1 + irr_daily) ** trading_days, 2)
            hypo_return  = round((hypo_final / capital - 1) * 100, 2)
        else:
            hypo_final  = None
            hypo_return = None
    except Exception:
        hypo_final  = None
        hypo_return = None
        irr_annual  = None

    # Label each equity point as 'holding' or 'cash'
    holding_dates = set()
    for t in trades:
        holding_dates.add(t["entryDate"])
        holding_dates.add(t["exitDate"])
        # fill dates between entry and exit
    trade_ranges = [(t["entryDate"], t["exitDate"]) for t in trades]
    def in_trade_on(d):
        for en, ex in trade_ranges:
            if en <= d <= ex:
                return True
        return False
    for e in equity:
        e["period"] = "holding" if in_trade_on(e["date"]) else "cash"

    # Holding days vs cash days
    holding_days = sum(1 for e in equity if e["period"] == "holding")
    cash_days    = len(equity) - holding_days
    deployed_pct = round(holding_days / len(equity) * 100, 1) if equity else 0

    stats = {"totalReturn":round(total_ret*100,1),"finalValue":round(ev[-1]),
             "totalTrades":len(trades),"winRate":round(len(wins)/len(tr)*100,1) if tr else 0,
             "avgWin":round(np.mean(wins),2) if wins else 0,"avgLoss":round(np.mean(losses),2) if losses else 0,
             "sharpe":sharpe,"maxDrawdown":maxdd,"bmReturn":bm_ret,
             "daysSpanned":days_spanned,"tradingDays":len(df)-1,"annualisedReturn":annualised_ret,
             "twa":twa,"mwa":mwa,"irr":irr_annual,
             "hypoFinal":hypo_final,"hypoReturn":hypo_return,
             "holdingDays":holding_days,"cashDays":cash_days,"deployedPct":deployed_pct}

    # Risk metrics
    bm_df2 = yf.download(bm_name, start=start, end=end, progress=False, auto_adjust=True)
    risk = {"bmTicker":bm_name,"beta":None,"trackingError":None,"var95":None,"corrBm":None}
    if not bm_df2.empty:
        sr2 = pd.Series(ev).pct_change().dropna()
        br2 = bm_df2["Close"].squeeze().pct_change().dropna()
        n   = min(len(sr2), len(br2))
        if n > 30:
            s, b = sr2.iloc[-n:].values, br2.iloc[-n:].values
            cov = np.cov(s, b)
            risk = {"bmTicker":bm_name,
                    "beta":round(cov[0,1]/cov[1,1],3) if cov[1,1] else None,
                    "trackingError":round(float(np.std(s-b))*np.sqrt(252)*100,2),
                    "var95":round(abs(float(np.percentile(s,5)))*100,2),
                    "corrBm":round(float(np.corrcoef(s,b)[0,1]),3)}

    # Stock name
    name = ticker
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or ticker
    except: pass

    # ── Rolling metrics: Sharpe, Drawdown, Volatility vs historical ──
    rolling_metrics = {}
    try:
        roll_win  = int(extra_params.get("rollWin",  30))
        roll_hist = int(extra_params.get("rollHist", 20))
        hist_start = str((pd.Timestamp(start) - pd.DateOffset(years=roll_hist)).date())
        hist_df   = yf.download(ticker, start=hist_start, end=end, progress=False, auto_adjust=True)
        if not hist_df.empty:
            hp = hist_df["Close"].squeeze()
            hr = hp.pct_change().dropna()
            dates_all = [str(d)[:10] for d in hr.index]
            n = len(hr)
            # Rolling Sharpe (annualised)
            roll_sharpe = []
            for i in range(n):
                if i < roll_win - 1:
                    roll_sharpe.append(None)
                else:
                    sl = hr.iloc[i-roll_win+1:i+1]
                    std = float(sl.std())
                    roll_sharpe.append(round(float(sl.mean())/std*np.sqrt(252),3) if std>0 else None)
            # Rolling Volatility (annualised %)
            roll_vol = []
            for i in range(n):
                if i < roll_win - 1:
                    roll_vol.append(None)
                else:
                    sl = hr.iloc[i-roll_win+1:i+1]
                    roll_vol.append(round(float(sl.std())*np.sqrt(252)*100,2))
            # Rolling Max Drawdown (%)
            roll_dd = []
            for i in range(n):
                if i < roll_win - 1:
                    roll_dd.append(None)
                else:
                    prices_sl = hp.iloc[i-roll_win+1:i+1]
                    peak = prices_sl.cummax()
                    dd   = ((prices_sl - peak)/peak).min()
                    roll_dd.append(round(float(dd)*100,2))
            # Historical percentiles for context lines
            valid_s = [v for v in roll_sharpe if v is not None]
            valid_v = [v for v in roll_vol    if v is not None]
            valid_d = [v for v in roll_dd     if v is not None]
            def pct(arr, p): return round(float(np.percentile(arr,p)),3) if arr else None
            rolling_metrics = {
                "dates":      dates_all,
                "rollWin":    roll_win,
                "rollHist":   roll_hist,
                "sharpe":     roll_sharpe,
                "vol":        roll_vol,
                "drawdown":   roll_dd,
                "sharpeP25":  pct(valid_s,25), "sharpeP50":pct(valid_s,50), "sharpeP75":pct(valid_s,75),
                "volP25":     pct(valid_v,25),  "volP50":pct(valid_v,50),   "volP75":pct(valid_v,75),
                "ddP25":      pct(valid_d,25),  "ddP50":pct(valid_d,50),    "ddP75":pct(valid_d,75),
            }
    except Exception as ex:
        rolling_metrics = {"error": str(ex)}

    return {"trades":trades,"equity":equity,"bmEquity":bm_eq,"bmLabel":bm_lbl,
            "stats":stats,"riskMetrics":risk,"stockName":name,"ticker":ticker,
            "rollingMetrics":rolling_metrics}, None


def calc_valuation(price, eps, gr, years=5, disc=0.10):
    if not all([price, eps, gr]): return {}
    pe  = round(price/eps, 2)
    peg = round(pe/gr, 2) if gr else None
    proj = [eps*((1+gr/100)**y) for y in range(1, years+1)]
    dcf  = sum(e/(1+disc)**y for y,e in enumerate(proj,1)) + proj[-1]*15/(1+disc)**years
    mos  = round((dcf-price)/price*100, 1)
    return {"pe":pe,"peg":peg,"dcfValue":round(dcf,2),"marginOfSafety":mos,
            "projEps":[round(e,2) for e in proj]}


def run_pair(t1, t2, start, end, window=60, custom=None):
    d1 = yf.download(t1, start=start, end=end, progress=False, auto_adjust=True)
    d2 = yf.download(t2, start=start, end=end, progress=False, auto_adjust=True)
    if d1.empty or d2.empty: return None, "No data"

    p1 = d1["Close"].squeeze(); p2 = d2["Close"].squeeze()
    n1 = (p1/p1.iloc[0]*100).round(2); n2 = (p2/p2.iloc[0]*100).round(2)
    cb = pd.DataFrame({t1:n1, t2:n2}).dropna()
    dates = [str(d)[:10] for d in cb.index]

    # Rolling correlation
    r1 = cb[t1].pct_change(); r2 = cb[t2].pct_change()
    rolling_corr = r1.rolling(window).corr(r2).round(3)
    rolling_corr_list = [round(float(v),3) if not np.isnan(v) else None for v in rolling_corr]

    # Overall correlation
    corr = round(float(cb[t1].corr(cb[t2])), 3)

    # Context assets: SPY, TLT (10Y bond proxy), GLD, optional custom
    def norm_series(ticker):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty: return None, None
            s = df["Close"].squeeze().reindex(cb.index, method="ffill")
            normed = (s/s.iloc[0]*100).round(2)
            ret    = round((float(s.iloc[-1])/float(s.iloc[0])-1)*100,1)
            return [v if not np.isnan(v) else None for v in normed], ret
        except: return None, None

    spy_n, spy_r = norm_series("SPY")
    tlt_n, tlt_r = norm_series("TLT")
    gld_n, gld_r = norm_series("GLD")
    cus_n, cus_r = norm_series(custom) if custom else (None, None)

    return {"dates":dates,
            "price1":cb[t1].tolist(),"price2":cb[t2].tolist(),
            "ratio":(cb[t1]/cb[t2]).round(4).tolist(),
            "corr":corr,"rollingCorr":rolling_corr_list,
            "ret1":round((float(p1.iloc[-1])/float(p1.iloc[0])-1)*100,1),
            "ret2":round((float(p2.iloc[-1])/float(p2.iloc[0])-1)*100,1),
            "ctx":{"spy":spy_n,"tlt":tlt_n,"gld":gld_n,"custom":cus_n},
            "ctxReturns":{"spy":spy_r,"tlt":tlt_r,"gld":gld_r,"custom":cus_r},
            "ctxCustomTicker":custom}, None


# ── Routes ────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Investment Mandate & Capital Deployment</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#F8FAFC;--sur:#fff;--sur2:#F1F5F9;--bdr:#E2E8F0;--bdr2:#CBD5E1;--txt:#0F172A;--mut:#64748B;--acc:#2563EB;--acl:#EFF6FF;--grn:#16A34A;--gnl:#F0FDF4;--red:#DC2626;--rdl:#FEF2F2;--shd:0 1px 3px rgba(0,0,0,.08);--r:10px;--f:'Inter',sans-serif}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--txt);font-family:var(--f);font-size:14px}
header{background:var(--sur);border-bottom:1px solid var(--bdr);padding:0 1.5rem;height:52px;display:flex;align-items:center;gap:1.5rem;position:sticky;top:0;z-index:100;box-shadow:var(--shd)}
.logo{font-size:.95rem;font-weight:700;color:var(--acc);white-space:nowrap}.logo span{color:var(--txt)}
nav{display:flex;gap:.2rem;overflow-x:auto}
.nb{padding:.35rem .75rem;border:none;background:transparent;color:var(--mut);font-family:var(--f);font-size:.8rem;font-weight:500;border-radius:6px;cursor:pointer;white-space:nowrap}
.nb:hover{background:var(--sur2);color:var(--txt)}.nb.active{background:var(--acl);color:var(--acc)}
.pg{display:none}.pg.active{display:block}
.two{display:grid;grid-template-columns:290px 1fr;gap:1rem;padding:1rem;min-height:calc(100vh - 52px);align-items:start}
.card{background:var(--sur);border:1px solid var(--bdr);border-radius:var(--r);box-shadow:var(--shd)}
.ch{padding:.75rem 1rem;border-bottom:1px solid var(--bdr);font-size:.68rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--mut)}
.cb{padding:1rem}
.sl{font-size:.62rem;font-weight:700;letter-spacing:.8px;text-transform:uppercase;color:var(--acc);margin:.85rem 0 .35rem}.sl:first-child{margin-top:0}
.fd{margin-bottom:.65rem}.fd label{display:block;font-size:.73rem;font-weight:500;color:var(--mut);margin-bottom:.28rem}
input,select,textarea{width:100%;min-width:0;background:var(--sur);border:1px solid var(--bdr2);color:var(--txt);font-family:var(--f);font-size:.83rem;padding:.48rem .6rem;border-radius:6px;outline:none;box-sizing:border-box;transition:border-color .15s}
input:focus,select:focus,textarea:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(37,99,235,.1)}
textarea{resize:vertical;min-height:68px}
.r2{display:grid;grid-template-columns:1fr 1fr;gap:.45rem}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:.3rem;padding:.52rem .9rem;border:none;border-radius:6px;font-family:var(--f);font-size:.8rem;font-weight:600;cursor:pointer;transition:all .15s}
.bp{background:var(--acc);color:#fff;width:100%}.bp:hover{background:#1D4ED8}.bp:disabled{opacity:.45;cursor:not-allowed}
.bg{background:transparent;color:var(--mut);border:1px solid var(--bdr)}.bg:hover{background:var(--sur2)}
.bs{background:var(--gnl);color:var(--grn);border:1px solid #BBF7D0;width:100%}
.sg{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin-bottom:1rem}
.sc{background:var(--sur);border:1px solid var(--bdr);border-radius:var(--r);padding:.85rem .9rem;box-shadow:var(--shd)}
.sc .sl2{font-size:.62rem;font-weight:600;letter-spacing:.4px;text-transform:uppercase;color:var(--mut);margin-bottom:.28rem}
.sc .sv{font-size:1.25rem;font-weight:700}.sc .ss{font-size:.68rem;color:var(--mut);margin-top:.25rem}
.pos{color:var(--grn)}.neg{color:var(--red)}.neu{color:var(--acc)}
.cc{background:var(--sur);border:1px solid var(--bdr);border-radius:var(--r);padding:1rem;margin-bottom:1rem;box-shadow:var(--shd)}
.ct{font-size:.65rem;font-weight:600;letter-spacing:.5px;text-transform:uppercase;color:var(--mut);margin-bottom:.75rem}
canvas{max-height:240px}
.cv-wrap{position:relative}
.cv-toolbar{display:flex;align-items:center;gap:.35rem;flex-wrap:wrap;margin-bottom:.65rem}
.cv-toolbar .ct{margin-bottom:0;flex:1;min-width:0}
.cv-btn{padding:.22rem .55rem;font-size:.68rem;font-weight:600;border:1px solid var(--bdr2);border-radius:5px;background:var(--sur2);color:var(--mut);cursor:pointer;font-family:var(--f);white-space:nowrap}
.cv-btn:hover{background:#F1F5F9;border-color:#94A3B8;color:var(--txt)}
.cv-btn.cv-active{background:#334155;border-color:#334155;color:#fff}
.cv-range{display:none;align-items:center;gap:.3rem;margin-top:.4rem;flex-wrap:wrap}
.cv-range.open{display:flex}
.cv-range input[type=date]{width:auto;min-width:0;font-size:.72rem;padding:.28rem .4rem;border-radius:5px}
.cv-range button{padding:.28rem .6rem;font-size:.7rem;font-weight:600;border:1px solid #94A3B8;border-radius:5px;background:#F1F5F9;color:#334155;cursor:pointer;font-family:var(--f)}
.tc{background:var(--sur);border:1px solid var(--bdr);border-radius:var(--r);overflow:hidden;box-shadow:var(--shd);margin-bottom:1rem}
.th{padding:.65rem 1rem;border-bottom:1px solid var(--bdr);font-size:.65rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--mut);display:flex;justify-content:space-between;align-items:center}
.bge{background:var(--acl);color:var(--acc);padding:.15rem .5rem;border-radius:20px;font-size:.65rem;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:.78rem}
th{background:var(--sur2);padding:.5rem .85rem;text-align:left;color:var(--mut);font-weight:600;font-size:.65rem;letter-spacing:.3px}
td{padding:.5rem .85rem;border-top:1px solid var(--bdr)}
tr:hover td{background:var(--sur2)}
.sw{display:none;flex-direction:column;align-items:center;justify-content:center;padding:4rem;gap:.75rem}
.sp{width:30px;height:30px;border:3px solid var(--bdr);border-top-color:var(--acc);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.sw p{font-size:.78rem;color:var(--mut)}
.ph{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;color:var(--mut);gap:.6rem;text-align:center}
.ph .ic{font-size:2rem}.ph p{font-size:.8rem;line-height:1.7}
.er{background:var(--rdl);border:1px solid #FECACA;border-radius:6px;padding:.7rem .85rem;font-size:.78rem;color:var(--red);margin-top:.6rem;display:none}
.mcols{display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-bottom:.7rem}
.mch{display:flex;flex-direction:column;align-items:flex-start;gap:.28rem;padding:.4rem .5rem;border-radius:6px 6px 0 0;border:1px solid transparent}
.mch.buy{background:#F1F5F9;border-color:#CBD5E1}.mch.sell{background:#F8F7F4;border-color:#D6D1C8}
.mch .mct{font-size:.65rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase}
.mch.buy .mct{color:#475569}.mch.sell .mct{color:#6B6459}
.aog{display:flex;width:100%;border-radius:5px;overflow:hidden;border:1px solid var(--bdr2)}
.aog button{flex:1;padding:.2rem 0;font-size:.68rem;font-weight:700;text-align:center;border:none;cursor:pointer;background:var(--sur2);color:var(--mut);font-family:var(--f)}
.aog button.aa{background:#334155;color:#fff}.aog button.ao{background:#64748B;color:#fff}
.mi{border:1px solid var(--bdr);border-top:none;border-radius:0 0 6px 6px;padding:.38rem;display:flex;flex-direction:column;gap:.3rem}
.mr label{font-size:.65rem;font-weight:500;color:var(--mut)}
.mr input{font-size:.78rem;padding:.35rem .48rem;border:1px solid var(--bdr2);border-radius:5px;background:var(--sur);color:var(--txt);font-family:var(--f);width:100%;outline:none;box-sizing:border-box}
.mr input:focus{border-color:var(--acc)}
.sb{display:flex;align-items:baseline;gap:.6rem;margin-bottom:.85rem;padding:.7rem .9rem;background:var(--sur);border:1px solid var(--bdr);border-radius:var(--r);box-shadow:var(--shd)}
.sn{font-size:.95rem;font-weight:700}.st{font-size:.75rem;font-weight:600;color:var(--acc);background:var(--acl);padding:.15rem .5rem;border-radius:20px}.sp2{font-size:.72rem;color:var(--mut);margin-left:auto}
.mo{position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:200;display:none;align-items:center;justify-content:center}
.mo.open{display:flex}
.md{background:var(--sur);border-radius:12px;padding:1.4rem;width:380px;max-width:92vw;box-shadow:0 20px 40px rgba(0,0,0,.15)}
.md h3{font-size:.9rem;font-weight:700;margin-bottom:.85rem}
.sg2{display:grid;grid-template-columns:repeat(3,1fr);gap:.45rem;margin-bottom:.75rem}
.shb{display:flex;flex-direction:column;align-items:center;gap:.3rem;padding:.7rem .4rem;background:var(--sur2);border:1px solid var(--bdr);border-radius:7px;cursor:pointer;font-size:.7rem;color:var(--txt);font-weight:500}
.shb:hover{background:var(--acl);border-color:var(--acc);color:var(--acc)}
.stx{background:var(--sur2);border:1px solid var(--bdr);border-radius:6px;padding:.6rem;font-size:.75rem;color:var(--txt);line-height:1.6;max-height:100px;overflow-y:auto;margin-bottom:.6rem;white-space:pre-wrap}
.hr{display:grid;grid-template-columns:1fr 1fr 1fr 1fr auto;gap:.35rem;margin-bottom:.45rem;align-items:center}
.hr input{font-size:.78rem;padding:.4rem .5rem}
.rmb{background:none;border:1px solid var(--bdr);border-radius:5px;color:var(--mut);cursor:pointer;padding:.35rem .45rem;font-size:.8rem}
.rmb:hover{background:var(--rdl);color:var(--red);border-color:#FECACA}
.tgl{position:relative;width:32px;height:17px;flex-shrink:0}
.tgl input{opacity:0;width:0;height:0}
.tsl{position:absolute;inset:0;background:var(--bdr2);border-radius:17px;cursor:pointer;transition:.2s}
.tsl:before{content:'';position:absolute;width:11px;height:11px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.2s;box-shadow:0 1px 2px rgba(0,0,0,.2)}
.tgl input:checked+.tsl{background:var(--acc)}
.tgl input:checked+.tsl:before{transform:translateX(15px)}
.tgr{display:flex;align-items:center;gap:.55rem;margin-bottom:.65rem}
.tgl-lbl{font-size:.78rem;color:var(--txt);font-weight:500}
.opt{display:none}.opt.on{display:block}
.ctb{border-collapse:collapse;font-size:.75rem;width:100%}
.ctb th{background:var(--sur2);padding:.45rem .65rem;text-align:center;color:var(--mut);font-weight:600}
.ctb td{padding:.45rem .65rem;text-align:center;border:1px solid var(--bdr);font-weight:500}
.ch{background:#DBEAFE;color:#1D4ED8}.cm{background:#FEF9C3;color:#92400E}
.cl{background:#F0FDF4;color:#15803D}.cn{background:#FEF2F2;color:#DC2626}
.tbar{background:var(--sur2);border-radius:8px;height:10px;margin:.4rem 0;overflow:hidden}
.tfil{height:100%;border-radius:8px;transition:width .5s}
@media(max-width:768px){.two{grid-template-columns:1fr;padding:.75rem}.sg{grid-template-columns:repeat(2,1fr)}header{padding:0 .75rem;gap:.75rem}.nb{padding:.3rem .5rem;font-size:.74rem}}
</style>
</head>
<body>
<header>
  <div class="logo">Investment Mandate<span> & Capital Deployment</span></div>
  <nav>
    <button class="nb active" onclick="showPage('mandate',this)">Mandate</button>
    <button class="nb" onclick="showPage('valuation',this)">Valuation</button>
    <button class="nb" onclick="showPage('pair',this)">Pair</button>
    <button class="nb" onclick="showPage('portfolio',this)">Portfolio</button>
    <button class="nb" onclick="showPage('monitor',this)">Live Monitor</button>
    <button class="nb" onclick="showPage('financials',this)">Financials</button>
    <button class="nb" onclick="showPage('records',this)">Records</button>
  </nav>
</header>

<!-- MANDATE -->
<div id="pg-mandate" class="pg active">
<div class="two">
  <aside>
    <div class="card"><div class="ch">Mandate Setup</div><div class="cb">
      <div class="sl">Asset</div>
      <div class="fd"><label>Ticker</label><input id="bt-ticker" value="QQQ"/></div>
      <div class="r2">
        <div class="fd"><label>Start</label><input type="date" id="bt-start" value="2000-01-01" style="font-size:.75rem;padding:.43rem .35rem" oninput="calcDates('bt',  this.id.includes('start')?'start':this.id.includes('end')?'end':'days')"/></div>
        <div class="fd"><label>End</label><input type="date" id="bt-end" value="2020-01-01" style="font-size:.75rem;padding:.43rem .35rem" oninput="calcDates('bt',  this.id.includes('start')?'start':this.id.includes('end')?'end':'days')"/></div>
      </div>
      <div class="fd">
        <label>Natural Days <span style="font-weight:400;color:var(--mut)">(optional — fills missing date)</span></label>
        <div style="display:flex;gap:.4rem;align-items:center">
          <input type="number" id="bt-days" placeholder="e.g. 3650" min="1" style="flex:1" oninput="calcDates('bt',  this.id.includes('start')?'start':this.id.includes('end')?'end':'days')"/>
          <div id="bt-days-hint" style="font-size:.68rem;color:var(--mut);white-space:nowrap;min-width:60px"></div>
        </div>
      </div>
      <div class="sl">Capital</div>
      <div class="fd"><label>Initial Capital (USD)</label><input type="number" id="bt-capital" value="10000" min="100"/></div>
      <div class="sl">Benchmark</div>
      <div class="fd"><label>Benchmark</label>
        <div style="display:flex;gap:.35rem">
          <select id="bt-bm" onchange="applyBm()" style="flex:1">
            <option value="SPY">SPY — S&amp;P 500</option>
            <option value="QQQ">QQQ — Nasdaq 100</option>
            <option value="DIA">DIA — Dow Jones</option>
            <option value="IWM">IWM — Russell 2000</option>
            <option value="custom">Custom…</option>
          </select>
          <input id="bt-bm-custom" placeholder="e.g. VTI" style="width:68px;display:none"/>
        </div>
      </div>
      <div class="fd"><label>Mode</label>
        <div class="aog" style="border-radius:6px;border:1px solid var(--bdr2)">
          <button id="bm-hold" class="ao" onclick="setBm('hold')" style="padding:.42rem 0;font-size:.76rem">Buy &amp; Hold</button>
          <button id="bm-rules" onclick="setBm('rules')" style="padding:.42rem 0;font-size:.76rem">Same Rules</button>
        </div>
      </div>
      <div class="sl">Metrics <span style="font-weight:400;font-size:.65rem;text-transform:none;letter-spacing:0;color:var(--mut)">— all optional</span></div>
      <div style="font-size:.68rem;color:var(--mut);margin-bottom:.55rem">AND = all must be met · OR = any one triggers</div>
      <div class="mcols">
        <div>
          <div class="mch buy"><span class="mct">Deploy Capital</span>
            <div class="aog"><button id="buy-and" onclick="setGate('buy','and')">AND</button><button id="buy-or" class="ao" onclick="setGate('buy','or')">OR</button></div>
          </div>
          <div class="mi">
            <div class="mr"><label>RSI below</label><input type="number" id="bt-rsiBuy" placeholder="e.g. 30"/></div>
            <div class="mr"><label>Max DD ≤ (%)</label><input type="number" id="bm-maxdd" placeholder="blank=skip"/></div>
            <div class="mr"><label>VaR 95% ≤ (%)</label><input type="number" id="bm-var" placeholder="blank=skip"/></div>
            <div class="mr"><label>P/E ≤ (×)</label><input type="number" id="bm-pe" placeholder="blank=skip"/></div>
            <div class="mr"><label>P/B ≤ (×)</label><input type="number" id="bm-pb" placeholder="blank=skip"/></div>
            <div class="mr"><label>EPS ≥ ($)</label><input type="number" id="bm-eps" placeholder="blank=skip"/></div>
            <div class="mr"><label>Div Yield ≥ (%)</label><input type="number" id="bm-divy" placeholder="blank=skip"/></div>
            <div class="mr"><label>ROE ≥ (%)</label><input type="number" id="bm-roe" placeholder="blank=skip"/></div>
            <div class="mr"><label>D/E ≤ (×)</label><input type="number" id="bm-de" placeholder="blank=skip"/></div>
            <div class="mr"><label>FCF Yield ≥ (%)</label><input type="number" id="bm-fcf" placeholder="blank=skip"/></div>
            <div class="mr"><label>Rev Growth ≥ (%)</label><input type="number" id="bm-rev" placeholder="blank=skip"/></div>
          </div>
        </div>
        <div>
          <div class="mch sell"><span class="mct">Return Capital</span>
            <div class="aog"><button id="sell-and" onclick="setGate('sell','and')">AND</button><button id="sell-or" class="ao" onclick="setGate('sell','or')">OR</button></div>
          </div>
          <div class="mi">
            <div class="mr"><label>RSI above</label><input type="number" id="bt-rsiSell" placeholder="e.g. 70"/></div>
            <div class="mr"><label>Profit target (%)</label><input type="number" id="bt-profit" placeholder="e.g. 15"/></div>
            <div class="mr"><label>Stop loss (%) ⚡</label><input type="number" id="bt-stop" placeholder="blank=skip"/></div>
            <div class="mr"><label>Max DD ≥ (%)</label><input type="number" id="sm-maxdd" placeholder="blank=skip"/></div>
            <div class="mr"><label>VaR 95% ≥ (%)</label><input type="number" id="sm-var" placeholder="blank=skip"/></div>
            <div class="mr"><label>P/E ≥ (×)</label><input type="number" id="sm-pe" placeholder="blank=skip"/></div>
            <div class="mr"><label>P/B ≥ (×)</label><input type="number" id="sm-pb" placeholder="blank=skip"/></div>
            <div class="mr"><label>EPS ≤ ($)</label><input type="number" id="sm-eps" placeholder="blank=skip"/></div>
            <div class="mr"><label>Div Yield ≤ (%)</label><input type="number" id="sm-divy" placeholder="blank=skip"/></div>
            <div class="mr"><label>ROE ≤ (%)</label><input type="number" id="sm-roe" placeholder="blank=skip"/></div>
            <div class="mr"><label>D/E ≥ (×)</label><input type="number" id="sm-de" placeholder="blank=skip"/></div>
            <div class="mr"><label>FCF Yield ≤ (%)</label><input type="number" id="sm-fcf" placeholder="blank=skip"/></div>
            <div class="mr"><label>Rev Growth ≤ (%)</label><input type="number" id="sm-rev" placeholder="blank=skip"/></div>
          </div>
        </div>
      </div>
      <div class="sl">Rolling Risk Metrics <span style="font-weight:400;font-size:.65rem;text-transform:none;letter-spacing:0;color:var(--mut)">— vs historical</span></div>
      <div class="r2">
        <div class="fd"><label>Window (days)</label><input type="number" id="bt-roll-win" value="30" min="5" max="252" placeholder="30"/></div>
        <div class="fd"><label>History (years)</label><input type="number" id="bt-roll-hist" value="20" min="1" max="50" placeholder="20"/></div>
      </div>
      <button class="btn bp" id="bt-run" onclick="runBacktest()" style="margin-top:.75rem">▶ Run Mandate</button>
      <div class="er" id="bt-err"></div>
    </div></div>
  </aside>
  <main>
    <div class="ph" id="bt-ph"><div class="ic">📈</div><p>Configure your strategy and hit Run Mandate.</p></div>
    <div class="sw" id="bt-sw"><div class="sp"></div><p>Fetching data &amp; running simulation…</p></div>
    <div id="bt-res" style="display:none">
      <div class="sb"><div class="sn" id="bt-sname">—</div><div class="st" id="bt-stick">—</div><div class="sp2" id="bt-sper">—</div></div>
      <div class="sg" id="bt-stats" style="grid-template-columns:repeat(4,1fr)"></div>
      <div class="card" style="margin-bottom:1rem">
        <div class="ch">Strategy vs Benchmark</div>
        <div class="cb" style="overflow-x:auto">
          <table><thead><tr><th>Metric</th><th id="cmp-s" style="text-align:right">Strategy</th><th id="cmp-b" style="text-align:right">Benchmark</th><th style="text-align:right">Edge</th></tr></thead><tbody id="bt-cmp"></tbody></table>
        </div>
      </div>
      <div class="cc"><div class="cv-toolbar"><div class="ct" id="bt-ctitle">Equity Curve</div><div style="display:flex;gap:.25rem;align-items:center">
  <button class="cv-btn cv-active" onclick="setCvView('bt',this,'D')">D</button>
  <button class="cv-btn" onclick="setCvView('bt',this,'M')">M</button>
  <button class="cv-btn" onclick="setCvView('bt',this,'Y')">Y</button>
  <button class="cv-btn" id="bt-range-btn" onclick="toggleCvRange('bt')">📅 Range</button>
</div></div>
<div class="cv-range" id="bt-range-wrap">
  <input type="date" id="bt-range-from" placeholder="From"/>
  <span style="font-size:.72rem;color:var(--mut)">to</span>
  <input type="date" id="bt-range-to" placeholder="To"/>
  <button onclick="applyCvRange('bt')">Apply</button>
</div>
<canvas id="bt-chart"></canvas></div>
      <div class="tc">
        <div class="th">Trade Log <span class="bge" id="bt-tc">0 trades</span></div>
        <div style="overflow-x:auto"><table><thead><tr><th>Entry</th><th>Exit</th><th>Entry $</th><th>Exit $</th><th>Return</th><th>P&L (USD)</th><th>Reason</th></tr></thead><tbody id="bt-trades"></tbody></table></div>
      </div>
      <div class="card" style="margin-bottom:1rem">
        <div class="ch">Risk vs Benchmark</div>
        <div class="cb"><div class="sg" id="bt-risk" style="grid-template-columns:repeat(4,1fr);margin-bottom:0"></div></div>
      </div>
      <div class="card" style="margin-bottom:1rem" id="hypo-card">
        <div class="ch">Hypothetical Return <span style="font-size:.65rem;font-weight:400;text-transform:none;letter-spacing:0;color:var(--mut)">— IRR applied to full period including cash days</span></div>
        <div class="cb" id="hypo-body"></div>
      </div>

      <div class="card" style="margin-bottom:1rem" id="rolling-card">
        <div class="ch">Rolling Risk Metrics <span id="rolling-ch-sub" style="font-size:.65rem;font-weight:400;text-transform:none;letter-spacing:0;color:var(--mut)"></span></div>
        <div class="cb" id="rolling-body"></div>
      </div>

      <div class="card" style="margin-bottom:1rem">
        <div class="ch">Notes &amp; Thoughts</div>
        <div class="cb">
          <div class="fd"><label>Analysis Notes</label><textarea id="bt-notes" placeholder="Observations…"></textarea></div>
          <div class="fd"><label>Investment Thesis</label><textarea id="bt-thesis" placeholder="Your thesis…"></textarea></div>
          <div style="display:flex;gap:.5rem"><button class="btn bs" onclick="saveRec('bt')">💾 Save</button><button class="btn bg" onclick="openShare('bt')">↗ Share</button></div>
        </div>
      </div>
    </div>
  </main>
</div></div>

<!-- VALUATION -->
<div id="pg-valuation" class="pg">
<div class="two">
  <aside><div class="card"><div class="ch">Valuation Inputs</div><div class="cb">
    <div class="fd"><label>Ticker (label)</label><input id="val-ticker" placeholder="e.g. AAPL"/></div>
    <div class="fd"><label>Current Price ($)</label><input type="number" id="val-price" placeholder="e.g. 185"/></div>
    <div class="fd"><label>EPS ($)</label><input type="number" id="val-eps" placeholder="e.g. 6.42"/></div>
    <div class="fd"><label>Growth Rate (% p.a.)</label><input type="number" id="val-gr" placeholder="e.g. 12"/></div>
    <div style="font-size:.68rem;color:var(--mut);margin-bottom:.7rem">DCF: 5yr, 10% discount, 15× terminal P/E</div>
    <button class="btn bp" onclick="runVal()">Calculate</button>
    <div class="er" id="val-err"></div>
  </div></div></aside>
  <main>
    <div class="ph" id="val-ph"><div class="ic">🔢</div><p>Enter price, EPS and growth rate.</p></div>
    <div id="val-res" style="display:none">
      <div class="card" style="margin-bottom:1rem"><div class="ch" id="val-title">Valuation</div><div class="cb">
        <div class="sg" id="val-grid" style="grid-template-columns:repeat(3,1fr)"></div>
        <div class="cc" style="margin:1rem 0 0"><div class="ct">Projected EPS (5 Years)</div><canvas id="val-chart" style="max-height:160px"></canvas></div>
      </div></div>
      <div class="card"><div class="ch">Notes &amp; Thoughts</div><div class="cb">
        <div class="fd"><label>Notes</label><textarea id="val-notes" placeholder="Your valuation notes…"></textarea></div>
        <div class="fd"><label>Thesis</label><textarea id="val-thesis" placeholder="Bull/bear case…"></textarea></div>
        <div style="display:flex;gap:.5rem"><button class="btn bs" onclick="saveRec('val')">💾 Save</button><button class="btn bg" onclick="openShare('val')">↗ Share</button></div>
      </div></div>
    </div>
  </main>
</div></div>

<!-- PAIR -->
<div id="pg-pair" class="pg">
<div class="two">
  <aside><div class="card"><div class="ch">Pair Analysis</div><div class="cb">
    <div class="sl">Securities to Compare</div>
    <div class="fd"><label>Ticker 1</label><input id="pair-t1" value="QQQ"/></div>
    <div class="fd"><label>Ticker 2</label><input id="pair-t2" value="SPY"/></div>
    <div class="r2">
      <div class="fd"><label>Start</label><input type="date" id="pair-start" value="2010-01-01" style="font-size:.75rem;padding:.43rem .35rem" oninput="calcDates('pair',this.id.includes('start')?'start':this.id.includes('end')?'end':'days')"/></div>
      <div class="fd"><label>End</label><input type="date" id="pair-end" value="2024-01-01" style="font-size:.75rem;padding:.43rem .35rem" oninput="calcDates('pair',this.id.includes('start')?'start':this.id.includes('end')?'end':'days')"/></div>
    </div>
    <div class="fd">
      <label>Natural Days <span style="font-weight:400;color:var(--mut)">(optional — fills missing date)</span></label>
      <div style="display:flex;gap:.4rem;align-items:center">
        <input type="number" id="pair-days" placeholder="e.g. 3650" min="1" style="flex:1" oninput="calcDates('pair',this.id.includes('start')?'start':this.id.includes('end')?'end':'days')"/>
        <div id="pair-days-hint" style="font-size:.68rem;color:var(--mut);white-space:nowrap;min-width:60px"></div>
      </div>
    </div>
    <div class="fd"><label>Rolling Correlation Window (days)</label><input type="number" id="pair-window" value="60" min="10" max="252" placeholder="e.g. 60"/></div>

    <div class="sl">Market Context <span style="font-weight:400;font-size:.65rem;text-transform:none;letter-spacing:0;color:var(--mut)">— fixed benchmarks always shown</span></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.3rem;margin-bottom:.5rem;font-size:.72rem">
      <div style="background:var(--sur2);border:1px solid var(--bdr);border-radius:5px;padding:.35rem .55rem;color:var(--mut)">📈 S&amp;P 500 (SPY)</div>
      <div style="background:var(--sur2);border:1px solid var(--bdr);border-radius:5px;padding:.35rem .55rem;color:var(--mut)">🏦 10Y Bond (TLT)</div>
      <div style="background:var(--sur2);border:1px solid var(--bdr);border-radius:5px;padding:.35rem .55rem;color:var(--mut)">🥇 Gold (GLD)</div>
      <div style="background:var(--acl);border:1px solid #BFDBFE;border-radius:5px;padding:.35rem .55rem;color:var(--acc)">➕ Custom (below)</div>
    </div>
    <div class="fd"><label>Custom Context Ticker <span style="font-weight:400;color:var(--mut)">(optional)</span></label><input id="pair-custom" placeholder="e.g. VIX, BTC-USD, EEM"/></div>
    <div style="font-size:.68rem;color:var(--mut);margin-bottom:.65rem">Context assets help explain whether correlation changes reflect market-wide stress, rate moves, or sector rotation.</div>

    <button class="btn bp" id="pair-run" onclick="runPair()">▶ Analyse Pair</button>
    <div class="er" id="pair-err"></div>
  </div></div></aside>
  <main>
    <div class="ph" id="pair-ph"><div class="ic">⚖️</div><p>Enter two tickers to compare trends.</p></div>
    <div class="sw" id="pair-sw"><div class="sp"></div><p>Fetching data…</p></div>
    <div id="pair-res" style="display:none">
      <div class="sg" id="pair-stats" style="grid-template-columns:repeat(4,1fr)"></div>

      <!-- Convergence / Divergence banner -->
      <div id="pair-cd-banner" style="margin-bottom:1rem;padding:.85rem 1.1rem;border-radius:var(--r);border:1px solid var(--bdr);box-shadow:var(--shd)">
        <div id="pair-cd-text"></div>
      </div>

      <div class="cc"><div class="cv-toolbar"><div class="ct">Normalised Price (rebased to 100) — <span id="pair-ct-sub" style="font-weight:400">price movement</span></div><div style="display:flex;gap:.25rem">
  <button class="cv-btn cv-active" onclick="setCvView('pair-norm',this,'D')">D</button>
  <button class="cv-btn" onclick="setCvView('pair-norm',this,'M')">M</button>
  <button class="cv-btn" onclick="setCvView('pair-norm',this,'Y')">Y</button>
  <button class="cv-btn" onclick="toggleCvRange('pair-norm')">📅 Range</button>
</div></div>
<div class="cv-range" id="pair-norm-range-wrap">
  <input type="date" id="pair-norm-range-from"/><span style="font-size:.72rem;color:var(--mut)">to</span>
  <input type="date" id="pair-norm-range-to"/>
  <button onclick="applyCvRange('pair-norm')">Apply</button>
</div>
<canvas id="pair-chart"></canvas></div>

      <div class="cc"><div class="cv-toolbar"><div class="ct">Rolling Correlation (<span id="pair-win-lbl">60</span>-day window) — convergence/divergence over time</div><div style="display:flex;gap:.25rem">
  <button class="cv-btn cv-active" onclick="setCvView('pair-corr',this,'D')">D</button>
  <button class="cv-btn" onclick="setCvView('pair-corr',this,'M')">M</button>
  <button class="cv-btn" onclick="setCvView('pair-corr',this,'Y')">Y</button>
  <button class="cv-btn" onclick="toggleCvRange('pair-corr')">📅 Range</button>
</div></div>
<div class="cv-range" id="pair-corr-range-wrap">
  <input type="date" id="pair-corr-range-from"/><span style="font-size:.72rem;color:var(--mut)">to</span>
  <input type="date" id="pair-corr-range-to"/>
  <button onclick="applyCvRange('pair-corr')">Apply</button>
</div>
<canvas id="rolling-corr-chart"></canvas></div>

      <div class="cc"><div class="cv-toolbar"><div class="ct">Price Ratio (Ticker 1 ÷ Ticker 2) — spread</div><div style="display:flex;gap:.25rem">
  <button class="cv-btn cv-active" onclick="setCvView('pair-ratio',this,'D')">D</button>
  <button class="cv-btn" onclick="setCvView('pair-ratio',this,'M')">M</button>
  <button class="cv-btn" onclick="setCvView('pair-ratio',this,'Y')">Y</button>
  <button class="cv-btn" onclick="toggleCvRange('pair-ratio')">📅 Range</button>
</div></div>
<div class="cv-range" id="pair-ratio-range-wrap">
  <input type="date" id="pair-ratio-range-from"/><span style="font-size:.72rem;color:var(--mut)">to</span>
  <input type="date" id="pair-ratio-range-to"/>
  <button onclick="applyCvRange('pair-ratio')">Apply</button>
</div>
<canvas id="ratio-chart"></canvas></div>

      <div class="cc"><div class="cv-toolbar"><div class="ct">Market Context — S&amp;P 500 · 10Y Bond · Gold<span id="pair-ctx-lbl"></span></div><div style="display:flex;gap:.25rem">
  <button class="cv-btn cv-active" onclick="setCvView('pair-ctx',this,'D')">D</button>
  <button class="cv-btn" onclick="setCvView('pair-ctx',this,'M')">M</button>
  <button class="cv-btn" onclick="setCvView('pair-ctx',this,'Y')">Y</button>
  <button class="cv-btn" onclick="toggleCvRange('pair-ctx')">📅 Range</button>
</div></div>
<div class="cv-range" id="pair-ctx-range-wrap">
  <input type="date" id="pair-ctx-range-from"/><span style="font-size:.72rem;color:var(--mut)">to</span>
  <input type="date" id="pair-ctx-range-to"/>
  <button onclick="applyCvRange('pair-ctx')">Apply</button>
</div>
<canvas id="ctx-chart"></canvas></div>

      <div class="card" style="margin-bottom:1rem">
        <div class="ch">Market Regime Interpretation</div>
        <div class="cb" id="pair-regime"></div>
      </div>

      <div class="card"><div class="ch">Notes &amp; Thoughts</div><div class="cb">
        <div class="fd"><label>Observations</label><textarea id="pair-notes" placeholder="What does the data tell you?"></textarea></div>
        <div class="fd"><label>Thesis</label><textarea id="pair-thesis" placeholder="Trade idea…"></textarea></div>
        <div style="display:flex;gap:.5rem"><button class="btn bs" onclick="saveRec('pair')">💾 Save</button><button class="btn bg" onclick="openShare('pair')">↗ Share</button></div>
      </div></div>
    </div>
  </main>
</div></div>

<!-- PORTFOLIO -->
<div id="pg-portfolio" class="pg">
<div class="two">
  <aside><div class="card"><div class="ch">Holdings</div><div class="cb">
    <div id="pf-list"></div>
    <button class="btn bg" style="width:100%;margin-bottom:.75rem" onclick="addHolding()">+ Add Holding</button>
    <div class="sl">Return Target</div>
    <div class="tgr"><label class="tgl"><input type="checkbox" id="pf-use-target" onchange="tog('pf-target-fld','pf-use-target')"/><span class="tsl"></span></label><span class="tgl-lbl">Set target return p.a.</span></div>
    <div class="opt" id="pf-target-fld"><div class="fd"><label>Target Return p.a. (%)</label><input type="number" id="pf-target" placeholder="e.g. 12"/></div></div>
    <div class="sl">Benchmark</div>
    <div class="tgr"><label class="tgl"><input type="checkbox" id="pf-use-bm" onchange="tog('pf-bm-fld','pf-use-bm')"/><span class="tsl"></span></label><span class="tgl-lbl">Compare to benchmark</span></div>
    <div class="opt" id="pf-bm-fld"><div class="fd"><label>Benchmark Ticker</label><input id="pf-bm" placeholder="e.g. SPY"/></div></div>
    <div class="sl">Lookback Period</div>
    <div class="r2" style="margin-bottom:.65rem">
      <div class="fd" style="margin-bottom:0"><label>Amount</label><input type="number" id="pf-lb-val" value="1" min="1"/></div>
      <div class="fd" style="margin-bottom:0"><label>Unit</label>
        <select id="pf-lb-unit">
          <option value="natural">Natural days</option>
          <option value="trading">Trading days</option>
          <option value="weeks">Weeks</option>
          <option value="months">Months</option>
          <option value="years" selected>Years</option>
        </select>
      </div>
    </div>
    <div style="font-size:.68rem;color:var(--mut);margin-bottom:.65rem" id="pf-lb-hint">≈ 365 calendar days</div>
    <div class="sl">Watchlist <span style="font-weight:400;font-size:.65rem;text-transform:none;letter-spacing:0;color:var(--mut)">— securities to appraise</span></div>
    <div style="font-size:.68rem;color:var(--mut);margin-bottom:.5rem">Add tickers to compute Appraisal Ratio vs your portfolio benchmark</div>
    <div id="wl-list"></div>
    <button class="btn bg" style="width:100%;margin-bottom:.75rem;font-size:.76rem" onclick="addWatchItem()">+ Add Security</button>
    <button class="btn bp" id="pf-run" onclick="runPortfolio()">▶ Analyse Portfolio</button>
    <div class="er" id="pf-err"></div>
  </div></div></aside>
  <main>
    <div class="ph" id="pf-ph"><div class="ic">🏦</div><p>Add your holdings on the left.</p></div>
    <div class="sw" id="pf-sw"><div class="sp"></div><p>Fetching prices…</p></div>
    <div id="pf-res" style="display:none">
      <div class="sg" id="pf-stats"></div>
      <div class="tc" style="margin-bottom:1rem">
        <div class="th">Holdings</div>
        <div style="overflow-x:auto"><table><thead><tr><th>Ticker</th><th>Shares</th><th>Avg Cost</th><th>Purchased</th><th>Held (days)</th><th>Price</th><th>Value</th><th>P&L</th><th>Return</th><th>Weight</th><th>Vol</th></tr></thead><tbody id="pf-tbl"></tbody></table></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem">
        <div class="cc" style="margin:0"><div class="ct">Asset Allocation</div><canvas id="pf-alloc" style="max-height:200px"></canvas></div>
        <div class="cc" style="margin:0"><div class="ct">Sector Allocation</div><canvas id="pf-sector" style="max-height:200px"></canvas></div>
      </div>
      <div class="cc" id="pf-bm-wrap" style="display:none"><div class="cv-toolbar"><div class="ct">Portfolio vs Benchmark</div><div style="display:flex;gap:.25rem">
  <button class="cv-btn cv-active" onclick="setCvView('pf-bm',this,'D')">D</button>
  <button class="cv-btn" onclick="setCvView('pf-bm',this,'M')">M</button>
  <button class="cv-btn" onclick="setCvView('pf-bm',this,'Y')">Y</button>
  <button class="cv-btn" onclick="toggleCvRange('pf-bm')">📅 Range</button>
</div></div>
<div class="cv-range" id="pf-bm-range-wrap">
  <input type="date" id="pf-bm-range-from"/><span style="font-size:.72rem;color:var(--mut)">to</span>
  <input type="date" id="pf-bm-range-to"/>
  <button onclick="applyCvRange('pf-bm')">Apply</button>
</div>
<canvas id="pf-bm-chart"></canvas></div>
      <div class="card" style="margin-bottom:1rem"><div class="ch">Correlation Matrix</div><div class="cb" style="overflow-x:auto"><div id="pf-corr"></div></div></div>
      <div class="card" id="pf-ir-card" style="display:none;margin-bottom:1rem">
        <div class="ch">Information Ratio &amp; Appraisal Ratio</div>
        <div class="cb" id="pf-ir-body"></div>
      </div>
      <div class="card" id="pf-tgt-card" style="display:none;margin-bottom:1rem"><div class="ch">Return Target</div><div class="cb" id="pf-tgt-body"></div></div>
      <div class="card"><div class="ch">Notes &amp; Thoughts</div><div class="cb">
        <div class="fd"><label>Portfolio Notes</label><textarea id="pf-notes" placeholder="Observations…"></textarea></div>
        <div class="fd"><label>Thesis</label><textarea id="pf-thesis" placeholder="Strategy…"></textarea></div>
        <div style="display:flex;gap:.5rem"><button class="btn bs" onclick="saveRec('pf')">💾 Save</button><button class="btn bg" onclick="openShare('pf')">↗ Share</button></div>
      </div></div>
    </div>
  </main>
</div></div>

<!-- LIVE MONITOR -->
<div id="pg-monitor" class="pg">
<div style="padding:1rem;max-width:1100px">

  <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:1rem;flex-wrap:wrap;gap:.75rem">
    <div>
      <div style="font-size:1rem;font-weight:700">Live Monitoring</div>
      <div style="font-size:.78rem;color:var(--mut)">Conditions checked daily at 7:00 AM AEST. Alerts sent via email &amp; Google Calendar when matched.</div>
    </div>
    <button class="btn bp" style="width:auto" onclick="saveMonitorConfig()">💾 Save Config</button>
  </div>

  <!-- Notification Settings -->
  <div class="card" style="margin-bottom:1rem">
    <div class="ch">Notification Settings</div>
    <div class="cb">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
        <div>
          <div class="sl">Email</div>
          <div class="fd"><label>Your Email Address</label><input type="email" id="mon-email" placeholder="you@example.com"/></div>
          <div class="fd"><label>SMTP Host</label><input id="mon-smtp-host" placeholder="smtp.gmail.com"/></div>
          <div class="r2">
            <div class="fd"><label>SMTP Port</label><input type="number" id="mon-smtp-port" value="587"/></div>
            <div class="fd"><label>From Email</label><input id="mon-smtp-from" placeholder="alerts@yourdomain.com"/></div>
          </div>
          <div class="fd"><label>SMTP Password / App Password</label><input type="password" id="mon-smtp-pass" placeholder="Gmail app password…"/></div>
          <div style="font-size:.68rem;color:var(--mut)">For Gmail: use an App Password (Google Account → Security → App Passwords)</div>
        </div>
        <div>
          <div class="sl">Google Calendar</div>
          <div class="fd"><label>Calendar ID</label><input id="mon-cal-id" placeholder="yourname@gmail.com or calendar ID"/></div>
          <div class="fd"><label>Service Account JSON Key (path or paste)</label><textarea id="mon-cal-key" placeholder='Paste service account JSON key here, or leave blank to skip Google Calendar...' style="min-height:100px;font-size:.72rem;font-family:monospace"></textarea></div>
          <div style="font-size:.68rem;color:var(--mut)">Create a Google Cloud service account with Calendar API access. Share your calendar with the service account email.</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Watchlist Builder -->
  <div class="card" style="margin-bottom:1rem">
    <div class="ch" style="display:flex;justify-content:space-between;align-items:center">
      <span>Watchlist Conditions</span>
      <button class="btn bg" style="padding:.28rem .7rem;font-size:.75rem" onclick="addMonitorItem()">+ Add Security</button>
    </div>
    <div class="cb">
      <div style="font-size:.72rem;color:var(--mut);margin-bottom:.85rem">
        Each row is one security. Set the conditions to watch — leave fields blank to skip. Use AND/OR to combine.
        The system fetches live data via Yahoo Finance each day at 7 AM AEST and checks whether conditions are met.
      </div>
      <div id="mon-list"></div>
    </div>
  </div>

  <!-- Status / Log -->
  <div class="card" style="margin-bottom:1rem">
    <div class="ch" style="display:flex;justify-content:space-between;align-items:center">
      <span>Status &amp; Alert Log</span>
      <div style="display:flex;gap:.5rem">
        <button class="btn bg" style="padding:.28rem .7rem;font-size:.75rem" onclick="testMonitor()">▶ Test Now</button>
        <button class="btn bg" style="padding:.28rem .7rem;font-size:.75rem" onclick="clearMonitorLog()">Clear Log</button>
      </div>
    </div>
    <div class="cb">
      <div style="display:flex;align-items:center;gap:.65rem;margin-bottom:.75rem">
        <div id="mon-status-dot" style="width:10px;height:10px;border-radius:50%;background:#CBD5E1;flex-shrink:0"></div>
        <div id="mon-status-txt" style="font-size:.8rem;color:var(--mut)">Not configured</div>
        <div id="mon-next-run" style="font-size:.72rem;color:var(--mut);margin-left:auto"></div>
      </div>
      <div id="mon-log" style="background:var(--sur2);border:1px solid var(--bdr);border-radius:6px;padding:.65rem;font-size:.75rem;font-family:monospace;max-height:220px;overflow-y:auto;line-height:1.8;color:var(--txt)">
        <span style="color:var(--mut)">No alerts yet. Configure conditions above and hit Save Config, or Test Now to run immediately.</span>
      </div>
    </div>
  </div>

</div>
</div>

<!-- FINANCIALS -->
<div id="pg-financials" class="pg">
<div style="padding:1rem;max-width:1200px">

  <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:1rem;flex-wrap:wrap;gap:.75rem">
    <div>
      <div style="font-size:1rem;font-weight:700">Financial Statements</div>
      <div style="font-size:.78rem;color:var(--mut)">Powered by Calcbench — latest 10-K &amp; 10-Q filings, income statement, balance sheet, cash flow, and management commentary.</div>
    </div>
    <a href="https://www.calcbench.com" target="_blank" style="font-size:.75rem;color:var(--acc);text-decoration:none;border:1px solid var(--bdr);padding:.3rem .7rem;border-radius:6px;background:var(--sur)">↗ Open Calcbench</a>
  </div>

  <!-- Credentials + Search -->
  <div class="card" style="margin-bottom:1rem">
    <div class="ch">Calcbench Credentials &amp; Search</div>
    <div class="cb">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:.75rem;align-items:end">
        <div class="fd" style="margin:0"><label>Calcbench Email</label><input type="email" id="cb-email" placeholder="you@example.com"/></div>
        <div class="fd" style="margin:0"><label>Password</label><input type="password" id="cb-pass" placeholder="Calcbench password"/></div>
        <div class="fd" style="margin:0"><label>Ticker Symbol</label><input id="cb-ticker" placeholder="e.g. AAPL, MSFT" style="text-transform:uppercase"/></div>
        <button class="btn bp" style="width:auto;white-space:nowrap" id="cb-run" onclick="runFinancials()">▶ Fetch</button>
      </div>
      <div style="font-size:.68rem;color:var(--mut);margin-top:.5rem">
        Credentials stored in-session only, never persisted. 
        <a href="https://www.calcbench.com/join" target="_blank" style="color:var(--acc)">Sign up for Calcbench →</a>
      </div>
      <div class="er" id="cb-err"></div>
    </div>
  </div>

  <div class="sw" id="cb-sw" style="display:none"><div class="sp"></div><p>Fetching from Calcbench…</p></div>

  <div id="cb-res" style="display:none">

    <!-- Stock header -->
    <div class="sb" style="margin-bottom:1rem">
      <div class="sn" id="cb-company-name">—</div>
      <div class="st" id="cb-company-ticker">—</div>
      <div class="sp2" id="cb-filing-date">—</div>
      <a id="cb-source-link" href="#" target="_blank" style="margin-left:auto;font-size:.72rem;color:var(--acc);text-decoration:none;border:1px solid var(--bdr);padding:.2rem .55rem;border-radius:5px;background:var(--sur)">↗ View on Calcbench</a>
    </div>

    <!-- Filing selector -->
    <div class="card" style="margin-bottom:1rem">
      <div class="ch" style="display:flex;justify-content:space-between;align-items:center">
        <span>Filing</span>
        <div style="display:flex;gap:.4rem">
          <select id="cb-filing-sel" onchange="switchFiling()" style="font-size:.78rem;padding:.3rem .5rem;width:auto"></select>
        </div>
      </div>
    </div>

    <!-- Tabs -->
    <div style="display:flex;gap:.2rem;margin-bottom:1rem;flex-wrap:wrap" id="cb-tabs">
      <button class="nb active" onclick="showFinTab('income',this)">Income Statement</button>
      <button class="nb" onclick="showFinTab('balance',this)">Balance Sheet</button>
      <button class="nb" onclick="showFinTab('cashflow',this)">Cash Flow</button>
      <button class="nb" onclick="showFinTab('commentary',this)">Management Commentary</button>
    </div>

    <!-- Income Statement -->
    <div id="cb-tab-income" class="cb-tab">
      <div class="card">
        <div class="ch" id="cb-income-title">Income Statement</div>
        <div class="cb" style="overflow-x:auto"><div id="cb-income-body"></div></div>
      </div>
    </div>

    <!-- Balance Sheet -->
    <div id="cb-tab-balance" class="cb-tab" style="display:none">
      <div class="card">
        <div class="ch" id="cb-balance-title">Balance Sheet</div>
        <div class="cb" style="overflow-x:auto"><div id="cb-balance-body"></div></div>
      </div>
    </div>

    <!-- Cash Flow -->
    <div id="cb-tab-cashflow" class="cb-tab" style="display:none">
      <div class="card">
        <div class="ch" id="cb-cashflow-title">Cash Flow Statement</div>
        <div class="cb" style="overflow-x:auto"><div id="cb-cashflow-body"></div></div>
      </div>
    </div>

    <!-- Management Commentary -->
    <div id="cb-tab-commentary" class="cb-tab" style="display:none">
      <div class="card">
        <div class="ch">Management Commentary &amp; Disclosures</div>
        <div class="cb">
          <div style="font-size:.72rem;color:var(--mut);margin-bottom:.75rem">
            Key text disclosures from the filing — MD&amp;A, risk factors, business overview. Click any section to expand.
          </div>
          <div id="cb-commentary-body"></div>
        </div>
      </div>
    </div>

  </div>
</div>
</div>

<!-- RECORDS -->
<div id="pg-records" class="pg">
<div style="padding:1rem">
  <div style="font-size:.95rem;font-weight:700;margin-bottom:.75rem">Saved Records</div>
  <input id="rec-search" placeholder="Search…" oninput="filterRecs()" style="margin-bottom:.85rem"/>
  <div id="rec-list"></div>
  <div class="ph" id="rec-empty" style="display:none"><div class="ic">🗂</div><p>No records yet.</p></div>
</div></div>

<!-- SHARE MODAL -->
<div class="mo" id="shareModal">
  <div class="md">
    <h3>Share Analysis <button onclick="closeShare()" style="float:right;background:none;border:none;cursor:pointer;color:var(--mut);font-size:1.1rem">✕</button></h3>
    <div class="stx" id="share-text"></div>
    <div class="sg2">
      <div class="shb" onclick="shareVia('copy')">📋 Copy</div>
      <div class="shb" onclick="shareVia('email')">✉️ Email</div>
      <div class="shb" onclick="shareVia('whatsapp')">💬 WhatsApp</div>
      <div class="shb" onclick="shareVia('telegram')">✈️ Telegram</div>
      <div class="shb" onclick="shareVia('twitter')">🐦 X/Twitter</div>
      <div class="shb" onclick="shareVia('linkedin')">💼 LinkedIn</div>
    </div>
    <button class="btn bg" style="width:100%" onclick="closeShare()">Close</button>
  </div>
</div>

<script>
// ══════════════════════════════════════════════════════
// FINANCIALS — Calcbench integration
// ══════════════════════════════════════════════════════
let cbData = null; // full response from server

function showFinTab(tab, btn){
  document.querySelectorAll('.cb-tab').forEach(t=>t.style.display='none');
  document.querySelectorAll('#cb-tabs .nb').forEach(b=>b.classList.remove('active'));
  document.getElementById('cb-tab-'+tab).style.display='block';
  btn.classList.add('active');
}

async function runFinancials(){
  const btn    = document.getElementById('cb-run');
  const errEl  = document.getElementById('cb-err');
  const sw     = document.getElementById('cb-sw');
  const resEl  = document.getElementById('cb-res');
  const email  = (document.getElementById('cb-email').value||'').trim();
  const pass   = document.getElementById('cb-pass').value||'';
  const ticker = (document.getElementById('cb-ticker').value||'').trim().toUpperCase();

  // Reset state
  errEl.style.display='none'; errEl.textContent='';
  errEl.style.background=''; errEl.style.borderColor=''; errEl.style.color='';

  if(!email){ errEl.textContent='⚠ Enter your Calcbench email.'; errEl.style.display='block'; return; }
  if(!pass){  errEl.textContent='⚠ Enter your Calcbench password.'; errEl.style.display='block'; return; }
  if(!ticker){errEl.textContent='⚠ Enter a ticker symbol.'; errEl.style.display='block'; return; }

  resEl.style.display='none';
  sw.style.display='flex';
  btn.disabled=true; btn.textContent='Fetching…';

  try{
    console.log('[Financials] Fetching', ticker, 'for', email);
    const r = await fetch('/financials/fetch', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({email, password:pass, ticker})
    });
    console.log('[Financials] HTTP status:', r.status);

    // Handle non-JSON responses gracefully
    const text = await r.text();
    console.log('[Financials] Raw response:', text.slice(0,200));
    let data;
    try { data = JSON.parse(text); }
    catch(pe){ throw new Error('Server returned unexpected response (status '+r.status+'). Check Railway logs.'); }

    if(data.error) throw new Error(data.error);
    cbData = data;
    renderFinancials(data);
    sw.style.display='none';
    resEl.style.display='block';
  } catch(e){
    sw.style.display='none';
    errEl.textContent='⚠ '+e.message;
    errEl.style.display='block';
    console.error('[Financials] Error:', e);
  } finally {
    btn.disabled=false; btn.textContent='▶ Fetch';
  }
}

async function testCbConnection(){
  const email  = (document.getElementById('cb-email').value||'').trim();
  const pass   = document.getElementById('cb-pass').value||'';
  const errEl  = document.getElementById('cb-err');
  errEl.style.display='none'; errEl.style.background=''; errEl.style.borderColor=''; errEl.style.color='';
  if(!email||!pass){ errEl.textContent='⚠ Enter email and password first.'; errEl.style.display='block'; return; }
  const btn = event.target;
  btn.textContent='Testing…'; btn.disabled=true;
  try{
    const r = await fetch('/financials/test',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({email,password:pass})});
    const text = await r.text();
    let d; try{d=JSON.parse(text);}catch(e){throw new Error('Server error ('+r.status+')');}
    if(d.ok){
      errEl.textContent='✅ Connected successfully as '+email;
      errEl.style.cssText='display:block;background:#F0FDF4;border-color:#BBF7D0;color:#16A34A;border:1px solid #BBF7D0;border-radius:6px;padding:.7rem .85rem;font-size:.78rem;margin-top:.6rem';
    } else {
      errEl.textContent='⚠ '+(d.error||'Connection failed');
      errEl.style.display='block';
    }
  } catch(e){
    errEl.textContent='⚠ '+e.message;
    errEl.style.display='block';
  }
  finally{ btn.textContent='Test Connection'; btn.disabled=false; }
}

function switchFiling(){
  if(!cbData) return;
  const sel = document.getElementById('cb-filing-sel').value;
  const filing = cbData.filings.find(f=>f.id===sel);
  if(filing) renderFilingData(filing);
}

function renderFinancials(data){
  document.getElementById('cb-company-name').textContent = data.companyName || data.ticker;
  document.getElementById('cb-company-ticker').textContent = data.ticker;
  document.getElementById('cb-filing-date').textContent = data.filings?.[0]?.filedOn || '—';
  const srcLink = document.getElementById('cb-source-link');
  srcLink.href = `https://www.calcbench.com/financial_statements/${data.ticker}`;

  // Populate filing selector
  const sel = document.getElementById('cb-filing-sel');
  sel.innerHTML = (data.filings||[]).map(f=>
    `<option value="${f.id}">${f.type} — ${f.period} (filed ${f.filedOn})</option>`
  ).join('');

  // Render first filing
  if(data.filings?.[0]) renderFilingData(data.filings[0]);
}

function renderFilingData(filing){
  document.getElementById('cb-income-title').textContent  = `Income Statement — ${filing.period}`;
  document.getElementById('cb-balance-title').textContent  = `Balance Sheet — ${filing.period}`;
  document.getElementById('cb-cashflow-title').textContent = `Cash Flow Statement — ${filing.period}`;
  document.getElementById('cb-filing-date').textContent    = `${filing.type} · ${filing.period} · filed ${filing.filedOn}`;

  renderFinTable('cb-income-body',  filing.income);
  renderFinTable('cb-balance-body', filing.balance);
  renderFinTable('cb-cashflow-body',filing.cashflow);
  renderCommentary('cb-commentary-body', filing.commentary);
}

function fmtNum(v){
  if(v==null||v==='') return '—';
  const n = parseFloat(v);
  if(isNaN(n)) return v;
  const abs = Math.abs(n);
  const sign = n<0?'(':'';
  const end  = n<0?')':'';
  if(abs>=1e9) return sign+'$'+(abs/1e9).toFixed(2)+'B'+end;
  if(abs>=1e6) return sign+'$'+(abs/1e6).toFixed(1)+'M'+end;
  if(abs>=1e3) return sign+'$'+(abs/1e3).toFixed(0)+'K'+end;
  return sign+'$'+abs.toLocaleString()+end;
}

function renderFinTable(elId, rows){
  const el = document.getElementById(elId);
  if(!rows||!rows.length){ el.innerHTML='<div style="color:var(--mut);font-size:.8rem;padding:.5rem">No data available</div>'; return; }
  // Group by section
  let html = '<table style="width:100%;border-collapse:collapse;font-size:.8rem">';
  let lastSection = null;
  rows.forEach(row=>{
    if(row.section && row.section !== lastSection){
      html += `<tr><td colspan="3" style="padding:.55rem .75rem .25rem;font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--mut);background:var(--sur2);border-top:2px solid var(--bdr)">${row.section}</td></tr>`;
      lastSection = row.section;
    }
    const isTotal = row.isTotal;
    const style = isTotal ? 'font-weight:700;border-top:1px solid var(--bdr2)' : '';
    const indent = row.indent>0 ? `padding-left:${0.75+row.indent*1}rem` : 'padding-left:.75rem';
    const val = fmtNum(row.value);
    const cls = parseFloat(row.value)<0 ? 'neg' : '';
    html += `<tr>
      <td style="${indent};padding-top:.42rem;padding-bottom:.42rem;${style};border-top:1px solid var(--bdr)">${row.label}</td>
      <td style="text-align:right;padding:.42rem .75rem;${style};border-top:1px solid var(--bdr)" class="${cls}">${val}</td>
      <td style="text-align:right;padding:.42rem .75rem;color:var(--mut);font-size:.72rem;border-top:1px solid var(--bdr)">${row.unit||''}</td>
    </tr>`;
  });
  html += '</table>';
  el.innerHTML = html;
}

function renderCommentary(elId, sections){
  const el = document.getElementById(elId);
  if(!sections||!sections.length){
    el.innerHTML='<div style="color:var(--mut);font-size:.8rem">No commentary available for this filing.</div>';
    return;
  }
  el.innerHTML = sections.map((s,i)=>`
    <div style="border:1px solid var(--bdr);border-radius:8px;margin-bottom:.65rem;overflow:hidden">
      <div onclick="toggleComm(${i})" style="padding:.7rem 1rem;display:flex;justify-content:space-between;align-items:center;cursor:pointer;background:var(--sur2)">
        <div style="font-size:.82rem;font-weight:600">${s.title}</div>
        <span id="comm-icon-${i}" style="color:var(--mut);font-size:.9rem">▼</span>
      </div>
      <div id="comm-body-${i}" style="display:none;padding:.85rem 1rem;font-size:.78rem;line-height:1.75;color:var(--txt);max-height:400px;overflow-y:auto;white-space:pre-wrap">${s.text}</div>
    </div>`).join('');
}

function toggleComm(i){
  const body = document.getElementById('comm-body-'+i);
  const icon = document.getElementById('comm-icon-'+i);
  const open = body.style.display==='none';
  body.style.display = open?'block':'none';
  icon.textContent   = open?'▲':'▼';
}

// ══════════════════════════════════════════════════════
// CHART VIEW ENGINE — resample + range filter for all line charts
// ══════════════════════════════════════════════════════
const cvState = {};  // { chartKey: { view:'D'|'M'|'Y'|'R', from, to } }

// Registry: chartKey → { chartRef getter, rawData getter, rebuildFn }
const cvRegistry = {};

function cvRegister(key, getChart, getRaw, rebuild){
  cvRegistry[key] = { getChart, getRaw, rebuild };
  cvState[key] = { view:'D', from:null, to:null };
}

// Resample daily data to monthly or yearly last-value
function resample(labels, datasets, freq){
  if(freq==='D') return { labels, datasets };
  const buckets = {};
  labels.forEach((d,i)=>{
    const key = freq==='M' ? d.slice(0,7) : d.slice(0,4);
    buckets[key] = { label:d, i };  // last point in bucket wins
  });
  const keys  = Object.keys(buckets).sort();
  const newLabels = keys.map(k=>buckets[k].label);
  const newDatasets = datasets.map(ds=>({
    ...ds,
    data: keys.map(k=>{
      const v = ds.data[buckets[k].i];
      return v;
    })
  }));
  return { labels:newLabels, datasets:newDatasets };
}

// Filter by date range
function filterRange(labels, datasets, from, to){
  if(!from && !to) return { labels, datasets };
  const idxs = labels.reduce((acc,d,i)=>{
    if((!from||d>=from) && (!to||d<=to)) acc.push(i);
    return acc;
  }, []);
  return {
    labels: idxs.map(i=>labels[i]),
    datasets: datasets.map(ds=>({ ...ds, data:idxs.map(i=>ds.data[i]) }))
  };
}

function applyView(key){
  const reg = cvRegistry[key];
  if(!reg) return;
  const raw  = reg.getRaw();
  if(!raw) return;
  const st   = cvState[key];
  let { labels, datasets } = raw;
  if(st.view==='R')
    ({ labels, datasets } = filterRange(labels, datasets, st.from, st.to));
  else
    ({ labels, datasets } = resample(labels, datasets, st.view));
  const chart = reg.getChart();
  if(!chart){ reg.rebuild(labels, datasets); return; }
  chart.data.labels = labels;
  chart.data.datasets.forEach((ds,i)=>{ if(datasets[i]) ds.data = datasets[i].data; });
  // Recompute y-axis scale based on visible data (skip corr chart which has fixed range)
  if(chart.options.scales && chart.options.scales.y && key !== 'pair-corr'){
    const visVals = allVals(...datasets.map(d=>d.data));
    if(visVals.length){
      const scaled = smartScale(visVals, chart.options.scales.y._prefix||'');
      if(scaled.min !== undefined){
        chart.options.scales.y.min = scaled.min;
        chart.options.scales.y.max = scaled.max;
        chart.options.scales.y.ticks = {...chart.options.scales.y.ticks, ...scaled.ticks};
      }
    }
  }
  chart.update('none');
}

function setCvView(key, btn, freq){
  // Update active button style
  const wrap = btn.closest('.cv-toolbar');
  if(wrap) wrap.querySelectorAll('.cv-btn').forEach(b=>b.classList.remove('cv-active'));
  btn.classList.add('cv-active');
  // Hide range picker if not range mode
  const rangeWrap = document.getElementById(key+'-range-wrap');
  if(rangeWrap) rangeWrap.classList.remove('open');
  cvState[key] = { ...cvState[key], view:freq };
  applyView(key);
}

function toggleCvRange(key){
  const rangeWrap = document.getElementById(key+'-range-wrap');
  if(!rangeWrap) return;
  rangeWrap.classList.toggle('open');
}

function applyCvRange(key){
  const from = document.getElementById(key+'-range-from')?.value || null;
  const to   = document.getElementById(key+'-range-to')?.value   || null;
  cvState[key] = { view:'R', from, to };
  // Mark range button active
  const rangeWrap = document.getElementById(key+'-range-wrap');
  if(rangeWrap){
    const toolbar = rangeWrap.previousElementSibling;
    if(toolbar) toolbar.querySelectorAll('.cv-btn').forEach(b=>b.classList.remove('cv-active'));
    // mark the range btn
    const btns = toolbar ? toolbar.querySelectorAll('.cv-btn') : [];
    if(btns.length) btns[btns.length-1].classList.add('cv-active');
  }
  applyView(key);
}


// ── State
let btChart=null,valChart=null,pairChart=null,ratioChart=null;
let pfAllocChart=null,pfSecChart=null,pfBmChart=null;
let curBt=null,curVal=null,curPair=null,curPf=null;
let bmMode='hold', gate={buy:'or',sell:'or'}, shareText='';
let pfCount=0;

// ── Nav
function showPage(n,b){
  document.querySelectorAll('.pg').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nb').forEach(x=>x.classList.remove('active'));
  document.getElementById('pg-'+n).classList.add('active');
  b.classList.add('active');
  if(n==='records') renderRecs();
}

// ── Helpers
const gv=id=>{const e=document.getElementById(id);return e&&e.value!==''?parseFloat(e.value):null};
const sv=id=>{const e=document.getElementById(id);return e?e.value.trim():''};
function fmt(v,s){return(v>0?'+':'')+v+s}
function showErr(id,msg){const e=document.getElementById(id);e.textContent='⚠ '+msg;e.style.display=msg?'block':'none'}
function setUI(pfx,state){
  ['ph','sw','res'].forEach(s=>{
    const el=document.getElementById(pfx+'-'+s);
    if(!el)return;
    el.style.display=s===state?(s==='sw'?'flex':'block'):'none';
  });
}
// ── Auto-scale: compute nice min/max/step from actual data values ──
function smartScale(allValues, pre){
  const vals = allValues.filter(v=>v!=null&&!isNaN(v));
  if(!vals.length) return {};
  const raw_min = Math.min(...vals);
  const raw_max = Math.max(...vals);
  const range   = raw_max - raw_min;
  if(range === 0) return {};           // flat line — let Chart.js handle it

  // Pick a step size that gives ~8-12 ticks
  const roughStep = range / 8;
  // Round step to a nice number: 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 500 1000 …
  const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
  const niceFracs = [1, 2, 2.5, 5, 10];
  let step = magnitude;
  for(const f of niceFracs){
    const candidate = magnitude * f;
    if(range / candidate <= 12){ step = candidate; break; }
  }

  // Expand min/max to nearest step boundary with one-step padding
  const min = Math.floor(raw_min / step) * step - step;
  const max = Math.ceil(raw_max  / step) * step + step;

  const fmt = v => {
    if(pre && pre !== '') return pre + v.toLocaleString();
    // format ticks nicely based on magnitude
    if(Math.abs(v) >= 1000) return v.toLocaleString();
    if(step < 0.01) return v.toFixed(4);
    if(step < 0.1)  return v.toFixed(3);
    if(step < 1)    return v.toFixed(2);
    if(step < 10)   return v.toFixed(1);
    return Math.round(v).toLocaleString();
  };

  return { min, max,
    ticks:{ color:'#94A3B8', font:{family:'Inter',size:10},
            stepSize: step, callback: fmt }
  };
}

function chartOpts(pre, allValues){
  // allValues: optional flat array of all y-values across all datasets for this chart
  const yScale = allValues ? smartScale(allValues, pre)
    : { ticks:{ color:'#94A3B8', font:{family:'Inter',size:10},
                callback: v=>(pre||'')+v.toLocaleString() }};
  return{responsive:true,interaction:{mode:'index',intersect:false},
    plugins:{legend:{labels:{color:'#64748B',font:{family:'Inter',size:11}}},
      tooltip:{backgroundColor:'#fff',borderColor:'#E2E8F0',borderWidth:1,titleColor:'#0F172A',bodyColor:'#64748B',
        titleFont:{family:'Inter',weight:'600'},bodyFont:{family:'Inter',size:11},
        callbacks:{label:c=>(pre||'')+(typeof c.parsed.y==='number'?c.parsed.y.toLocaleString():c.parsed.y)}}},
    scales:{
      x:{ticks:{color:'#94A3B8',font:{family:'Inter',size:10},maxTicksLimit:10},grid:{color:'#F1F5F9'}},
      y:{grid:{color:'#F1F5F9'}, ...yScale}
    }};
}

// Helper: collect all non-null values across datasets
function allVals(...datasets){
  return datasets.flat().filter(v=>v!=null&&!isNaN(v));
}
// ── Three-way date calculator ──────────────────────────────────────────
// The field the user JUST edited is the trigger — the other two are solved.
// Track which field triggered via the 'changed' param.
function calcDates(pfx, changed){
  const sEl = document.getElementById(pfx+'-start');
  const eEl = document.getElementById(pfx+'-end');
  const dEl = document.getElementById(pfx+'-days');
  const hEl = document.getElementById(pfx+'-days-hint');
  if(!sEl||!eEl||!dEl) return;

  function addDays(dateStr, n){
    const d = new Date(dateStr); d.setDate(d.getDate()+n);
    return d.toISOString().slice(0,10);
  }
  function diffDays(s,e){ return Math.round((new Date(e)-new Date(s))/86400000); }
  function approxYears(d){ return '≈ '+(d/365.25).toFixed(1)+'y'; }
  function setHint(txt){ if(hEl) hEl.textContent=txt; }

  const s=sEl.value, e=eEl.value, d=dEl.value?parseInt(dEl.value):null;
  const hasS=s!=='', hasE=e!=='', hasD=d!==null&&d>0;

  if(changed==='days'){
    // User typed days — need one date to calculate the other
    if(hasS && hasD){
      eEl.value = addDays(s, d);
      setHint('→ '+eEl.value+' '+approxYears(d));
    } else if(hasE && hasD){
      sEl.value = addDays(e, -d);
      setHint('← '+sEl.value+' '+approxYears(d));
    } else if(hasD){
      setHint(approxYears(d));
    }
  } else if(changed==='start'){
    if(hasS && hasE){
      const diff=diffDays(s,e);
      if(diff>0){ dEl.value=diff; setHint(approxYears(diff)); }
    } else if(hasS && hasD){
      eEl.value=addDays(s,d);
      setHint('→ '+eEl.value);
    } else { setHint(''); }
  } else if(changed==='end'){
    if(hasS && hasE){
      const diff=diffDays(s,e);
      if(diff>0){ dEl.value=diff; setHint(approxYears(diff)); }
    } else if(hasE && hasD){
      sEl.value=addDays(e,-d);
      setHint('← '+sEl.value);
    } else { setHint(''); }
  } else {
    // Init call — just compute days from defaults
    if(hasS && hasE){
      const diff=diffDays(s,e);
      if(diff>0){ dEl.value=diff; setHint(approxYears(diff)); }
    }
  }
}

window.addEventListener('DOMContentLoaded',()=>{
  ['bt','pair'].forEach(pfx=>calcDates(pfx,null));
  // Pre-populate 2 monitor rows
  addMonitorItem(); addMonitorItem();
  // Load saved monitor config from server
  loadMonitorConfig();
});
// ───────────────────────────────────────────────────────────────────────

function tog(fldId,cbId){document.getElementById(fldId).classList.toggle('on',document.getElementById(cbId).checked)}
function applyBm(){document.getElementById('bt-bm-custom').style.display=document.getElementById('bt-bm').value==='custom'?'block':'none'}
function setBm(m){bmMode=m;document.getElementById('bm-hold').className=m==='hold'?'ao':'';document.getElementById('bm-rules').className=m==='rules'?'aa':''}
function setGate(side,m){gate[side]=m;['and','or'].forEach(x=>{document.getElementById(side+'-'+x).className=m===x?'a'+x[0]:''});}

// ── BACKTEST
async function runBacktest(){
  const btn=document.getElementById('bt-run');
  showErr('bt-err',''); setUI('bt','sw'); btn.disabled=true; btn.textContent='Running…';
  const bmp=document.getElementById('bt-bm').value;
  const bmt=bmp==='custom'?(sv('bt-bm-custom')||'SPY'):bmp;
  const payload={
    ticker:sv('bt-ticker').toUpperCase(),start:sv('bt-start'),end:sv('bt-end'),
    capital:parseFloat(sv('bt-capital'))||10000,
    rsiBuy:gv('bt-rsiBuy'),rsiSell:gv('bt-rsiSell')||70,
    rsiSellEnabled:sv('bt-rsiSell')!=='',
    profitTarget:gv('bt-profit'),stopLoss:gv('bt-stop'),
    benchmark:bmt,bmMode,
    andorLogic:{buyLogic:gate.buy,sellLogic:gate.sell},
    buyMetrics:{maxDrawdown:gv('bm-maxdd'),var95:gv('bm-var'),pe:gv('bm-pe'),pb:gv('bm-pb'),eps:gv('bm-eps'),divYield:gv('bm-divy'),roe:gv('bm-roe'),de:gv('bm-de'),fcfYield:gv('bm-fcf'),revGrowth:gv('bm-rev')},
    sellMetrics:{maxDrawdown:gv('sm-maxdd'),var95:gv('sm-var'),pe:gv('sm-pe'),pb:gv('sm-pb'),eps:gv('sm-eps'),divYield:gv('sm-divy'),roe:gv('sm-roe'),de:gv('sm-de'),fcfYield:gv('sm-fcf'),revGrowth:gv('sm-rev')},
    rollWin: parseInt(sv('bt-roll-win'))||30,
    rollHist: parseInt(sv('bt-roll-hist'))||20,
  };
  try{
    const r=await fetch('/backtest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await r.json();
    if(data.error)throw new Error(data.error);
    curBt={data,payload}; renderBt(data,payload); setUI('bt','res');
  }catch(e){showErr('bt-err',e.message);setUI('bt','ph');}
  finally{btn.disabled=false;btn.textContent='▶ Run Mandate';}
}

function renderBt(data,params){
  const s=data.stats, ticker=params.ticker.toUpperCase();
  const alpha=Math.round((s.totalReturn-s.bmReturn)*100)/100;
  const bmlbl=data.bmLabel||'Benchmark';
  document.getElementById('bt-sname').textContent=data.stockName||ticker;
  document.getElementById('bt-stick').textContent=ticker;
  document.getElementById('bt-sper').textContent=`${params.start} → ${params.end}`;
  document.getElementById('cmp-s').textContent=ticker+' Strategy';
  document.getElementById('cmp-b').textContent=bmlbl;
  document.getElementById('bt-ctitle').textContent=`Equity Curve — ${data.stockName||ticker} vs ${bmlbl}`;
  document.getElementById('bt-stats').innerHTML=[
    {l:'Strategy Return',v:fmt(s.totalReturn,'%'),s:`Final $${s.finalValue.toLocaleString()}`,c:s.totalReturn>=0?'pos':'neg'},
    {l:bmlbl,v:fmt(s.bmReturn,'%'),s:'benchmark',c:s.bmReturn>=0?'pos':'neg'},
    {l:'Alpha vs BM',v:fmt(alpha,'%'),s:'outperformance',c:alpha>=0?'pos':'neg'},
    {l:'Win Rate',v:fmt(s.winRate,'%'),s:s.totalTrades+' trades',c:'neu'},
    {l:'Avg Win',v:fmt(s.avgWin,'%'),s:'per winner',c:'pos'},
    {l:'Avg Loss',v:fmt(s.avgLoss,'%'),s:'per loser',c:'neg'},
    {l:'Sharpe',v:s.sharpe,s:'risk-adj return',c:s.sharpe>=1?'pos':'neu'},
    {l:'Max Drawdown',v:fmt(s.maxDrawdown,'%'),s:'peak→trough',c:s.maxDrawdown<=-20?'neg':'neu'},
    {l:'Days Spanned',v:s.daysSpanned,s:`${s.holdingDays}d holding · ${s.cashDays}d cash`,c:'neu'},
    {l:'Trading Days',v:s.tradingDays,s:'market open days in period',c:'neu'},
    {l:'Annualised Return',v:fmt(s.annualisedReturn,'%'),s:'CAGR over period',c:s.annualisedReturn>=0?'pos':'neg'},
    {l:'TWA Return',v:fmt(s.twa,'%'),s:'time-weighted',c:s.twa>=0?'pos':'neg'},
    {l:'MWA / IRR',v:s.irr!=null?fmt(s.irr,'%'):'—',s:'money-weighted (IRR)',c:s.irr!=null&&s.irr>=0?'pos':'neg'},
    {l:'Capital Deployed',v:s.deployedPct+'%',s:`${s.holdingDays} of ${s.daysSpanned} days`,c:'neu'},
  ].map(c=>`<div class="sc"><div class="sl2">${c.l}</div><div class="sv ${c.c}">${c.v}</div><div class="ss">${c.s}</div></div>`).join('');
  const rm=data.riskMetrics||{};
  document.getElementById('bt-cmp').innerHTML=[
    {m:'Total Return',st:fmt(s.totalReturn,'%'),bm:fmt(s.bmReturn,'%'),edge:alpha,es:'%'},
    {m:'Max Drawdown',st:fmt(s.maxDrawdown,'%'),bm:'—',edge:null},
    {m:'Sharpe Ratio',st:s.sharpe,bm:'—',edge:null},
    {m:'Win Rate',st:fmt(s.winRate,'%'),bm:'—',edge:null},
    {m:'Beta',st:rm.beta??'—',bm:'1.00',edge:null},
    {m:'Tracking Error',st:rm.trackingError?rm.trackingError+'%':'—',bm:'0%',edge:null},
    {m:'Correlation',st:rm.corrBm??'—',bm:'1.00',edge:null},
    {m:'Total Trades',st:s.totalTrades,bm:'1 (hold)',edge:null},
    {m:'Days Spanned',st:s.daysSpanned+' days',bm:s.daysSpanned+' days',edge:null},
    {m:'Trading Days',st:s.tradingDays+' days',bm:s.tradingDays+' days',edge:null},
    {m:'Annualised Return',st:fmt(s.annualisedReturn,'%'),bm:'—',edge:null},
    {m:'Time-Weighted (TWA)',st:fmt(s.twa,'%'),bm:'—',edge:null},
    {m:'Money-Weighted (MWA)',st:s.mwa!=null?fmt(s.mwa,'%'):'—',bm:'—',edge:null},
    {m:'Capital Deployed',st:s.deployedPct+'%',bm:'100%',edge:null},
    {m:'IRR (annualised)',st:s.irr!=null?fmt(s.irr,'%'):'—',bm:'—',edge:null},
    {m:'Hypothetical Return',st:s.hypoReturn!=null?fmt(s.hypoReturn,'%'):'—',bm:'—',edge:null},
  ].map(r=>{const ec=r.edge!==null?(r.edge>=0?'pos':'neg'):'';const ed=r.edge!==null?fmt(r.edge,r.es||''):'—';
    return`<tr><td style="color:var(--mut);font-size:.76rem">${r.m}</td><td style="text-align:right;font-weight:600">${r.st}</td><td style="text-align:right;color:var(--mut)">${r.bm}</td><td style="text-align:right;font-weight:600" class="${ec}">${ed}</td></tr>`;}).join('');
  if(btChart)btChart.destroy();
  // Build segment-coloured strategy line: green=holding, grey=cash
  const eqLabels = data.equity.map(e=>e.date);
  const eqVals   = data.equity.map(e=>e.value);
  const periods  = data.equity.map(e=>e.period||'cash');
  // Create point background colours
  const ptColors = periods.map(p=>p==='holding'?'#2563EB':'#94A3B8');
  const segColors = eqVals.map((_,i)=>{
    if(i===0) return periods[0]==='holding'?'rgba(37,99,235,0.9)':'rgba(148,163,184,0.6)';
    return periods[i]==='holding'?'rgba(37,99,235,0.9)':'rgba(148,163,184,0.6)';
  });
  // Annotation plugin not available — use two overlapping datasets instead
  // Dataset 1: holding periods (null where cash)
  const holdingData = eqVals.map((v,i)=>periods[i]==='holding'?v:null);
  const cashData    = eqVals.map((v,i)=>periods[i]==='cash'?v:null);
  // Fill gaps so lines connect at transitions
  // Use spanGaps to connect through nulls
  btChart=new Chart(document.getElementById('bt-chart'),{type:'line',data:{labels:eqLabels,datasets:[
    {label:'📍 Holding Period',data:holdingData,borderColor:'#2563EB',backgroundColor:'rgba(37,99,235,0.08)',borderWidth:2.5,pointRadius:0,tension:0,fill:true,spanGaps:true},
    {label:'💰 Cash (Ready to Deploy)',data:cashData,borderColor:'#94A3B8',backgroundColor:'rgba(148,163,184,0.05)',borderWidth:1.5,pointRadius:0,tension:0,fill:true,spanGaps:true},
    {label:bmlbl,data:data.bmEquity.map(e=>e.value),borderColor:'#F59E0B',borderWidth:1.5,pointRadius:0,borderDash:[5,4],fill:false,tension:0},
  ]},options:{...chartOpts('$', allVals(eqVals, data.bmEquity.map(e=>e.value))),plugins:{...chartOpts('$',allVals(eqVals,data.bmEquity.map(e=>e.value))).plugins,legend:{labels:{color:'#64748B',font:{family:'Inter',size:11},usePointStyle:true}}}}});
  // Register with ChartView engine
  cvRegister('bt',
    ()=>btChart,
    ()=>({labels:eqLabels, datasets:[
      {label:'📍 Holding Period',data:holdingData,borderColor:'#2563EB',backgroundColor:'rgba(37,99,235,0.08)',borderWidth:2.5,pointRadius:0,tension:0,fill:true,spanGaps:true},
      {label:'💰 Cash (Ready to Deploy)',data:cashData,borderColor:'#94A3B8',backgroundColor:'rgba(148,163,184,0.05)',borderWidth:1.5,pointRadius:0,tension:0,fill:true,spanGaps:true},
      {label:bmlbl,data:data.bmEquity.map(e=>e.value),borderColor:'#F59E0B',borderWidth:1.5,pointRadius:0,borderDash:[5,4],fill:false,tension:0},
    ]}),
    (lbl,ds)=>{ btChart.data.labels=lbl; ds.forEach((d,i)=>{if(btChart.data.datasets[i])btChart.data.datasets[i].data=d.data;}); btChart.update(); }
  );
  cvState['bt']={view:'D',from:null,to:null};
  document.getElementById('bt-tc').textContent=data.trades.length+' trades';
  document.getElementById('bt-trades').innerHTML=data.trades.map(t=>`<tr>
    <td>${t.entryDate}</td><td>${t.exitDate}</td><td>$${t.entryPrice}</td><td>$${t.exitPrice}</td>
    <td class="${t.returnPct>=0?'pos':'neg'}">${t.returnPct>0?'+':''}${t.returnPct}%</td>
    <td class="${t.pnl>=0?'pos':'neg'}">${t.pnl>=0?'+':'-'}$${Math.abs(t.pnl).toLocaleString()}</td>
    <td style="color:var(--mut);font-size:.74rem">${t.reason}</td></tr>`).join('');
  document.getElementById('bt-risk').innerHTML=[
    {l:'Beta vs BM',v:rm.beta??'—',s:rm.beta?(Math.abs(rm.beta-1)<.2?'tracks BM':rm.beta>1?'more volatile':'less volatile'):'',c:'neu'},
    {l:'Tracking Error',v:rm.trackingError?rm.trackingError+'%':'—',s:'annualised',c:'neu'},
    {l:'VaR 95% (1-day)',v:rm.var95?rm.var95+'%':'—',s:'historical',c:rm.var95>3?'neg':'neu'},
    {l:'Correlation',v:rm.corrBm??'—',s:'vs benchmark',c:'neu'},
  ].map(c=>`<div class="sc"><div class="sl2">${c.l}</div><div class="sv ${c.c}">${c.v}</div><div class="ss">${c.s}</div></div>`).join('');

  // Rolling metrics
  renderRolling(data.rollingMetrics);

  // ── Hypothetical Return card ──
  const hypoCard = document.getElementById('hypo-card');
  const hypoBody = document.getElementById('hypo-body');
  if(s.hypoReturn!=null && s.irr!=null){
    hypoCard.style.display='block';
    const actualFinal   = s.finalValue;
    const hypoFinal     = s.hypoFinal;
    const gap           = Math.round((hypoFinal - actualFinal)*100)/100;
    const gapPct        = Math.round((hypoFinal/actualFinal-1)*10000)/100;
    const gapCol        = gap>=0?'var(--grn)':'var(--red)';
    const deployed      = s.deployedPct;
    const undeployed    = Math.round((100-deployed)*10)/10;

    hypoBody.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-bottom:1rem">
        <div class="sc">
          <div class="sl2">IRR (Daily → Annual)</div>
          <div class="sv neu">${fmt(s.irr,'%')}</div>
          <div class="ss">Internal Rate of Return</div>
        </div>
        <div class="sc">
          <div class="sl2">Actual Final Value</div>
          <div class="sv">\\$${actualFinal.toLocaleString()}</div>
          <div class="ss">${deployed}% of days deployed</div>
        </div>
        <div class="sc">
          <div class="sl2">Hypothetical Final Value</div>
          <div class="sv ${gap>=0?'pos':'neg'}">\\$${hypoFinal.toLocaleString()}</div>
          <div class="ss">if 100% of days invested at same IRR</div>
        </div>
      </div>
      <div style="background:var(--sur2);border-radius:8px;padding:.85rem 1rem;font-size:.82rem;line-height:1.8;border:1px solid var(--bdr)">
        <div style="font-weight:600;margin-bottom:.35rem">What does this mean?</div>
        Your strategy achieved an IRR of <strong>${fmt(s.irr,'%')}</strong> on invested capital.
        However, your capital sat in cash for <strong>${undeployed}%</strong> of the period (${s.cashDays} days) waiting for entry conditions to be met.
        If that same IRR had been earned on every day — including cash-idle days — your \\$${(params.capital||10000).toLocaleString()} would have grown to
        <strong style="color:${gapCol}">\\$${hypoFinal.toLocaleString()}</strong>
        — a difference of <strong style="color:${gapCol}">${gap>=0?'+':''}\\$${Math.abs(gap).toLocaleString()} (${fmt(gapPct,'%')})</strong>
        vs your actual result.
        <div style="margin-top:.5rem;font-size:.75rem;color:var(--mut)">
          Note: Hypothetical return assumes no transaction costs, perfect reinvestment, and identical IRR maintained across idle periods — for illustrative purposes only.
        </div>
      </div>
    `;
  } else {
    hypoCard.style.display='none';
  }
}

// ── ROLLING METRICS ──────────────────────────────────────────────────
let rollSharpeChart=null, rollVolChart=null, rollDdChart=null;

function renderRolling(rm){
  const card = document.getElementById('rolling-card');
  if(!rm || rm.error || !rm.dates){ card.style.display='none'; return; }
  card.style.display='block';
  document.getElementById('rolling-ch-sub').textContent =
    `${rm.rollWin}-day rolling window · ${rm.rollHist}-yr history · 25th/50th/75th percentile bands`;

  // Percentile reference line helper
  function pctLine(val, n){ return val !== null ? new Array(n).fill(val) : []; }
  const n = rm.dates.length;

  // ── Helper to build chart with percentile bands ──
  function makeRollChart(canvasId, label, data, p25, p50, p75, prefix, color){
    const existing = Chart.getChart(canvasId);
    if(existing) existing.destroy();
    const visVals = data.filter(v=>v!=null);
    const autoScale = visVals.length ? smartScale(visVals.concat([p25,p50,p75].filter(v=>v!=null)), prefix) : {};
    return new Chart(document.getElementById(canvasId),{type:'line',data:{labels:rm.dates,datasets:[
      {label,data,borderColor:color,borderWidth:2,pointRadius:0,tension:.3,fill:false,spanGaps:true,order:1},
      {label:'75th pct',data:pctLine(p75,n),borderColor:'#CBD5E1',borderWidth:1,pointRadius:0,borderDash:[3,3],fill:false,order:2},
      {label:'50th pct (median)',data:pctLine(p50,n),borderColor:'#94A3B8',borderWidth:1.5,pointRadius:0,borderDash:[5,3],fill:false,order:3},
      {label:'25th pct',data:pctLine(p25,n),borderColor:'#CBD5E1',borderWidth:1,pointRadius:0,borderDash:[3,3],fill:false,order:4},
    ]},options:{...chartOpts(prefix,visVals),
      plugins:{...chartOpts(prefix,visVals).plugins,
        legend:{labels:{color:'#94A3B8',font:{family:'Inter',size:10},
          filter:item=>!['75th pct','25th pct'].includes(item.text)}}}}});
  }

  document.getElementById('rolling-body').innerHTML = `
    <div style="font-size:.72rem;color:var(--mut);margin-bottom:.85rem">
      Each point is computed over the prior <strong>${rm.rollWin} days</strong>. Dashed lines show the 25th, 50th and 75th percentile of all rolling windows across the full <strong>${rm.rollHist}-year history</strong> — so you can see whether current conditions are elevated, compressed, or typical.
    </div>
    <div class="ct" style="margin-bottom:.5rem">Rolling Sharpe Ratio (${rm.rollWin}-day)</div>
    <div class="cc" style="margin-bottom:.85rem;padding:.75rem"><canvas id="roll-sharpe-chart" style="max-height:160px"></canvas></div>
    <div class="ct" style="margin-bottom:.5rem">Rolling Volatility — annualised % (${rm.rollWin}-day)</div>
    <div class="cc" style="margin-bottom:.85rem;padding:.75rem"><canvas id="roll-vol-chart" style="max-height:160px"></canvas></div>
    <div class="ct" style="margin-bottom:.5rem">Rolling Max Drawdown % (${rm.rollWin}-day)</div>
    <div class="cc" style="margin-bottom:.85rem;padding:.75rem"><canvas id="roll-dd-chart" style="max-height:160px"></canvas></div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-top:.75rem">
      <div class="sc"><div class="sl2">Current Sharpe</div>
        <div class="sv ${rm.sharpe.filter(v=>v!=null).slice(-1)[0]>=rm.sharpeP50?'pos':'neg'}">${rm.sharpe.filter(v=>v!=null).slice(-1)[0]??'—'}</div>
        <div class="ss">median ${rm.sharpeP50} · 75th ${rm.sharpeP75}</div></div>
      <div class="sc"><div class="sl2">Current Volatility</div>
        <div class="sv ${rm.vol.filter(v=>v!=null).slice(-1)[0]<=rm.volP50?'pos':'neg'}">${rm.vol.filter(v=>v!=null).slice(-1)[0]??'—'}%</div>
        <div class="ss">median ${rm.volP50}% · 75th ${rm.volP75}%</div></div>
      <div class="sc"><div class="sl2">Current Drawdown</div>
        <div class="sv ${rm.drawdown.filter(v=>v!=null).slice(-1)[0]>=rm.ddP50?'neg':'pos'}">${rm.drawdown.filter(v=>v!=null).slice(-1)[0]??'—'}%</div>
        <div class="ss">median ${rm.ddP50}% · 25th ${rm.ddP25}%</div></div>
    </div>`;

  // Render after DOM update
  setTimeout(()=>{
    makeRollChart('roll-sharpe-chart','Rolling Sharpe',rm.sharpe,rm.sharpeP25,rm.sharpeP50,rm.sharpeP75,'','#334155');
    makeRollChart('roll-vol-chart','Rolling Volatility (%)',rm.vol,rm.volP25,rm.volP50,rm.volP75,'','#D97706');
    makeRollChart('roll-dd-chart','Rolling Max Drawdown (%)',rm.drawdown,rm.ddP25,rm.ddP50,rm.ddP75,'','#DC2626');
  }, 50);
}

// ── VALUATION
function runVal(){
  const price=gv('val-price'),eps=gv('val-eps'),gr=gv('val-gr');
  if(!price||!eps||!gr){showErr('val-err','Enter price, EPS and growth rate.');return;}
  showErr('val-err','');
  const pe=Math.round(price/eps*100)/100,peg=Math.round(pe/gr*100)/100;
  const proj=[1,2,3,4,5].map(y=>Math.round(eps*(1+gr/100)**y*100)/100);
  const dcf=proj.reduce((s,e,i)=>s+e/(1.1**(i+1)),0)+proj[4]*15/(1.1**5);
  const mos=Math.round((dcf-price)/price*10000)/100;
  const ticker=(sv('val-ticker')||'Stock').toUpperCase();
  document.getElementById('val-title').textContent=ticker+' Valuation';
  document.getElementById('val-ph').style.display='none';
  document.getElementById('val-res').style.display='block';
  document.getElementById('val-grid').innerHTML=[
    {l:'Current Price',v:'$'+price,s:'market price',c:''},
    {l:'P/E Ratio',v:pe+'×',s:'price / EPS',c:'neu'},
    {l:'PEG Ratio',v:peg+'×',s:'P/E ÷ growth',c:'neu'},
    {l:'DCF Value',v:'$'+Math.round(dcf*100)/100,s:'5yr estimate',c:'neu'},
    {l:'Margin of Safety',v:(mos>=0?'+':'')+mos+'%',s:mos>=0?'undervalued':'overvalued',c:mos>=0?'pos':'neg'},
    {l:'Growth Rate',v:gr+'%',s:'annual',c:''},
  ].map(c=>`<div class="sc"><div class="sl2">${c.l}</div><div class="sv ${c.c}">${c.v}</div><div class="ss">${c.s}</div></div>`).join('');
  if(valChart)valChart.destroy();
  valChart=new Chart(document.getElementById('val-chart'),{type:'bar',data:{labels:['Y1','Y2','Y3','Y4','Y5'],datasets:[{label:'EPS',data:proj,backgroundColor:'#DBEAFE',borderColor:'#2563EB',borderWidth:1.5,borderRadius:4}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#64748B',font:{family:'Inter',size:11}},grid:{display:false}},y:{ticks:{color:'#64748B',font:{family:'Inter',size:11},callback:v=>'$'+v},grid:{color:'#F1F5F9'}}}}});
  curVal={price,eps,gr,pe,peg,dcf:Math.round(dcf*100)/100,mos,ticker,proj};
}

// ── PAIR
let rollingCorrChart=null,ctxChart=null;
async function runPair(){
  const btn=document.getElementById('pair-run');
  showErr('pair-err',''); setUI('pair','sw'); btn.disabled=true; btn.textContent='Running…';
  const t1=sv('pair-t1').toUpperCase(), t2=sv('pair-t2').toUpperCase();
  const window_days=parseInt(sv('pair-window'))||60;
  const custom=sv('pair-custom').toUpperCase();
  try{
    const r=await fetch('/pair',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ticker1:t1,ticker2:t2,start:sv('pair-start'),end:sv('pair-end'),
        window:window_days,custom:custom||null})});
    const data=await r.json();
    if(data.error)throw new Error(data.error);

    // ── Stats cards ──
    const lastCorr=data.rollingCorr[data.rollingCorr.length-1];
    const firstCorr=data.rollingCorr.find(v=>v!==null);
    const corrTrend=lastCorr>firstCorr?'rising':lastCorr<firstCorr?'falling':'flat';
    document.getElementById('pair-win-lbl').textContent=window_days;
    document.getElementById('pair-stats').innerHTML=[
      {l:t1+' Return',v:fmt(data.ret1,'%'),c:data.ret1>=0?'pos':'neg'},
      {l:t2+' Return',v:fmt(data.ret2,'%'),c:data.ret2>=0?'pos':'neg'},
      {l:'Current Correlation',v:lastCorr!=null?lastCorr:'—',s:lastCorr>=.7?'High +ve':lastCorr<=-0.7?'High -ve':'Low/Moderate',c:'neu'},
      {l:'Corr Trend',v:corrTrend==='rising'?'↑ Rising':corrTrend==='falling'?'↓ Falling':'→ Stable',s:`from ${firstCorr} to ${lastCorr}`,c:corrTrend==='rising'?'pos':'neu'},
    ].map(c=>`<div class="sc"><div class="sl2">${c.l}</div><div class="sv ${c.c}">${c.v}</div><div class="ss">${c.s||''}</div></div>`).join('');

    // ── Convergence / Divergence banner ──
    const spread=data.ratio[data.ratio.length-1]-data.ratio[0];
    const recentSpread=data.ratio[data.ratio.length-1]-data.ratio[Math.max(0,data.ratio.length-window_days)];
    const isConverging=Math.abs(recentSpread)<Math.abs(spread)*0.3;
    const isdiverging=Math.abs(recentSpread)>Math.abs(spread)*0.5;
    const cdBanner=document.getElementById('pair-cd-banner');
    const cdText=document.getElementById('pair-cd-text');
    if(isConverging){
      cdBanner.style.background='#F0FDF4';cdBanner.style.borderColor='#BBF7D0';
      cdText.innerHTML=`<span style="font-size:.85rem;font-weight:700;color:#15803D">⟶⟵ Converging</span> <span style="font-size:.78rem;color:var(--mut);margin-left:.5rem">The price spread between ${t1} and ${t2} has narrowed recently — the pair is moving closer together. Correlation: <strong>${lastCorr}</strong>.</span>`;
    } else if(isdiverging){
      cdBanner.style.background='#FEF2F2';cdBanner.style.borderColor='#FECACA';
      cdText.innerHTML=`<span style="font-size:.85rem;font-weight:700;color:#DC2626">⟵⟶ Diverging</span> <span style="font-size:.78rem;color:var(--mut);margin-left:.5rem">The spread between ${t1} and ${t2} has widened — the pair is moving apart. Correlation: <strong>${lastCorr}</strong>.</span>`;
    } else {
      cdBanner.style.background='var(--sur2)';cdBanner.style.borderColor='var(--bdr)';
      cdText.innerHTML=`<span style="font-size:.85rem;font-weight:700;color:var(--mut)">↔ Sideways</span> <span style="font-size:.78rem;color:var(--mut);margin-left:.5rem">No strong convergence or divergence recently. Correlation: <strong>${lastCorr}</strong>.</span>`;
    }

    // ── Normalised price chart ──
    document.getElementById('pair-ct-sub').textContent=`${t1} vs ${t2}`;
    if(pairChart)pairChart.destroy();
    pairChart=new Chart(document.getElementById('pair-chart'),{type:'line',data:{labels:data.dates,datasets:[
      {label:t1,data:data.price1,borderColor:'#2563EB',borderWidth:2,pointRadius:0,tension:.1,fill:false},
      {label:t2,data:data.price2,borderColor:'#F59E0B',borderWidth:2,pointRadius:0,tension:.1,fill:false}
    ]},options:chartOpts('',allVals(data.price1,data.price2))});
    cvRegister('pair-norm',()=>pairChart,
      ()=>({labels:data.dates,datasets:[
        {label:t1,data:data.price1,borderColor:'#2563EB',borderWidth:2,pointRadius:0,tension:.1,fill:false},
        {label:t2,data:data.price2,borderColor:'#F59E0B',borderWidth:2,pointRadius:0,tension:.1,fill:false}
      ]}), (lbl,ds)=>{pairChart.data.labels=lbl;ds.forEach((d,i)=>{if(pairChart.data.datasets[i])pairChart.data.datasets[i].data=d.data;});pairChart.update();});
    cvState['pair-norm']={view:'D',from:null,to:null};

    // ── Rolling correlation chart ──
    if(rollingCorrChart)rollingCorrChart.destroy();
    rollingCorrChart=new Chart(document.getElementById('rolling-corr-chart'),{type:'line',data:{labels:data.dates,datasets:[
      {label:`${window_days}-day Rolling Correlation`,data:data.rollingCorr,borderColor:'#7C3AED',borderWidth:2,pointRadius:0,tension:.3,fill:false,spanGaps:true},
      {label:'Zero line',data:data.dates.map(()=>0),borderColor:'#E2E8F0',borderWidth:1,pointRadius:0,borderDash:[4,4],fill:false},
    ]},options:{...chartOpts(''),scales:{...chartOpts('').scales,y:{...chartOpts('').scales.y,min:-1,max:1,ticks:{...chartOpts('').scales.y.ticks,callback:v=>v.toFixed(1)}}}}});
    cvRegister('pair-corr',()=>rollingCorrChart,
      ()=>({labels:data.dates,datasets:[
        {label:`${window_days}-day Rolling Correlation`,data:data.rollingCorr,borderColor:'#7C3AED',borderWidth:2,pointRadius:0,tension:.3,fill:false,spanGaps:true},
        {label:'Zero line',data:data.dates.map(()=>0),borderColor:'#E2E8F0',borderWidth:1,pointRadius:0,borderDash:[4,4],fill:false}
      ]}), (lbl,ds)=>{rollingCorrChart.data.labels=lbl;ds.forEach((d,i)=>{if(rollingCorrChart.data.datasets[i])rollingCorrChart.data.datasets[i].data=d.data;});rollingCorrChart.update();});
    cvState['pair-corr']={view:'D',from:null,to:null};

    // ── Price ratio chart ──
    if(ratioChart)ratioChart.destroy();
    ratioChart=new Chart(document.getElementById('ratio-chart'),{type:'line',data:{labels:data.dates,datasets:[
      {label:t1+'/'+t2+' ratio',data:data.ratio,borderColor:'#0891B2',borderWidth:1.5,pointRadius:0,tension:.1,fill:'origin',backgroundColor:'rgba(8,145,178,.05)'}
    ]},options:chartOpts('',allVals(data.ratio))});
    cvRegister('pair-ratio',()=>ratioChart,
      ()=>({labels:data.dates,datasets:[{label:t1+'/'+t2+' ratio',data:data.ratio,borderColor:'#0891B2',borderWidth:1.5,pointRadius:0,tension:.1,fill:'origin',backgroundColor:'rgba(8,145,178,.05)'}]}),
      (lbl,ds)=>{ratioChart.data.labels=lbl;ds.forEach((d,i)=>{if(ratioChart.data.datasets[i])ratioChart.data.datasets[i].data=d.data;});ratioChart.update();});
    cvState['pair-ratio']={view:'D',from:null,to:null};

    // ── Context chart ──
    document.getElementById('pair-ctx-lbl').textContent=data.ctxCustomTicker?' · '+data.ctxCustomTicker:'';
    const ctxDatasets=[
      {label:'S&P 500 (SPY)',data:data.ctx.spy,borderColor:'#2563EB',borderWidth:1.5,pointRadius:0,tension:.1,fill:false},
      {label:'10Y Bond (TLT)',data:data.ctx.tlt,borderColor:'#16A34A',borderWidth:1.5,pointRadius:0,tension:.1,fill:false},
      {label:'Gold (GLD)',data:data.ctx.gld,borderColor:'#D97706',borderWidth:1.5,pointRadius:0,tension:.1,fill:false},
    ];
    if(data.ctx.custom&&data.ctxCustomTicker){
      ctxDatasets.push({label:data.ctxCustomTicker,data:data.ctx.custom,borderColor:'#DC2626',borderWidth:1.5,pointRadius:0,borderDash:[4,3],tension:.1,fill:false});
    }
    if(ctxChart)ctxChart.destroy();
    const ctxAllVals=allVals(...ctxDatasets.map(d=>d.data));
    ctxChart=new Chart(document.getElementById('ctx-chart'),{type:'line',data:{labels:data.dates,datasets:ctxDatasets},options:chartOpts('',ctxAllVals)});
    cvRegister('pair-ctx',()=>ctxChart,
      ()=>({labels:data.dates,datasets:ctxDatasets}),
      (lbl,ds)=>{ctxChart.data.labels=lbl;ds.forEach((d,i)=>{if(ctxChart.data.datasets[i])ctxChart.data.datasets[i].data=d.data;});ctxChart.update();});
    cvState['pair-ctx']={view:'D',from:null,to:null};

    // ── Regime interpretation ──
    const spy_ret=data.ctxReturns.spy, tlt_ret=data.ctxReturns.tlt, gld_ret=data.ctxReturns.gld;
    let regime='Unclear / mixed signals';
    let regimeColor='var(--mut)';
    let regimeDetail='';
    if(spy_ret<-10&&tlt_ret>5&&gld_ret>5){
      regime='⚠️ Risk-Off / Stress';regimeColor='#DC2626';
      regimeDetail='Equities falling, bonds and gold rising — classic flight-to-safety pattern. High correlation between the two securities during this period may reflect shared macro risk. Divergence may signal relative safe-haven demand.';
    } else if(spy_ret>10&&tlt_ret<0&&gld_ret<0){
      regime='✅ Risk-On / Growth';regimeColor='#16A34A';
      regimeDetail='Equities strong, bonds and gold soft — typical risk-on environment. Pairs correlating here likely have shared growth exposure. Divergence may signal sector rotation.';
    } else if(tlt_ret<-10){
      regime='📈 Rising Rates Environment';regimeColor='#D97706';
      regimeDetail='Bonds selling off sharply — rates rising. Rate-sensitive sectors may show unusual correlation patterns. Divergence between the pair could reflect different rate sensitivities.';
    } else if(gld_ret>15){
      regime='🥇 Inflation / Uncertainty Hedge';regimeColor='#D97706';
      regimeDetail='Gold outperforming strongly — inflation concerns or geopolitical uncertainty. Correlations may be elevated across risk assets.';
    } else if(Math.abs(spy_ret)<5&&Math.abs(tlt_ret)<5){
      regime='😴 Low Volatility / Sideways';regimeColor='#64748B';
      regimeDetail='Markets relatively calm. Correlation patterns between the pair are more likely driven by security-specific factors than macro conditions.';
    }
    document.getElementById('pair-regime').innerHTML=`
      <div style="font-size:.9rem;font-weight:700;color:${regimeColor};margin-bottom:.5rem">${regime}</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;margin-bottom:.75rem">
        <div class="sc"><div class="sl2">S&P 500</div><div class="sv ${spy_ret>=0?'pos':'neg'}">${fmt(spy_ret,'%')}</div><div class="ss">period return</div></div>
        <div class="sc"><div class="sl2">10Y Bond (TLT)</div><div class="sv ${tlt_ret>=0?'pos':'neg'}">${fmt(tlt_ret,'%')}</div><div class="ss">period return</div></div>
        <div class="sc"><div class="sl2">Gold (GLD)</div><div class="sv ${gld_ret>=0?'pos':'neg'}">${fmt(gld_ret,'%')}</div><div class="ss">period return</div></div>
      </div>
      ${data.ctxCustomTicker&&data.ctxReturns.custom!=null?`<div style="font-size:.78rem;color:var(--mut);margin-bottom:.5rem">${data.ctxCustomTicker}: <strong class="${data.ctxReturns.custom>=0?'pos':'neg'}">${fmt(data.ctxReturns.custom,'%')}</strong> over period</div>`:''}
      <div style="font-size:.8rem;color:var(--txt);line-height:1.7;background:var(--sur2);border-radius:7px;padding:.65rem .85rem;border:1px solid var(--bdr)">${regimeDetail||'No strong regime signal identified. Examine individual chart patterns for context.'}</div>`;

    setUI('pair','res'); curPair={data,t1,t2};
  }catch(e){showErr('pair-err',e.message);setUI('pair','ph');}
  finally{btn.disabled=false;btn.textContent='▶ Analyse Pair';}
}

// ── PORTFOLIO
function lbDays(){const v=parseInt(sv('pf-lb-val'))||1;const u=document.getElementById('pf-lb-unit').value;return{natural:v,trading:Math.round(v*1.4),weeks:v*7,months:Math.round(v*30.4),years:Math.round(v*365)}[u]||v;}
function lbHint(){const d=lbDays();document.getElementById('pf-lb-hint').textContent=`≈ ${d} calendar days`;}
document.addEventListener('DOMContentLoaded',()=>{
  document.getElementById('pf-lb-val').addEventListener('input',lbHint);
  document.getElementById('pf-lb-unit').addEventListener('change',lbHint);
  addHolding();addHolding();addHolding();
  // watchlist
  let wlCount=0;
  window.addWatchItem=function(t=''){
    wlCount++;const id=wlCount;
    const div=document.createElement('div');div.className='r2';div.id='wl-'+id;div.style.marginBottom='.35rem';
    div.innerHTML=`<input placeholder="Ticker e.g. AAPL" value="${t}" class="wl-tk" style="text-transform:uppercase;font-size:.76rem"/><button class="rmb" onclick="document.getElementById('wl-${id}').remove()" style="width:auto">✕</button>`;
    document.getElementById('wl-list').appendChild(div);
  };
  addWatchItem();addWatchItem();
});
function addHolding(t='',s='',c='',d=''){
  pfCount++;const id=pfCount;
  const div=document.createElement('div');div.className='hr';div.id='hr-'+id;
  div.innerHTML=`<input placeholder="Ticker" value="${t}" class="pf-tk" style="text-transform:uppercase;font-size:.76rem"/><input type="number" placeholder="Shares" value="${s}" class="pf-sh" style="font-size:.76rem"/><input type="number" placeholder="Avg Cost $" value="${c}" class="pf-co" style="font-size:.76rem"/><input type="date" value="${d}" class="pf-pd" title="Purchase date" style="font-size:.72rem;padding:.35rem .3rem"/><button class="rmb" onclick="document.getElementById('hr-${id}').remove()">✕</button>`;
  document.getElementById('pf-list').appendChild(div);
}
function getHoldings(){
  return [...document.querySelectorAll('.hr')].map(r=>({
    ticker:r.querySelector('.pf-tk').value.trim().toUpperCase(),
    shares:parseFloat(r.querySelector('.pf-sh').value),
    avgCost:parseFloat(r.querySelector('.pf-co').value),
    purchaseDate:r.querySelector('.pf-pd').value||null
  })).filter(h=>h.ticker&&h.shares&&h.avgCost);
}
async function runPortfolio(){
  const btn=document.getElementById('pf-run');
  const holdings=getHoldings();
  if(!holdings.length){showErr('pf-err','Add at least one holding.');return;}
  showErr('pf-err',''); setUI('pf','sw'); btn.disabled=true; btn.textContent='Running…';
  const lbV=sv('pf-lb-val'),lbU=document.getElementById('pf-lb-unit').value;
  try{
    const watchlist=[...document.querySelectorAll('.wl-tk')].map(e=>e.value.trim().toUpperCase()).filter(Boolean);
    const r=await fetch('/portfolio',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({holdings,targetReturn:document.getElementById('pf-use-target').checked?gv('pf-target'):null,benchmark:document.getElementById('pf-use-bm').checked?sv('pf-bm'):'',lookbackDays:lbDays(),lookbackLabel:lbV+' '+lbU,watchlist})});
    const data=await r.json();
    if(data.error)throw new Error(data.error);
    curPf=data; renderPf(data); setUI('pf','res');
  }catch(e){showErr('pf-err',e.message);setUI('pf','ph');}
  finally{btn.disabled=false;btn.textContent='▶ Analyse Portfolio';}
}
function renderPf(d){
  const gap=d.targetGap;
  document.getElementById('pf-stats').innerHTML=[
    {l:'Total Market Value',v:'$'+d.totalMarketValue.toLocaleString(),s:'current',c:'neu'},
    {l:'Total Cost Basis',v:'$'+d.totalCost.toLocaleString(),s:'invested',c:''},
    {l:'Unrealised P&L',v:(d.totalPnl>=0?'+':'')+' $'+Math.abs(d.totalPnl).toLocaleString(),s:(d.totalPnlPct>=0?'+':'')+d.totalPnlPct+'%',c:d.totalPnl>=0?'pos':'neg'},
    {l:'Est. Annual Return',v:fmt(d.portReturn,'%'),s:'over '+(d.lookbackLabel||'lookback'),c:d.portReturn>=0?'pos':'neg'},
    {l:'Volatility',v:d.portVol+'%',s:'annualised',c:'neu'},
    {l:'Sharpe Ratio',v:d.sharpe,s:'return/risk',c:d.sharpe>=1?'pos':'neu'},
    gap!=null?{l:'Target Gap',v:(gap>=0?'+':'')+gap+'%',s:gap>=0?'✅ on track':'⚠ below target',c:gap>=0?'pos':'neg'}:{l:'Positions',v:d.positions.length,s:'holdings',c:'neu'},
    {l:'Holdings',v:d.positions.length,s:'positions',c:'neu'},
  ].map(c=>`<div class="sc"><div class="sl2">${c.l}</div><div class="sv ${c.c}">${c.v}</div><div class="ss">${c.s}</div></div>`).join('');
  document.getElementById('pf-tbl').innerHTML=d.positions.map(p=>`<tr>
    <td><strong>${p.ticker}</strong></td><td>${p.shares}</td><td>$${p.avgCost}</td>
    <td style="color:var(--mut);font-size:.74rem">${p.purchaseDate||'—'}</td>
    <td style="color:var(--mut)">${p.holdingDays!=null?p.holdingDays+' d':'—'}</td>
    <td>$${p.currentPrice}</td><td>$${p.marketValue.toLocaleString()}</td>
    <td class="${p.pnl>=0?'pos':'neg'}">${p.pnl>=0?'+':'-'}$${Math.abs(p.pnl).toLocaleString()} (${p.pnlPct}%)</td>
    <td class="${p.returnSincePurchase!=null?p.returnSincePurchase>=0?'pos':'neg':p.return1y>=0?'pos':'neg'}">${p.returnSincePurchase!=null?fmt(p.returnSincePurchase,'%'):fmt(p.return1y,'%')}</td>
    <td>${p.weight}%</td><td class="${p.volatility>30?'neg':''}">${p.volatility}%</td></tr>`).join('');
  const COLS=['#2563EB','#7C3AED','#059669','#D97706','#DC2626','#0891B2','#BE185D','#65A30D','#9333EA','#EA580C'];
  if(pfAllocChart)pfAllocChart.destroy();
  pfAllocChart=new Chart(document.getElementById('pf-alloc'),{type:'doughnut',data:{labels:d.positions.map(p=>p.ticker),datasets:[{data:d.positions.map(p=>p.weight),backgroundColor:COLS.slice(0,d.positions.length),borderWidth:2,borderColor:'#fff'}]},options:{responsive:true,cutout:'60%',plugins:{legend:{position:'right',labels:{color:'#64748B',font:{family:'Inter',size:11},padding:8}}}}});
  if(pfSecChart)pfSecChart.destroy();
  const sl=Object.keys(d.sectorWeights),sv2=Object.values(d.sectorWeights);
  pfSecChart=new Chart(document.getElementById('pf-sector'),{type:'doughnut',data:{labels:sl,datasets:[{data:sv2,backgroundColor:COLS.slice(0,sl.length),borderWidth:2,borderColor:'#fff'}]},options:{responsive:true,cutout:'60%',plugins:{legend:{position:'right',labels:{color:'#64748B',font:{family:'Inter',size:11},padding:8}}}}});
  if(d.benchmark){
    document.getElementById('pf-bm-wrap').style.display='block';
    if(pfBmChart)pfBmChart.destroy();
    pfBmChart=new Chart(document.getElementById('pf-bm-chart'),{type:'line',data:{labels:d.benchmark.dates,datasets:[{label:'Portfolio',data:d.benchmark.portCurve,borderColor:'#2563EB',borderWidth:2,pointRadius:0,tension:.1,fill:false},{label:d.benchmark.ticker,data:d.benchmark.bmCurve,borderColor:'#F59E0B',borderWidth:1.5,pointRadius:0,borderDash:[5,4],fill:false}]},options:chartOpts('',allVals(d.benchmark.portCurve,d.benchmark.bmCurve))});
    cvRegister('pf-bm',()=>pfBmChart,
      ()=>({labels:d.benchmark.dates,datasets:[{label:'Portfolio',data:d.benchmark.portCurve,borderColor:'#2563EB',borderWidth:2,pointRadius:0,tension:.1,fill:false},{label:d.benchmark.ticker,data:d.benchmark.bmCurve,borderColor:'#F59E0B',borderWidth:1.5,pointRadius:0,borderDash:[5,4],fill:false}]}),
      (lbl,ds)=>{pfBmChart.data.labels=lbl;ds.forEach((dd,i)=>{if(pfBmChart.data.datasets[i])pfBmChart.data.datasets[i].data=dd.data;});pfBmChart.update();});
    cvState['pf-bm']={view:'D',from:null,to:null};
  }
  const tickers=d.corr.tickers,matrix=d.corr.matrix;
  let ch=`<table class="ctb"><thead><tr><th></th>${tickers.map(t=>`<th>${t}</th>`).join('')}</tr></thead><tbody>`;
  matrix.forEach((row,i)=>{ch+=`<tr><th style="text-align:left;background:var(--sur2)">${tickers[i]}</th>`;row.forEach((v,j)=>{const cls=i===j?'ch':v>=.7?'ch':v>=.4?'cm':v>=0?'cl':'cn';ch+=`<td class="${cls}">${v}</td>`;});ch+='</tr>';});
  document.getElementById('pf-corr').innerHTML=ch+'</tbody></table>';
  // ── Information Ratio & Appraisal Ratio ──
  const irCard=document.getElementById('pf-ir-card');
  if(d.informationRatio!=null||d.appraisalRatios){
    irCard.style.display='block';
    let irHtml=`<div class="sg" style="grid-template-columns:repeat(3,1fr);margin-bottom:1rem">
      <div class="sc"><div class="sl2">Information Ratio</div><div class="sv ${d.informationRatio>=0?'pos':'neg'}">${d.informationRatio!=null?d.informationRatio:'—'}</div><div class="ss">Active return ÷ tracking error</div></div>
      <div class="sc"><div class="sl2">Active Return</div><div class="sv ${d.activeReturn>=0?'pos':'neg'}">${d.activeReturn!=null?fmt(d.activeReturn,'%'):'—'}</div><div class="ss">vs ${d.benchmarkUsed||'benchmark'}</div></div>
      <div class="sc"><div class="sl2">Tracking Error</div><div class="sv neu">${d.pfTrackingError!=null?d.pfTrackingError+'%':'—'}</div><div class="ss">annualised std of active returns</div></div>
    </div>`;
    if(d.appraisalRatios&&d.appraisalRatios.length){
      irHtml+=`<div style="font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--mut);margin-bottom:.5rem">Appraisal Ratio — Watchlist Securities</div>
      <div style="font-size:.68rem;color:var(--mut);margin-bottom:.65rem">Alpha ÷ residual risk (σ of unsystematic return). Higher = better risk-adjusted active contribution.</div>
      <table><thead><tr><th>Security</th><th>Alpha (%)</th><th>Beta</th><th>Residual Risk (%)</th><th>Appraisal Ratio</th><th>Interpretation</th></tr></thead><tbody>`;
      d.appraisalRatios.forEach(ar=>{
        const cls=ar.appraisalRatio>0.5?'pos':ar.appraisalRatio<0?'neg':'neu';
        const interp=ar.appraisalRatio>1?'Strong add':ar.appraisalRatio>0.5?'Moderate add':ar.appraisalRatio>0?'Weak add':'Detracts';
        irHtml+=`<tr><td><strong>${ar.ticker}</strong></td><td class="${ar.alpha>=0?'pos':'neg'}">${ar.alpha>=0?'+':''}${ar.alpha}%</td><td>${ar.beta}</td><td>${ar.residualRisk}%</td><td class="${cls}">${ar.appraisalRatio}</td><td style="color:var(--mut);font-size:.74rem">${interp}</td></tr>`;
      });
      irHtml+='</tbody></table>';
    }
    document.getElementById('pf-ir-body').innerHTML=irHtml;
  } else {irCard.style.display='none';}

  if(gap!=null&&d.targetGap!==null){
    document.getElementById('pf-tgt-card').style.display='block';
    const pct=Math.min(Math.max(d.portReturn/parseFloat(document.getElementById('pf-target').value||1)*100,0),100);
    const col=gap>=0?'#16A34A':'#DC2626';
    document.getElementById('pf-tgt-body').innerHTML=`<div style="display:flex;justify-content:space-between;margin-bottom:.4rem"><span style="color:var(--mut);font-size:.8rem">Actual return</span><span style="font-weight:700;color:${col}">${fmt(d.portReturn,'%')}</span></div><div class="tbar"><div class="tfil" style="width:${pct}%;background:${col}"></div></div><div style="font-size:.8rem;color:${col};font-weight:600;margin-top:.4rem">${gap>=0?'✅ On track — +'+gap+'% above target':'⚠ Gap of '+Math.abs(gap)+'% to close'}</div>`;
  }
}

// ── RECORDS
function saveRec(type){
  let rec={id:Date.now().toString(),type,savedAt:new Date().toISOString().slice(0,19)};
  if(type==='bt'&&curBt){rec.ticker=curBt.payload.ticker;rec.stockName=curBt.data.stockName;rec.stats=curBt.data.stats;rec.notes=sv('bt-notes');rec.thoughts=sv('bt-thesis');}
  else if(type==='val'&&curVal){rec.ticker=curVal.ticker;rec.valData=curVal;rec.notes=sv('val-notes');rec.thoughts=sv('val-thesis');}
  else if(type==='pair'&&curPair){rec.ticker=curPair.t1+'/'+curPair.t2;rec.pairData=curPair.data;rec.notes=sv('pair-notes');rec.thoughts=sv('pair-thesis');}
  else if(type==='pf'&&curPf){rec.ticker='Portfolio';rec.pfData={totalMarketValue:curPf.totalMarketValue,totalPnl:curPf.totalPnl,totalPnlPct:curPf.totalPnlPct,portReturn:curPf.portReturn,sharpe:curPf.sharpe};rec.notes=sv('pf-notes');rec.thoughts=sv('pf-thesis');}
  else{alert('Run an analysis first.');return;}
  fetch('/records',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rec)}).then(()=>alert('✅ Saved!'));
}
function renderRecs(){
  fetch('/records').then(r=>r.json()).then(recs=>{
    const list=document.getElementById('rec-list'),empty=document.getElementById('rec-empty');
    if(!recs.length){list.innerHTML='';empty.style.display='flex';return;}
    empty.style.display='none';
    list.innerHTML=recs.map(r=>`<div style="background:var(--sur);border:1px solid var(--bdr);border-radius:var(--r);padding:.85rem 1rem;margin-bottom:.6rem;box-shadow:var(--shd)">
      <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:.4rem">
        <div><span style="font-size:.9rem;font-weight:700;color:var(--acc)">${r.ticker||'—'}</span>${r.stockName?`<span style="font-size:.72rem;color:var(--mut);margin-left:.4rem">${r.stockName}</span>`:''}<div style="font-size:.67rem;color:var(--mut);margin-top:.12rem">${r.type} · ${r.savedAt}</div></div>
        <button onclick="delRec('${r.id}')" style="background:var(--rdl);color:var(--red);border:1px solid #FECACA;border-radius:5px;padding:.22rem .5rem;font-size:.68rem;cursor:pointer">Delete</button>
      </div>
      ${r.stats?`<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.35rem"><span style="background:var(--sur2);border:1px solid var(--bdr);padding:.12rem .45rem;border-radius:20px;font-size:.67rem;color:${r.stats.totalReturn>=0?'var(--grn)':'var(--red)'}">${fmt(r.stats.totalReturn,'%')}</span><span style="background:var(--sur2);border:1px solid var(--bdr);padding:.12rem .45rem;border-radius:20px;font-size:.67rem;color:var(--mut)">${r.stats.totalTrades} trades</span><span style="background:var(--sur2);border:1px solid var(--bdr);padding:.12rem .45rem;border-radius:20px;font-size:.67rem;color:var(--mut)">Sharpe ${r.stats.sharpe}</span></div>`:''}
      ${r.valData?`<div style="font-size:.75rem;color:var(--mut)">P/E: ${r.valData.pe}× · DCF: $${r.valData.dcf} · MoS: <span class="${r.valData.mos>=0?'pos':'neg'}">${r.valData.mos>=0?'+':''}${r.valData.mos}%</span></div>`:''}
      ${r.pairData?`<div style="font-size:.75rem;color:var(--mut)">${r.ticker}: ${fmt(r.pairData.ret1,'%')} vs ${fmt(r.pairData.ret2,'%')} · Corr ${r.pairData.corr}</div>`:''}
      ${r.pfData?`<div style="font-size:.75rem;color:var(--mut)">Value: $${r.pfData.totalMarketValue?.toLocaleString()} · P&L: ${r.pfData.totalPnlPct}% · Return: ${r.pfData.portReturn}%</div>`:''}
      ${r.notes?`<div style="font-size:.77rem;background:var(--sur2);border-radius:5px;padding:.45rem .65rem;margin-top:.35rem"><strong>Notes:</strong> ${r.notes}</div>`:''}
    </div>`).join('');
  });
}
function filterRecs(){const q=document.getElementById('rec-search').value.toLowerCase();fetch('/records').then(r=>r.json()).then(recs=>{const filtered=q?recs.filter(r=>JSON.stringify(r).toLowerCase().includes(q)):recs;document.getElementById('rec-list').innerHTML='';const tmp=recs;recs.length=0;recs.push(...filtered);renderRecs();recs.length=0;recs.push(...tmp);});}
function delRec(id){if(!confirm('Delete?'))return;fetch('/records/'+id,{method:'DELETE'}).then(()=>renderRecs());}

// ── LIVE MONITOR ───────────────────────────────────────────────────────
let monCount = 0;
const MON_STORE_KEY = 'imcd_monitor_config';

// Security condition row template
function addMonitorItem(cfg={}){
  monCount++;
  const id = monCount;
  const div = document.createElement('div');
  div.id = 'mon-item-'+id;
  div.style.cssText = 'background:var(--sur2);border:1px solid var(--bdr);border-radius:8px;padding:.75rem;margin-bottom:.65rem';
  div.innerHTML = `
    <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem">
      <input placeholder="Ticker (e.g. AAPL)" value="${cfg.ticker||''}" class="mon-ticker" style="width:120px;font-size:.8rem;text-transform:uppercase;font-weight:600"/>
      <div class="aog" style="width:90px;border-radius:5px;border:1px solid var(--bdr2)">
        <button class="${(cfg.logic||'or')==='and'?'aa':''}" onclick="monToggleGate(this,'and')">AND</button>
        <button class="${(cfg.logic||'or')==='or'?'ao':''}"  onclick="monToggleGate(this,'or')">OR</button>
      </div>
      <div style="font-size:.68rem;color:var(--mut);flex:1">All conditions must match (AND) or any one (OR)</div>
      <button class="rmb" onclick="document.getElementById('mon-item-${id}').remove()">✕</button>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem">
      <div><label style="font-size:.65rem;color:var(--mut);display:block;margin-bottom:.2rem">Price ≤ ($)</label>
        <input type="number" class="mon-price" placeholder="blank=skip" value="${cfg.price||''}" style="font-size:.78rem"/></div>
      <div><label style="font-size:.65rem;color:var(--mut);display:block;margin-bottom:.2rem">P/E ≤ (×)</label>
        <input type="number" class="mon-pe" placeholder="blank=skip" value="${cfg.pe||''}" style="font-size:.78rem"/></div>
      <div><label style="font-size:.65rem;color:var(--mut);display:block;margin-bottom:.2rem">P/B ≤ (×)</label>
        <input type="number" class="mon-pb" placeholder="blank=skip" value="${cfg.pb||''}" style="font-size:.78rem"/></div>
      <div><label style="font-size:.65rem;color:var(--mut);display:block;margin-bottom:.2rem">Volatility ≤ (%)</label>
        <input type="number" class="mon-vol" placeholder="blank=skip" value="${cfg.vol||''}" style="font-size:.78rem"/></div>
      <div><label style="font-size:.65rem;color:var(--mut);display:block;margin-bottom:.2rem">Drawdown ≤ (%)</label>
        <input type="number" class="mon-dd" placeholder="blank=skip" value="${cfg.dd||''}" style="font-size:.78rem"/></div>
      <div><label style="font-size:.65rem;color:var(--mut);display:block;margin-bottom:.2rem">Benchmark</label>
        <select class="mon-bm" style="font-size:.78rem">
          <option value=""${(!cfg.bm)?'selected':''}>— None —</option>
          <option value="SPY"${cfg.bm==='SPY'?'selected':''}>SPY (S&P 500)</option>
          <option value="QQQ"${cfg.bm==='QQQ'?'selected':''}>QQQ (Nasdaq)</option>
          <option value="IWM"${cfg.bm==='IWM'?'selected':''}>IWM (Russell 2000)</option>
          <option value="EFA"${cfg.bm==='EFA'?'selected':''}>EFA (Intl)</option>
          <option value="custom"${cfg.bm==='custom'?'selected':''}>Custom…</option>
        </select></div>
      <div><label style="font-size:.65rem;color:var(--mut);display:block;margin-bottom:.2rem">IR vs BM ≥</label>
        <input type="number" class="mon-ir" placeholder="blank=skip" value="${cfg.ir||''}" step="0.01" style="font-size:.78rem"/></div>
      <div><label style="font-size:.65rem;color:var(--mut);display:block;margin-bottom:.2rem">RSI ≤</label>
        <input type="number" class="mon-rsi" placeholder="blank=skip" value="${cfg.rsi||''}" style="font-size:.78rem"/></div>
      <div><label style="font-size:.65rem;color:var(--mut);display:block;margin-bottom:.2rem">Notes (label)</label>
        <input class="mon-note" placeholder="e.g. Buy signal" value="${cfg.note||''}" style="font-size:.78rem"/></div>
    </div>`;
  document.getElementById('mon-list').appendChild(div);
}

function monToggleGate(btn, mode){
  const row = btn.closest('.aog');
  row.querySelectorAll('button').forEach(b=>{ b.className=''; });
  btn.className = mode==='and'?'aa':'ao';
}

function getMonitorItems(){
  return [...document.querySelectorAll('[id^="mon-item-"]')].map(row=>({
    ticker: row.querySelector('.mon-ticker')?.value.trim().toUpperCase()||'',
    logic:  row.querySelector('.aog button.ao') ? 'or' : 'and',
    price:  row.querySelector('.mon-price')?.value||null,
    pe:     row.querySelector('.mon-pe')?.value||null,
    pb:     row.querySelector('.mon-pb')?.value||null,
    vol:    row.querySelector('.mon-vol')?.value||null,
    dd:     row.querySelector('.mon-dd')?.value||null,
    bm:     row.querySelector('.mon-bm')?.value||null,
    ir:     row.querySelector('.mon-ir')?.value||null,
    rsi:    row.querySelector('.mon-rsi')?.value||null,
    note:   row.querySelector('.mon-note')?.value||'',
  })).filter(r=>r.ticker);
}

function saveMonitorConfig(){
  const cfg = {
    email:    document.getElementById('mon-email')?.value||'',
    smtpHost: document.getElementById('mon-smtp-host')?.value||'',
    smtpPort: document.getElementById('mon-smtp-port')?.value||'587',
    smtpFrom: document.getElementById('mon-smtp-from')?.value||'',
    smtpPass: document.getElementById('mon-smtp-pass')?.value||'',
    calId:    document.getElementById('mon-cal-id')?.value||'',
    calKey:   document.getElementById('mon-cal-key')?.value||'',
    items:    getMonitorItems(),
  };
  fetch('/monitor/config', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)})
    .then(r=>r.json()).then(d=>{
      if(d.ok){ monLog('✅ Config saved. Next check: 7:00 AM AEST tomorrow.','grn'); updateMonStatus(true); }
      else monLog('❌ Save failed: '+(d.error||'unknown'),'red');
    }).catch(e=>monLog('❌ '+e.message,'red'));
}

function testMonitor(){
  monLog('⏳ Running check now…','mut');
  fetch('/monitor/run',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.results){
      d.results.forEach(r=>{
        if(r.matched) monLog(`🔔 ${r.ticker}: ${r.reason} — alert sent`,'grn');
        else monLog(`— ${r.ticker}: no match (${r.reason})`,'mut');
      });
    }
    if(d.error) monLog('❌ '+d.error,'red');
  }).catch(e=>monLog('❌ '+e.message,'red'));
}

function clearMonitorLog(){
  document.getElementById('mon-log').innerHTML='<span style="color:var(--mut)">Log cleared.</span>';
}

function monLog(msg, tone){
  const log = document.getElementById('mon-log');
  const colors = {grn:'#16A34A',red:'#DC2626',mut:'#64748B',txt:'var(--txt)'};
  const now = new Date().toLocaleTimeString('en-AU',{hour:'2-digit',minute:'2-digit'});
  log.innerHTML += `<div><span style="color:#94A3B8">${now}</span> <span style="color:${colors[tone]||colors.txt}">${msg}</span></div>`;
  log.scrollTop = log.scrollHeight;
}

function updateMonStatus(active){
  const dot = document.getElementById('mon-status-dot');
  const txt = document.getElementById('mon-status-txt');
  const nxt = document.getElementById('mon-next-run');
  if(dot){ dot.style.background = active ? '#16A34A' : '#CBD5E1'; }
  if(txt){ txt.textContent = active ? 'Active — checking daily at 7:00 AM AEST' : 'Not configured'; }
  if(nxt && active){
    // Calculate next 7AM AEST
    const now = new Date();
    const aest = new Date(now.toLocaleString('en-AU',{timeZone:'Australia/Sydney'}));
    const next = new Date(aest);
    next.setHours(7,0,0,0);
    if(aest >= next) next.setDate(next.getDate()+1);
    nxt.textContent = 'Next: '+next.toLocaleDateString('en-AU',{weekday:'short',month:'short',day:'numeric'})+' 7:00 AM AEST';
  }
}

// Load saved config on page load
function loadMonitorConfig(){
  fetch('/monitor/config').then(r=>r.json()).then(cfg=>{
    if(!cfg || cfg.error) return;
    ['email','smtp-host','smtp-port','smtp-from','smtp-pass','cal-id','cal-key'].forEach(k=>{
      const el=document.getElementById('mon-'+k.replace('-','_').replace('-','_'));
      // try both dash and underscore
      const el2=document.getElementById('mon-'+k);
      if(el2 && cfg[k.replace(/-./g,m=>m[1].toUpperCase())]) el2.value=cfg[k.replace(/-./g,m=>m[1].toUpperCase())]||'';
    });
    if(cfg.smtpHost) document.getElementById('mon-smtp-host').value=cfg.smtpHost;
    if(cfg.smtpPort) document.getElementById('mon-smtp-port').value=cfg.smtpPort;
    if(cfg.smtpFrom) document.getElementById('mon-smtp-from').value=cfg.smtpFrom;
    if(cfg.email)    document.getElementById('mon-email').value=cfg.email;
    if(cfg.calId)    document.getElementById('mon-cal-id').value=cfg.calId;
    if(cfg.calKey)   document.getElementById('mon-cal-key').value=cfg.calKey;
    // Load watchlist items
    document.getElementById('mon-list').innerHTML='';
    monCount=0;
    (cfg.items||[]).forEach(item=>addMonitorItem(item));
    if((cfg.items||[]).length) updateMonStatus(true);
  }).catch(()=>{});
}

// ── SHARE
function openShare(type){
  if(type==='bt'&&curBt){const s=curBt.data.stats;shareText=`📊 ${curBt.data.stockName} (${curBt.payload.ticker})\\n${curBt.payload.start} → ${curBt.payload.end}\\nReturn: ${fmt(s.totalReturn,'%')} | BM: ${fmt(s.bmReturn,'%')}\\nWin Rate: ${s.winRate}% | Trades: ${s.totalTrades} | Sharpe: ${s.sharpe}\\nMax DD: ${s.maxDrawdown}%`;}
  else if(type==='val'&&curVal){shareText=`🔢 ${curVal.ticker}\\nPrice: $${curVal.price} | EPS: $${curVal.eps} | Growth: ${curVal.gr}%\\nP/E: ${curVal.pe}× | DCF: $${curVal.dcf} | MoS: ${curVal.mos>=0?'+':''}${curVal.mos}%`;}
  else if(type==='pair'&&curPair){shareText=`⚖️ ${curPair.t1} vs ${curPair.t2}\\n${curPair.t1}: ${fmt(curPair.data.ret1,'%')} | ${curPair.t2}: ${fmt(curPair.data.ret2,'%')}\\nCorrelation: ${curPair.data.corr}`;}
  else if(type==='pf'&&curPf){shareText=`🏦 Portfolio\\nValue: $${curPf.totalMarketValue.toLocaleString()} | P&L: ${curPf.totalPnlPct}%\\nReturn: ${fmt(curPf.portReturn,'%')} | Sharpe: ${curPf.sharpe}`;}
  else{alert('Run an analysis first.');return;}
  document.getElementById('share-text').textContent=shareText;
  document.getElementById('shareModal').classList.add('open');
}
function closeShare(){document.getElementById('shareModal').classList.remove('open')}
function shareVia(p){
  const enc=encodeURIComponent(shareText);
  if(p==='copy'){navigator.clipboard.writeText(shareText).then(()=>alert('✅ Copied!'));return;}
  const urls={email:`mailto:?subject=Investment Analysis&body=${enc}`,whatsapp:`https://wa.me/?text=${enc}`,telegram:`https://t.me/share/url?url=&text=${enc}`,twitter:`https://twitter.com/intent/tweet?text=${enc}`,linkedin:`https://www.linkedin.com/sharing/share-offsite/?summary=${enc}`};
  window.open(urls[p],'_blank');
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/backtest", methods=["POST"])
def backtest():
    d = request.json
    try:
        r, e = run_backtest(
            ticker=(d.get("ticker") or "QQQ").upper().strip(),
            start=d.get("start","2000-01-01"), end=d.get("end","2020-01-01"),
            capital=float(d.get("capital",10000)),
            rsi_buy=float(d["rsiBuy"]) if d.get("rsiBuy") is not None else None,
            rsi_sell_enabled=bool(d.get("rsiSellEnabled",False)),
            rsi_sell=float(d.get("rsiSell",70)),
            profit_target=float(d["profitTarget"]) if d.get("profitTarget") else None,
            stop_loss=float(d["stopLoss"]) if d.get("stopLoss") else None,
            buy_metrics=d.get("buyMetrics",{}), sell_metrics=d.get("sellMetrics",{}),
            buy_logic=d.get("andorLogic",{}).get("buyLogic","or"),
            sell_logic=d.get("andorLogic",{}).get("sellLogic","or"),
            benchmark=(d.get("benchmark") or "SPY").upper().strip(),
            bm_mode=d.get("bmMode","hold"),
            extra_params={"rollWin":d.get("rollWin",30),"rollHist":d.get("rollHist",20)})
        if e: return jsonify({"error":e}), 400
        return jsonify(r)
    except Exception as ex:
        return jsonify({"error":str(ex)}), 500

@app.route("/valuation", methods=["POST"])
def valuation():
    d = request.json
    try:
        return jsonify(calc_valuation(
            float(d["price"]) if d.get("price") else None,
            float(d["eps"]) if d.get("eps") else None,
            float(d["growthRate"]) if d.get("growthRate") else None))
    except Exception as ex:
        return jsonify({"error":str(ex)}), 500

@app.route("/pair", methods=["POST"])
def pair():
    d = request.json
    try:
        r, e = run_pair((d.get("ticker1") or "").upper(), (d.get("ticker2") or "").upper(),
                        d.get("start","2010-01-01"), d.get("end","2024-01-01"),
                        window=int(d.get("window",60)),
                        custom=(d.get("custom") or "").upper().strip() or None)
        if e: return jsonify({"error":e}), 400
        return jsonify(r)
    except Exception as ex:
        return jsonify({"error":str(ex)}), 500

@app.route("/records", methods=["GET"])
def get_records():
    return jsonify(load_records())

@app.route("/records", methods=["POST"])
def add_record():
    d = request.json
    recs = load_records()
    rec  = {**d, "id":str(uuid.uuid4()), "savedAt":datetime.utcnow().isoformat()[:19]}
    recs.insert(0, rec); save_records(recs)
    return jsonify(rec)

@app.route("/records/<rid>", methods=["DELETE"])
def del_record(rid):
    save_records([r for r in load_records() if r.get("id") != rid])
    return jsonify({"ok":True})

@app.route("/portfolio", methods=["POST"])
def portfolio():
    d = request.json
    try:
        holdings  = d.get("holdings",[])
        watchlist = [t.upper().strip() for t in d.get("watchlist",[]) if t.strip()]
        if not holdings: return jsonify({"error":"No holdings"}), 400
        tickers   = [h["ticker"].upper().strip() for h in holdings]
        shares    = {h["ticker"].upper().strip():float(h["shares"]) for h in holdings}
        costs     = {h["ticker"].upper().strip():float(h["avgCost"]) for h in holdings}
        pur_dates = {h["ticker"].upper().strip():h.get("purchaseDate") for h in holdings}
        days      = int(d.get("lookbackDays",365))
        benchmark = (d.get("benchmark") or "").upper().strip()
        today     = pd.Timestamp.today()

        raw = yf.download(tickers, period=f"{days}d", progress=False, auto_adjust=True)
        if raw.empty: return jsonify({"error":"No data"}), 400
        close = raw["Close"] if len(tickers)>1 else raw["Close"].to_frame(tickers[0])
        close = close[tickers].dropna()

        latest   = {t:float(close[t].iloc[-1]) for t in tickers}
        mkt      = {t:latest[t]*shares[t] for t in tickers}
        cost_val = {t:costs[t]*shares[t] for t in tickers}
        tot_mkt  = sum(mkt.values()); tot_cost = sum(cost_val.values())
        tot_pnl  = tot_mkt - tot_cost
        weights  = {t:round(mkt[t]/tot_mkt*100,2) for t in tickers}

        positions = []
        for t in tickers:
            pnl = mkt[t]-cost_val[t]
            pd_str = pur_dates.get(t)
            holding_days = None
            return_since_purchase = None
            if pd_str:
                try:
                    pur_dt = pd.Timestamp(pd_str)
                    holding_days = (today - pur_dt).days
                    # Fetch price at purchase date for accurate return
                    hist = yf.download(t, start=pd_str, end=str(today.date()),
                                       progress=False, auto_adjust=True)
                    if not hist.empty:
                        pur_price = float(hist["Close"].squeeze().iloc[0])
                        return_since_purchase = round((latest[t]-pur_price)/pur_price*100,2)
                except: pass
            positions.append({"ticker":t,"shares":shares[t],"avgCost":costs[t],
                "purchaseDate":pd_str,"holdingDays":holding_days,
                "returnSincePurchase":return_since_purchase,
                "currentPrice":round(latest[t],2),"marketValue":round(mkt[t],2),
                "pnl":round(pnl,2),"pnlPct":round(pnl/cost_val[t]*100,2) if cost_val[t] else 0,
                "weight":weights[t],
                "return1y":round((float(close[t].iloc[-1])/float(close[t].iloc[0])-1)*100,1),
                "volatility":round(float(close[t].pct_change().dropna().std())*np.sqrt(252)*100,2)})

        rets_df = close.pct_change().dropna()
        corr    = rets_df.corr().round(3)
        wts     = pd.Series({t:weights[t]/100 for t in tickers})
        pr      = (rets_df*wts).sum(axis=1)
        pv      = round(float(pr.std())*np.sqrt(252)*100,2)
        pret    = round(float(pr.mean())*252*100,2)
        sharpe  = round(pret/pv,2) if pv else 0
        tgap    = round(float(d["targetReturn"])-pret,2) if d.get("targetReturn") else None

        # ── Information Ratio vs benchmark ──
        ir = None; active_ret = None; pf_te = None; bm_used = None
        if benchmark:
            try:
                bm_df = yf.download(benchmark, period=f"{days}d", progress=False, auto_adjust=True)
                if not bm_df.empty:
                    bm_rets = bm_df["Close"].squeeze().pct_change().dropna()
                    pr_aligned = pr.reindex(bm_rets.index).dropna()
                    bm_aligned = bm_rets.reindex(pr_aligned.index).dropna()
                    mn = min(len(pr_aligned), len(bm_aligned))
                    if mn > 20:
                        active = pr_aligned.iloc[-mn:].values - bm_aligned.iloc[-mn:].values
                        te = float(np.std(active)) * np.sqrt(252) * 100
                        ar = float(np.mean(active)) * 252 * 100
                        ir = round(ar / te, 3) if te > 0 else None
                        active_ret = round(ar, 2)
                        pf_te = round(te, 2)
                        bm_used = benchmark
            except: pass

        # ── Appraisal Ratio for watchlist securities ──
        appraisal_ratios = []
        if watchlist and benchmark:
            try:
                bm_df2 = yf.download(benchmark, period=f"{days}d", progress=False, auto_adjust=True)
                bm_r   = bm_df2["Close"].squeeze().pct_change().dropna() if not bm_df2.empty else None
                for wt in watchlist:
                    try:
                        wt_df = yf.download(wt, period=f"{days}d", progress=False, auto_adjust=True)
                        if wt_df.empty or bm_r is None: continue
                        wt_r  = wt_df["Close"].squeeze().pct_change().dropna()
                        mn2   = min(len(wt_r), len(bm_r))
                        wr    = wt_r.iloc[-mn2:].values
                        br    = bm_r.iloc[-mn2:].values
                        cov2  = np.cov(wr, br)
                        beta  = round(cov2[0,1]/cov2[1,1], 3) if cov2[1,1] else None
                        if beta is None: continue
                        alpha_daily = float(np.mean(wr)) - beta * float(np.mean(br))
                        alpha_ann   = round(alpha_daily * 252 * 100, 2)
                        # Residual (unsystematic) risk
                        residuals   = wr - (alpha_daily + beta * br)
                        resid_risk  = round(float(np.std(residuals)) * np.sqrt(252) * 100, 2)
                        appraisal   = round(alpha_ann / resid_risk, 3) if resid_risk > 0 else None
                        appraisal_ratios.append({"ticker":wt,"alpha":alpha_ann,"beta":beta,
                            "residualRisk":resid_risk,"appraisalRatio":appraisal})
                    except: continue
            except: pass

        sectors = {}
        for t in tickers:
            try: s = yf.Ticker(t).info.get("sector","Unknown") or "Unknown"
            except: s = "Unknown"
            sectors[s] = round(sectors.get(s,0)+weights[t],2)

        return jsonify({"positions":positions,"totalMarketValue":round(tot_mkt,2),
            "totalCost":round(tot_cost,2),"totalPnl":round(tot_pnl,2),
            "totalPnlPct":round(tot_pnl/tot_cost*100,2) if tot_cost else 0,
            "portReturn":pret,"portVol":pv,"sharpe":sharpe,"targetGap":tgap,
            "informationRatio":ir,"activeReturn":active_ret,
            "pfTrackingError":pf_te,"benchmarkUsed":bm_used,
            "appraisalRatios":appraisal_ratios,
            "corr":{"tickers":tickers,"matrix":corr.values.tolist()},
            "sectorWeights":sectors,"lookbackLabel":d.get("lookbackLabel","")})
    except Exception as ex:
        return jsonify({"error":str(ex)}), 500


# ══════════════════════════════════════════════════════════════════════
# LIVE MONITOR — config, runner, scheduler
# ══════════════════════════════════════════════════════════════════════
import smtplib, threading, time as _time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

MONITOR_CONFIG_FILE = "monitor_config.json"

def load_monitor_config():
    if os.path.exists(MONITOR_CONFIG_FILE):
        with open(MONITOR_CONFIG_FILE) as f:
            return json.load(f)
    return {}

def save_monitor_config_file(cfg):
    with open(MONITOR_CONFIG_FILE, "w") as f:
        json.dump(cfg, f)

def send_email(cfg, subject, body):
    """Send alert email via SMTP."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = cfg.get("smtpFrom", cfg.get("email",""))
        msg["To"]      = cfg.get("email","")
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP(cfg.get("smtpHost","smtp.gmail.com"),
                          int(cfg.get("smtpPort", 587))) as s:
            s.ehlo(); s.starttls()
            s.login(cfg.get("smtpFrom", cfg.get("email","")),
                    cfg.get("smtpPass",""))
            s.sendmail(msg["From"], msg["To"], msg.as_string())
        return True, "sent"
    except Exception as ex:
        return False, str(ex)

def add_google_calendar_event(cfg, title, description, dt_str):
    """Create a Google Calendar event using service account."""
    try:
        import json as _json
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        key_data = _json.loads(cfg.get("calKey","{}"))
        creds = service_account.Credentials.from_service_account_info(
            key_data, scopes=["https://www.googleapis.com/auth/calendar"])
        service = build("googleapiclient.discovery", "v3", credentials=creds,
                        serviceName="calendar", version="v3",
                        discoveryServiceUrl="https://www.googleapis.com/discovery/v1/apis/calendar/v3/rest")
        event = {
            "summary": title,
            "description": description,
            "start": {"dateTime": dt_str, "timeZone": "Australia/Sydney"},
            "end":   {"dateTime": dt_str, "timeZone": "Australia/Sydney"},
            "reminders": {"useDefault": False,
                "overrides": [{"method":"email","minutes":10},{"method":"popup","minutes":0}]}
        }
        service.events().insert(calendarId=cfg.get("calId","primary"), body=event).execute()
        return True, "created"
    except ImportError:
        return False, "google-api-python-client not installed"
    except Exception as ex:
        return False, str(ex)

def check_conditions(item):
    """Fetch live data for one ticker and check all conditions."""
    ticker  = item.get("ticker","").upper().strip()
    logic   = item.get("logic","or")
    if not ticker:
        return False, "no ticker"
    try:
        info   = yf.Ticker(ticker).info
        hist   = yf.download(ticker, period="60d", progress=False, auto_adjust=True)
        if hist.empty:
            return False, "no data"
        price  = float(hist["Close"].squeeze().iloc[-1])
        rets   = hist["Close"].squeeze().pct_change().dropna()
        vol    = round(float(rets.std())*np.sqrt(252)*100, 2)
        peak   = float(hist["Close"].squeeze().cummax().iloc[-1])
        drawdown = round((price-peak)/peak*100, 2)
        pe     = info.get("trailingPE")
        pb     = info.get("priceToBook")
        # RSI
        rsi_s  = compute_rsi(hist["Close"].squeeze())
        rsi_v  = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else None
        # IR vs benchmark
        ir_val = None
        bm_ticker = item.get("bm","")
        if bm_ticker and bm_ticker not in ("", "custom"):
            try:
                bm_h = yf.download(bm_ticker, period="60d", progress=False, auto_adjust=True)
                if not bm_h.empty:
                    sr = rets.iloc[-min(len(rets),len(bm_h)-1):]
                    br = bm_h["Close"].squeeze().pct_change().dropna().iloc[-len(sr):]
                    mn = min(len(sr),len(br))
                    if mn > 5:
                        active = sr.iloc[-mn:].values - br.iloc[-mn:].values
                        te = float(np.std(active))*np.sqrt(252)*100
                        ar = float(np.mean(active))*252*100
                        ir_val = round(ar/te,3) if te>0 else None
            except: pass

        checks = []
        reasons_pass = []
        reasons_fail = []

        def chk(label, cond_val, actual, fmt_actual):
            if cond_val is None or cond_val == "": return
            passed = cond_val(actual) if callable(cond_val) else False
            checks.append(passed)
            if passed: reasons_pass.append(f"{label}={fmt_actual}")
            else:       reasons_fail.append(f"{label}={fmt_actual}")

        p = lambda v,t: float(v) if v not in (None,"") else None
        chk("Price",    (lambda a: a<=float(item["price"])) if item.get("price") else None, price, f"${price:.2f}")
        chk("P/E",      (lambda a: a is not None and a<=float(item["pe"])) if item.get("pe") else None, pe, f"{round(pe,1) if pe else 'N/A'}×")
        chk("P/B",      (lambda a: a is not None and a<=float(item["pb"])) if item.get("pb") else None, pb, f"{round(pb,2) if pb else 'N/A'}×")
        chk("Vol",      (lambda a: a<=float(item["vol"])) if item.get("vol") else None, vol, f"{vol}%")
        chk("DD",       (lambda a: a>=float(item["dd"])) if item.get("dd") else None, drawdown, f"{drawdown}%")
        chk("RSI",      (lambda a: a is not None and a<=float(item["rsi"])) if item.get("rsi") else None, rsi_v, f"{round(rsi_v,1) if rsi_v else 'N/A'}")
        chk("IR",       (lambda a: a is not None and a>=float(item["ir"])) if item.get("ir") else None, ir_val, f"{ir_val}")

        if not checks:
            return False, "no conditions set"

        matched = all(checks) if logic=="and" else any(checks)
        reason  = ("ALL met: " if logic=="and" else "Match: ")+", ".join(reasons_pass)
        if not matched:
            reason = ("Not all met — " if logic=="and" else "None matched — ")+", ".join(reasons_fail[:3])
        return matched, reason
    except Exception as ex:
        return False, str(ex)

def run_monitor_check():
    """Run all conditions, send alerts if matched."""
    cfg   = load_monitor_config()
    items = cfg.get("items", [])
    results = []
    alerts  = []
    for item in items:
        matched, reason = check_conditions(item)
        results.append({"ticker":item.get("ticker",""), "matched":matched, "reason":reason})
        if matched:
            alerts.append({"ticker":item.get("ticker",""), "reason":reason, "note":item.get("note","")})

    if alerts:
        now_str  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        subject  = f"[Investment Mandate] {len(alerts)} Alert{'s' if len(alerts)>1 else ''} — {now_str}"
        rows     = "".join(f"<tr><td style='padding:6px 12px;font-weight:600'>{a['ticker']}</td><td style='padding:6px 12px;color:#16A34A'>{a['reason']}</td><td style='padding:6px 12px;color:#64748B'>{a['note']}</td></tr>" for a in alerts)
        body     = f"""<html><body style='font-family:Inter,sans-serif;color:#0F172A'>
<h2 style='color:#2563EB'>Investment Mandate Alert</h2>
<p>The following conditions were met at {now_str}:</p>
<table border='0' cellspacing='0' style='border-collapse:collapse;width:100%;background:#F8FAFC;border-radius:8px'>
<thead><tr style='background:#E2E8F0'><th style='padding:8px 12px;text-align:left'>Ticker</th><th style='padding:8px 12px;text-align:left'>Condition</th><th style='padding:8px 12px;text-align:left'>Note</th></tr></thead>
<tbody>{rows}</tbody></table>
<p style='color:#64748B;font-size:12px;margin-top:16px'>Checked daily at 7:00 AM AEST · Investment Mandate App</p>
</body></html>"""
        # Email
        if cfg.get("email") and cfg.get("smtpHost"):
            send_email(cfg, subject, body)
        # Google Calendar
        if cfg.get("calKey") and cfg.get("calId"):
            import datetime as _dt
            aest_7am = (_dt.datetime.utcnow().replace(hour=21,minute=0,second=0,microsecond=0)
                        + _dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
            desc = "\n".join(f"{a['ticker']}: {a['reason']}" for a in alerts)
            add_google_calendar_event(cfg, subject, desc, aest_7am)

    # Log result
    log_entry = {"ts":datetime.utcnow().isoformat()[:19], "results":results, "alertCount":len(alerts)}
    log_file = "monitor_log.json"
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file) as f: logs=json.load(f)
        except: pass
    logs.insert(0, log_entry)
    logs = logs[:200]  # keep last 200 runs
    with open(log_file,"w") as f: json.dump(logs,f)
    return results

# ── Scheduler: fire at 7AM AEST (= 21:00 UTC prev day) daily ──
def _scheduler_loop():
    import datetime as _dt
    while True:
        try:
            now_utc = _dt.datetime.utcnow()
            # 7AM AEST = UTC+10 = 21:00 UTC previous day
            target  = now_utc.replace(hour=21, minute=0, second=0, microsecond=0)
            if now_utc >= target:
                target += _dt.timedelta(days=1)
            wait_sec = (target - now_utc).total_seconds()
            _time.sleep(wait_sec)
            cfg = load_monitor_config()
            if cfg.get("items"):
                run_monitor_check()
        except Exception:
            _time.sleep(3600)  # on error, retry in 1h

_sched_thread = threading.Thread(target=_scheduler_loop, daemon=True)
_sched_thread.start()

# ── Monitor routes ──────────────────────────────────────────────────
@app.route("/monitor/config", methods=["GET"])
def get_monitor_config():
    cfg = load_monitor_config()
    # Redact password in response
    safe = {**cfg, "smtpPass":"" if cfg.get("smtpPass") else ""}
    return jsonify(safe)

@app.route("/monitor/config", methods=["POST"])
def post_monitor_config():
    try:
        d = request.json
        existing = load_monitor_config()
        # Preserve password if not re-submitted
        if not d.get("smtpPass") and existing.get("smtpPass"):
            d["smtpPass"] = existing["smtpPass"]
        save_monitor_config_file(d)
        return jsonify({"ok": True})
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

@app.route("/monitor/run", methods=["POST"])
def run_monitor_now():
    try:
        results = run_monitor_check()
        return jsonify({"results": results})
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

@app.route("/monitor/log", methods=["GET"])
def get_monitor_log():
    log_file = "monitor_log.json"
    if os.path.exists(log_file):
        with open(log_file) as f:
            return jsonify(json.load(f))
    return jsonify([])



# ══════════════════════════════════════════════════════════════════════
# CALCBENCH — Financial Statements (official calcbench-api-client)
# ══════════════════════════════════════════════════════════════════════
import requests as _req

CB_BASE = "https://www.calcbench.com"

CB_INCOME = [
    ("Revenue","Revenue"),("Cost of Revenue","CostOfRevenue"),
    ("Gross Profit","GrossProfit"),("R&D Expense","ResearchAndDevelopmentExpense"),
    ("SG&A","SellingGeneralAdministrative"),("Operating Income","OperatingIncomeLoss"),
    ("Interest Expense","InterestExpense"),
    ("Pretax Income","IncomeLossFromContinuingOperationsBeforeIncomeTaxes"),
    ("Income Tax","IncomeTaxExpenseBenefit"),("Net Income","NetIncomeLoss"),
    ("EPS Basic","EarningsPerShareBasic"),("EPS Diluted","EarningsPerShareDiluted"),
]
CB_BALANCE = [
    ("Cash & Equivalents","CashAndCashEquivalentsAtCarryingValue"),
    ("Accounts Receivable","AccountsReceivableNetCurrent"),
    ("Inventory","InventoryNet"),("Total Current Assets","AssetsCurrent"),
    ("PP&E Net","PropertyPlantAndEquipmentNet"),("Goodwill","Goodwill"),
    ("Total Assets","Assets"),("Accounts Payable","AccountsPayableCurrent"),
    ("Total Current Liabilities","LiabilitiesCurrent"),
    ("Long-term Debt","LongTermDebt"),("Total Liabilities","Liabilities"),
    ("Total Equity","StockholdersEquity"),
]
CB_CASHFLOW = [
    ("Operating Cash Flow","NetCashProvidedByUsedInOperatingActivities"),
    ("Depreciation & Amortisation","DepreciationDepletionAndAmortization"),
    ("CapEx","PaymentsToAcquirePropertyPlantAndEquipment"),
    ("Investing Activities","NetCashProvidedByUsedInInvestingActivities"),
    ("Financing Activities","NetCashProvidedByUsedInFinancingActivities"),
    ("Dividends Paid","PaymentsOfDividends"),
    ("Share Buybacks","PaymentsForRepurchaseOfCommonStock"),
    ("Net Change in Cash","CashAndCashEquivalentsPeriodIncreaseDecrease"),
]
CB_COMMENTARY = [
    ("Management Discussion & Analysis","ManagementsDiscussionAndAnalysisOfFinancialConditionAndResultsOfOperations"),
    ("Business Overview","Business"),
    ("Risk Factors","RiskFactors"),
    ("Liquidity & Capital Resources","LiquidityAndCapitalResources"),
    ("Critical Accounting Policies","CriticalAccountingPoliciesAndEstimates"),
]

TOTALS = {"GrossProfit","OperatingIncomeLoss","NetIncomeLoss","AssetsCurrent",
          "Assets","LiabilitiesCurrent","Liabilities","StockholdersEquity",
          "NetCashProvidedByUsedInOperatingActivities",
          "NetCashProvidedByUsedInInvestingActivities",
          "NetCashProvidedByUsedInFinancingActivities"}

def cb_session(email, password):
    s = _req.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 investment-app/1.0"
    try:
        r = s.post(f"{CB_BASE}/account/LogOnAjax",
                   data={"email":email,"strng":password,"rememberMe":"true"},
                   timeout=20)
    except _req.exceptions.ConnectionError as e:
        raise ValueError(f"Cannot reach Calcbench: {e}")
    except _req.exceptions.Timeout:
        raise ValueError("Calcbench timed out.")
    body = r.text.strip().strip('"').lower()
    if body != "true":
        raise ValueError(f"Login failed (HTTP {r.status_code}): {r.text[:100]}")
    return s

def cb_json(r):
    if not r.text.strip(): return None
    try: return r.json()
    except Exception: return None

def cb_standardized(session, ticker, metrics, fy, fp):
    """Batch fetch standardized values. Returns {metric_lower: value}."""
    try:
        r = session.post(f"{CB_BASE}/api/NormalizedValues",
            json={"start_year":fy,"start_period":fp,"end_year":fy,"end_period":fp,
                  "company_identifiers":[ticker],"metrics":[m[1] for m in metrics]},
            timeout=25)
        data = cb_json(r)
        if not data: return {}
        return {d.get("metric","").lower(): d.get("value") for d in data}
    except Exception: return {}

def cb_rows(val_map, defs):
    rows = []
    for label, metric in defs:
        v = val_map.get(metric.lower())
        rows.append({"label":label,"value":v,
                     "isTotal": metric in TOTALS,
                     "indent":0,"section":"","unit":""})
    return rows

def cb_commentary(session, ticker, calcbench_id):
    import re as _re
    sections = []
    for title, tag in CB_COMMENTARY:
        try:
            r = session.get(
                f"{CB_BASE}/api/disclosures?ticker={ticker}"
                f"&disclosure_type=AS_REPORTED"
                f"&accession_number={calcbench_id}"
                f"&disclosure_field={tag}",
                timeout=20)
            data = cb_json(r)
            if not data or not isinstance(data,list): continue
            text = data[0].get("disclosure_text","") if data else ""
            text = _re.sub(r"<[^>]+>"," ",text)
            text = " ".join(text.split()).strip()
            if len(text)>150:
                sections.append({"title":title,"text":text[:12000]})
        except Exception: continue
    return sections

def cb_infer_periods(session, ticker):
    """
    Infer available fiscal periods directly from standardized data —
    no filings API needed. Returns list of (year, period, label, form_type).
    Annual (10-K): period=0. Quarterly (10-Q): period=1..4.
    """
    periods = []
    try:
        # Fetch last 3 years of annual + quarterly Revenue to discover periods
        import datetime as _dt
        cur_year = _dt.datetime.utcnow().year
        r = session.post(f"{CB_BASE}/api/NormalizedValues",
            json={"start_year": cur_year-3, "start_period": 1,
                  "end_year":   cur_year,   "end_period":   4,
                  "company_identifiers": [ticker],
                  "metrics": ["Revenue"]},
            timeout=20)
        data = cb_json(r)
        if not data: return []
        seen = set()
        for item in data:
            fy  = item.get("fiscal_year")  or item.get("calendar_year")
            fp  = item.get("fiscal_period") if item.get("fiscal_period") is not None else item.get("calendar_period")
            if fy is None or fp is None: continue
            key = (int(fy), int(fp))
            if key in seen: continue
            seen.add(key)
            form  = "10-K" if int(fp)==0 else "10-Q"
            q_lbl = f"Q{fp}" if int(fp)>0 else "Annual"
            label = f"{fy} {q_lbl}"
            periods.append({"fy":int(fy),"fp":int(fp),"label":label,
                            "form":form,"period":f"{fy}-{q_lbl}","filedOn":""})
        # Sort most recent first: annual then quarterly within year
        periods.sort(key=lambda x: (x["fy"], x["fp"]), reverse=True)
        return periods[:6]
    except Exception:
        return []

@app.route("/financials/test", methods=["POST"])
def financials_test():
    d = request.json
    try:
        session = cb_session(d.get("email",""), d.get("password",""))
        r = session.get(f"{CB_BASE}/api/companies?tickers=MSFT", timeout=10)
        data = cb_json(r)
        if r.status_code == 200:
            return jsonify({"ok": True})
        return jsonify({"error": f"API returned {r.status_code}: {r.text[:100]}"}), 400
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 401
    except Exception as ex:
        return jsonify({"error": f"Error: {ex}"}), 500

@app.route("/financials/fetch", methods=["POST"])
def financials_fetch():
    d = request.json
    ticker   = (d.get("ticker") or "").upper().strip()
    email    = d.get("email","")
    password = d.get("password","")
    if not all([ticker, email, password]):
        return jsonify({"error":"Ticker, email and password required."}), 400
    try:
        session = cb_session(email, password)

        # Company name
        company_name = ticker
        try:
            r_co = session.get(f"{CB_BASE}/api/companies?tickers={ticker}", timeout=10)
            co   = cb_json(r_co)
            if co and isinstance(co,list): company_name = co[0].get("name", ticker)
        except Exception: pass

        # Infer available periods directly from standardized data
        periods = cb_infer_periods(session, ticker)
        if not periods:
            return jsonify({"error":
                f"No data found for {ticker}. "
                f"Verify at https://www.calcbench.com/financial_statements/{ticker}"}), 404

        result = []
        for p in periods:
            fy, fp = p["fy"], p["fp"]
            inc_map = cb_standardized(session, ticker, CB_INCOME,   fy, fp)
            bal_map = cb_standardized(session, ticker, CB_BALANCE,  fy, fp)
            cf_map  = cb_standardized(session, ticker, CB_CASHFLOW, fy, fp)
            # Only include period if we got at least some data
            if not any([inc_map, bal_map, cf_map]): continue
            result.append({
                "id":         f"{p['form']}-{fy}-{fp}",
                "type":       p["form"],
                "period":     p["period"],
                "filedOn":    p["filedOn"],
                "income":     cb_rows(inc_map, CB_INCOME),
                "balance":    cb_rows(bal_map, CB_BALANCE),
                "cashflow":   cb_rows(cf_map,  CB_CASHFLOW),
                "commentary": [],   # commentary requires filing ID — not available without filings API
                "sourceUrl":  f"https://www.calcbench.com/financial_statements/{ticker}",
            })

        if not result:
            return jsonify({"error": f"Data found but all empty for {ticker}."}), 404

        return jsonify({"ticker":ticker,"companyName":company_name,"filings":result})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 401
    except Exception as ex:
        return jsonify({"error": f"Calcbench error: {str(ex)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
