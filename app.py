import streamlit as st, pandas as pd, numpy as np, requests, datetime as dt, re

st.set_page_config(page_title="Capital AI — Polygon (моя логика)", layout="centered")
st.title("Capital AI — Polygon")

API_KEY = st.secrets.get("POLYGON_API_KEY","")

# ---------- utils ----------
def macd_hist(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

def atr(df: pd.DataFrame, period: int = 14):
    h,l,c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(period).mean()

def heikin_ashi(df: pd.DataFrame):
    ha = df.copy()
    ha["HA_Close"] = (df["Open"]+df["High"]+df["Low"]+df["Close"])/4
    ha["HA_Open"] = ha["HA_Close"].copy()
    for i in range(1,len(ha)):
        ha.iat[i, ha.columns.get_loc("HA_Open")] = (ha.iat[i-1, ha.columns.get_loc("HA_Open")] + ha.iat[i-1, ha.columns.get_loc("HA_Close")])/2
    return ha

def streak_len_bool(series: pd.Series, is_green=True) -> int:
    s = series.dropna()
    if s.empty: return 0
    sign = 1 if is_green else -1
    k = 0
    for v in reversed(s.values):
        if (v > 0 and sign==1) or (v < 0 and sign==-1): k += 1
        elif v == 0: continue
        else: break
    return k

def near(x, lvl, rel=0.008):
    if x is None or lvl is None: return False
    return abs(x - lvl) / max(1e-9, x) <= rel

def fib_pivots(H,L,C):
    P=(H+L+C)/3.0; R=H-L
    return {"P":P,"R1":P+0.382*R,"R2":P+0.618*R,"R3":P+1.0*R,
            "S1":P-0.382*R,"S2":P-0.618*R,"S3":P-1.0*R}

# ---------- prev HLC ----------
def prev_period_hlc(df: pd.DataFrame, horizon: str):
    if horizon == "Трейд (1–5 дней)":
        grp = df.resample("D").agg({"High":"max","Low":"min","Close":"last"}).dropna()
    elif horizon == "Среднесрок (1–4 недели)":
        grp = df.resample("M").agg({"High":"max","Low":"min","Close":"last"}).dropna()
        if len(grp)<2: grp = df.resample("W").agg({"High":"max","Low":"min","Close":"last"}).dropna()
    else:
        grp = df.resample("Y").agg({"High":"max","Low":"min","Close":"last"}).dropna()
        if len(grp)<2: grp = df.resample("M").agg({"High":"max","Low":"min","Close":"last"}).dropna()
    row = grp.iloc[-2] if len(grp)>=2 else df.iloc[-2]
    return float(row["High"]), float(row["Low"]), float(row["Close"])

# ---------- overheat (твоя логика) ----------
def detect_overheat_logic(df: pd.DataFrame, piv: dict, horizon: str):
    price = float(df["Close"].iloc[-1])
    ha = heikin_ashi(df)
    ha_green_streak = streak_len_bool(ha["HA_Close"].diff(), is_green=True)
    hist = macd_hist(df["Close"])
    macd_green_streak = streak_len_bool(hist, is_green=True)

    need_ha   = {"Трейд (1–5 дней)":4, "Среднесрок (1–4 недели)":5, "Долгосрок (1–6 месяцев)":6}[horizon]
    need_macd = {"Трейд (1–5 дней)":4, "Среднесрок (1–4 недели)":6, "Долгосрок (1–6 месяцев)":8}[horizon]

    R1,R2,R3 = piv["R1"],piv["R2"],piv["R3"]; S1,S2 = piv["S1"],piv["S2"]; P = piv["P"]
    rel_tol = {"Трейд (1–5 дней)":0.006, "Среднесрок (1–4 недели)":0.009, "Долгосрок (1–6 месяцев)":0.012}[horizon]
    at_res = any([near(price,R1,rel_tol), near(price,R2,rel_tol), near(price,R3,rel_tol)]) or price >= R2
    is_overheat = (ha_green_streak >= need_ha) and (macd_green_streak >= need_macd) and at_res

    pullback_low  = S2 if price >= R2 and macd_green_streak >= need_macd+2 else S1
    pullback_high = max(P, S1)
    pullback_zone = (round(min(pullback_low, pullback_high),2), round(max(pullback_low, pullback_high),2))

    return {"overheat": bool(is_overheat), "pullback_zone": pullback_zone}

# ---------- Polygon data ----------
def norm_symbol(raw: str) -> str:
    s = (raw or "").strip().upper()
    if s.startswith("X:") or s.startswith("I:"): return s
    s = s.replace("-", "").replace("/", "")
    if s.endswith("USDT"): return f"X:{s[:-4]}USD"
    if s.endswith("USD"):  return f"X:{s[:-3]}USD"
    return s

def polygon_aggs(symbol: str, mult: int, span: str, start: dt.date, end: dt.date, api_key: str):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    params = {"adjusted":"true","sort":"asc","limit":50000,"apiKey":api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("status") not in ("OK","DELAYED") or not js.get("results"):
        raise RuntimeError(f"Polygon пусто/ошибка: {js.get('status')}")
    df = pd.DataFrame(js["results"])
    df["Date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.set_index("Date").sort_index()
    df = df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close"})
    return df[["Open","High","Low","Close"]]

# ---------- Decision ----------
def build_decision(df: pd.DataFrame, horizon: str):
    price = float(df["Close"].iloc[-1])
    H,L,C = prev_period_hlc(df, horizon)
    piv = fib_pivots(H,L,C)
    over = detect_overheat_logic(df, piv, horizon)

    atr_last = float(atr(df).iloc[-1])
    atr_k = {"Трейд (1–5 дней)":1.0, "Среднесрок (1–4 недели)":1.6, "Долгосрок (1–6 месяцев)":2.2}[horizon]
    def r(x): 
        try: return None if x is None else round(float(x),2)
        except: return None

    if over["overheat"]:
        base="WAIT"; entry=tp1=tp2=sl=None
        alt="SHORT"
        alt_entry = max(piv["R2"], price)
        lo,hi = over["pullback_zone"]
        alt_t1, alt_t2 = hi, lo
        alt_sl = alt_entry + atr_k*atr_last
    else:
        if price >= piv["R2"]:
            base="SHORT"; entry=max(piv["R2"], price)
            lo,hi = detect_overheat_logic(df, piv, horizon)["pullback_zone"]
            tp1, tp2 = hi, lo
            sl = entry + atr_k*atr_last
            alt="LONG"; alt_entry=min(piv["P"], price)
            alt_t1, alt_t2 = piv["R1"], piv["R2"]
            alt_sl = alt_entry - atr_k*atr_last
        elif price <= piv["S2"]:
            base="LONG"; entry=min(piv["P"], price)
            tp1, tp2 = piv["R1"], piv["R2"]
            sl = entry - atr_k*atr_last
            alt="SHORT"; alt_entry=max(piv["R1"], price)
            alt_t1, alt_t2 = piv["P"], detect_overheat_logic(df, piv, horizon)["pullback_zone"][0]
            alt_sl = alt_entry + atr_k*atr_last
        else:
            base="WAIT"; entry=tp1=tp2=sl=None
            alt="LONG"; alt_entry=min(piv["P"], price)
            alt_t1, alt_t2 = piv["R1"], piv["R2"]
            alt_sl = alt_entry - atr_k*atr_last

    return {
        "price": round(price,2),
        "horizon": horizon,
        "base": {"action": base, "entry": r(entry), "tp1": r(tp1), "tp2": r(tp2), "sl": r(sl)},
        "alt":  {"action": alt,  "entry": r(alt_entry), "tp1": r(alt_t1), "tp2": r(alt_t2), "sl": r(alt_sl)},
        "ctx": {"overheat": over}
    }

# ---------- text ----------
def humanize(ticker: str, d: dict) -> str:
    price = d["price"]; hz = d["horizon"]
    b, a = d["base"], d["alt"]
    def v(x): return "-" if x in (None, "None") else f"{x:.2f}"
    def act(x): return {"WAIT":"WAIT","LONG":"BUY","SHORT":"SHORT"}.get(x,x)
    intro = ["Под потолком важнее точность, чем скорость.",
             "Импульс остывает — беру от сильных зон, а не с вершины.",
             "Похоже на распределение — не гонюсь за ценой."][hash(ticker+hz)%3]
    lines = []
    lines.append(f"📊 Результат анализа: {ticker} — {hz}")
    lines.append(intro); lines.append("")
    lines.append(f"✅ Базовый план: {act(b['action'])}")
    if b["action"]=="WAIT":
        lines.append("→ На текущих вход не даёт преимущества. Жду более выгодной цены или подтверждения.")
    elif b["action"]=="LONG":
        lines.append(f"→ Вход: {v(b['entry'])}  |  Цели: {v(b['tp1'])} / {v(b['tp2'])}  |  Защита: {v(b['sl'])}")
    else:
        lines.append(f"→ Вход: {v(b['entry'])}  |  Цели: {v(b['tp1'])} / {v(b['tp2'])}  |  Защита: выше {v(b['sl'])}")
    lines.append("")
    lines.append(f"🔄 Альтернатива: {act(a['action']).lower()}")
    lines.append(f"→ Вход: {v(a['entry'])}  |  Цели: {v(a['tp1'])} / {v(a['tp2'])}  |  Защита: {v(a['sl'])}")
    lines.append("")
    over = d.get("ctx",{}).get("overheat")
    if over and over.get("overheat"):
        lo, hi = over["pullback_zone"]
        lines.append(f"⚠️ Перегрев у верхней границы. Жду откат в диапазон {lo}–{hi} и уже там отбираю лонг.")
        lines.append("")
    lines.append("⚠️ Если сценарий ломается — быстро выходим и ждём новую формацию.")
    return "\n".join(lines)

# ---------- UI ----------
symbol_in = st.text_input("Тикер (QQQ, AAPL, X:BTCUSD…)", "QQQ")
horiz = st.selectbox("Горизонт", ["Трейд (1–5 дней)","Среднесрок (1–4 недели)","Долгосрок (1–6 месяцев)"])
if st.button("Проанализировать"):
    if not API_KEY:
        st.error("Вставьте POLYGON_API_KEY в .streamlit/secrets.toml"); st.stop()
    sym = norm_symbol(symbol_in)
    today = dt.date.today()
    if horiz=="Трейд (1–5 дней)":
        start = today - dt.timedelta(days=21); mult, span = 60, "minute"
    elif horiz=="Среднесрок (1–4 недели)":
        start = today - dt.timedelta(days=210); mult, span = 1, "day"
    else:
        start = today - dt.timedelta(days=600); mult, span = 1, "day"
    try:
        df = polygon_aggs(sym, mult, span, start, today, API_KEY)
    except Exception as e:
        st.error(f"Ошибка Polygon: {e}"); st.stop()
    rep = build_decision(df, horiz)
    st.markdown(f"### {sym} — текущая цена: ${rep['price']:.2f}")
    st.write(humanize(sym, rep))
    with st.expander("Диагностика"):
        st.write({"rows": len(df), "period": f"{df.index.min().date()} → {df.index.max().date()}", "horizon": horiz, "context": rep["ctx"]})

  
