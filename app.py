import streamlit as st
import pandas as pd, numpy as np, requests, datetime as dt, re

st.set_page_config(page_title="Capital AI — Polygon MVP", layout="centered")
st.title("Capital AI — Polygon MVP")

API_KEY = st.secrets.get("POLYGON_API_KEY", "")

def norm_symbol(raw: str) -> str:
    s = (raw or "").strip().upper()
    if not s: return s
    if s.startswith("X:"): return s
    s = s.replace("-", "").replace("/", "")
    if s.endswith("USDT"): return f"X:{s[:-4]}USD"
    if s.endswith("USD"):  return f"X:{s[:-3]}USD"
    if re.fullmatch(r"[A-Z][A-Z0-9\.]{0,9}", s): return s
    return s

def polygon_aggs(symbol: str, mult: int, span: str, start: dt.date, end: dt.date, api_key: str):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    params = {"adjusted":"true","sort":"asc","limit":50000,"apiKey":api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    status = js.get("status"); res = js.get("results", [])
    if status not in ("OK","DELAYED") or not res:
        raise RuntimeError(f"Polygon пусто/ошибка: {status}")
    df = pd.DataFrame(res)
    df["Date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.set_index("Date").sort_index()
    df = df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close"})
    return df[["Open","High","Low","Close"]]

def fib_pivots(H,L,C):
    P=(H+L+C)/3.0; R=H-L
    return {"P":P,"R1":P+0.382*R,"R2":P+0.618*R,"R3":P+1.0*R,
            "S1":P-0.382*R,"S2":P-0.618*R,"S3":P-1.0*R}

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

def macd_hist(close: pd.Series):
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    m = e12 - e26
    s = m.ewm(span=9, adjust=False).mean()
    return m - s

def sign_streak(series: pd.Series):
    s = np.sign(series.dropna())
    if len(s)==0: return 0
    last = s.iloc[-1]; k=0
    for v in reversed(s.values):
        if v==last and v!=0: k+=1
        else: break
    return int(k if last>0 else -k)

def build_decision(df: pd.DataFrame, horizon: str):
    price = float(df["Close"].iloc[-1])
    ha = heikin_ashi(df)
    ha_streak = sign_streak(ha["HA_Close"].diff())
    mom_streak = sign_streak(macd_hist(df["Close"]))
    atr_last = float(atr(df).iloc[-1])

    H,L,C = prev_period_hlc(df, horizon)
    piv = fib_pivots(H,L,C); P,R1,R2,R3,S1,S2,S3 = [piv[k] for k in ("P","R1","R2","R3","S1","S2","S3")]

    need = {"Трейд (1–5 дней)":0.012, "Среднесрок (1–4 недели)":0.08, "Долгосрок (1–6 месяцев)":0.16}[horizon]
    width = max(abs(R1-P)+abs(P-S1), abs(R3-S3)*0.5, price*need)
    ext_up1 = P + 1.272*width; ext_up2 = P + 1.618*width
    ext_dn1 = P - 1.272*width; ext_dn2 = P - 1.618*width

    def pick(direction):
        if direction=="LONG":
            cands = sorted({x for x in [R1,R2,R3,ext_up1,ext_up2] if x>price})
        else:
            cands = sorted({x for x in [S1,S2,S3,ext_dn1,ext_dn2] if x<price}, reverse=True)
        cands = [x for x in cands if abs(x-price)/price >= need*0.9] or cands
        if not cands: return None, None
        return round(cands[0],2), round((cands[1] if len(cands)>1 else cands[0]),2)

    overheat = (horizon!="Трейд (1–5 дней)" and ha_streak>=4 and mom_streak>=4 and price>=R1)
    oversold = (horizon!="Трейд (1–5 дней)" and ha_streak<=-4 and mom_streak<=-4 and price<=S1)
    atr_k = {"Трейд (1–5 дней)":1.0, "Среднесрок (1–4 недели)":1.6, "Долгосрок (1–6 месяцев)":2.2}[horizon]

    if overheat or price>=R2:
        base="SHORT"; entry=max(R2, price); t1,t2=pick("SHORT"); sl=entry+atr_k*atr_last
        alt="LONG"; alt_e=min(P, price); at1,at2=pick("LONG"); alt_sl=alt_e-atr_k*atr_last
    elif oversold or price<=S2:
        base="LONG"; entry=min(P, price); t1,t2=pick("LONG"); sl=entry-atr_k*atr_last
        alt="SHORT"; alt_e=max(R2, price); at1,at2=pick("SHORT"); alt_sl=alt_e+atr_k*atr_last
    else:
        base="WAIT"; entry=t1=t2=sl=None
        alt="LONG"; alt_e=min(P, price); at1,at2=pick("LONG"); alt_sl=alt_e-atr_k*atr_last

    def r(x):
        try: return None if x is None else round(float(x),2)
        except: return None

    return {
        "price": round(price,2),
        "horizon": horizon,
        "base": {"action": base, "entry": r(entry), "tp1": r(t1), "tp2": r(t2), "sl": r(sl)},
        "alt":  {"action": alt,  "entry": r(alt_e), "tp1": r(at1), "tp2": r(at2), "sl": r(alt_sl)}
    }

def humanize(ticker: str, d: dict) -> str:
    price = d["price"]; hz = d["horizon"]
    b, a = d["base"], d["alt"]
    def v(x): return "-" if x in (None, "None") else f"{x:.2f}"
    def act(x): return {"WAIT":"WAIT","LONG":"BUY","SHORT":"SHORT"}.get(x,x)
    intro = ["Импульс остывает — лучше действовать выборочно.",
             "Цена высоковата для уверенного лонга, берём там, где перевес наш.",
             "Рынок замедлился — спешка сейчас редко платит."][hash(ticker+hz)%3]
    return (
f"📊 Результат анализа: {ticker} — {hz}\n"
f"{intro}\n\n"
f"✅ Базовый план: {act(b['action'])}\n"
+ ( "→ На текущих вход не даёт преимущества. Ждём подтверждения.\n" if b['action']=="WAIT"
    else f"→ Вход: {v(b['entry'])}  |  Цели: {v(b['tp1'])} / {v(b['tp2'])}  |  Защита: {('выше ' if b['action']=='SHORT' else '')}{v(b['sl'])}\n" )
+ "\n"
f"🔄 Альтернатива: {act(a['action']).lower()}\n"
f"→ Вход: {v(a['entry'])}  |  Цели: {v(a['tp1'])} / {v(a['tp2'])}  |  Защита: {v(a['sl'])}\n\n"
"⚠️ Если сценарий ломается — быстро выходим и ждём новую формацию."
    )

symbol_in = st.text_input("Тикер (QQQ, AAPL, X:BTCUSD, BTCUSDT, BTC/USD):", "QQQ")
horiz = st.selectbox("Горизонт", ["Трейд (1–5 дней)","Среднесрок (1–4 недели)","Долгосрок (1–6 месяцев)"])
go = st.button("Проанализировать")

if go:
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
        st.write({"rows": len(df), "period": f"{df.index.min().date()} → {df.index.max().date()}"})
