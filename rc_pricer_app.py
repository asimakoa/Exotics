# rc_pricer_app.py
# Streamlit app to price and visualize a Reverse Convertible (worst-of put) on a 3-stock basket

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# --- Reverse Convertible Pricer Function ---
def price_reverse_convertible(
    spots, vols, divs, corr, rate, maturity, strike_pct, n_paths=100_000, n_steps=12
):
    """Monte Carlo pricing for a 3-name worst-of put (Reverse Convertible structure)."""
    rng = np.random.default_rng(42)
    dt = maturity / n_steps
    L = np.linalg.cholesky(corr + 1e-12 * np.eye(3))
    mu = (rate - divs - 0.5 * vols**2) * dt
    sig = vols * np.sqrt(dt)

    half = n_paths // 2
    S = np.tile(spots, (half, 1))
    S_anti = S.copy()

    for _ in range(n_steps):
        Z = rng.standard_normal(size=(half, 3))
        dW = Z @ L.T
        S *= np.exp(mu + sig * dW)
        S_anti *= np.exp(mu - sig * dW)

    S_T = np.vstack([S, S_anti])
    perf = S_T / spots
    worst_idx = np.argmin(perf, axis=1)
    S_T_worst = S_T[np.arange(S_T.shape[0]), worst_idx]
    S0_worst = spots[worst_idx]
    K = strike_pct * S0_worst
    payoff = np.maximum(K - S_T_worst, 0.0)
    disc = np.exp(-rate * maturity)
    price = disc * payoff.mean()
    std_error = disc * payoff.std(ddof=1) / np.sqrt(S_T.shape[0])
    put_premium_pct = price / np.mean(spots) * 100
    fair_coupon = 100 * (rate + price / maturity / np.mean(spots))
    return put_premium_pct, fair_coupon, std_error

# --- Streamlit UI ---
st.set_page_config(page_title="Reverse Convertible Pricer", layout="centered")
st.title("💰 Reverse Convertible Pricer (3-name Worst-of Put)")

st.markdown("""
This app prices a **Reverse Convertible** (worst-of put option structure)
on three equities: **Nestlé**, **Tesla**, and **Audi (proxy like VW)**.
""")

# Market Inputs
st.header("Market Inputs")
c1, c2, c3 = st.columns(3)
S1 = c1.number_input("Nestlé Spot", value=100.0)
S2 = c2.number_input("Tesla Spot", value=100.0)
S3 = c3.number_input("Audi (Proxy) Spot", value=100.0)

v1 = c1.number_input("Nestlé Vol", value=0.20)
v2 = c2.number_input("Tesla Vol", value=0.55)
v3 = c3.number_input("Audi Vol", value=0.30)

q1 = c1.number_input("Nestlé Div Yield", value=0.02)
q2 = c2.number_input("Tesla Div Yield", value=0.00)
q3 = c3.number_input("Audi Div Yield", value=0.03)

st.header("Correlation Matrix")
corr = np.eye(3)
labels = ["Nestlé", "Tesla", "Audi"]
for i in range(3):
    for j in range(i + 1, 3):
        corr[i, j] = st.number_input(f"Corr {labels[i]}-{labels[j]}", value=0.25)
        corr[j, i] = corr[i, j]

# Trade Inputs
st.header("Trade Inputs")
rate = st.number_input("Risk-Free Rate", value=0.02)
maturity = st.number_input("Maturity (years)", value=1.0)
strike_pct = st.number_input("Strike (% of spot)", value=0.80)
n_paths = st.number_input("Monte Carlo Paths", value=100_000)
n_steps = st.number_input("Steps per Year", value=12)
coupon_input = st.number_input("Coupon offered to investor (%)", value=10.0)

# Button to run
if st.button("💸 Price Reverse Convertible"):
    spots = np.array([S1, S2, S3])
    vols = np.array([v1, v2, v3])
    divs = np.array([q1, q2, q3])
    put_premium, fair_coupon, std_error = price_reverse_convertible(
        spots, vols, divs, corr, rate, maturity, strike_pct, n_paths=int(n_paths), n_steps=int(n_steps)
    )
    st.success("Pricing Completed ✅")
    st.metric("Short Worst-of Put Value (you receive)", f"{put_premium:.3f} % of spot")
    st.metric("Fair Coupon (approx. p.a.)", f"{fair_coupon:.3f} %")
    st.caption(f"Monte Carlo Std. Error: {std_error:.5f}")

    # --- Visualization ---
    st.header("Payoff Visualization (Investor's P&L vs. Worst Underlying)")
    perf = np.linspace(0.4, 1.2, 200)
    payoff = np.where(perf < strike_pct, perf - 1 + coupon_input/100, coupon_input/100)
    plt.figure(figsize=(7,4))
    plt.plot(perf*100, payoff*100, label="Investor P&L (%)")
    plt.axvline(strike_pct*100, color="red", linestyle="--", label="Strike")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Reverse Convertible Payoff vs. Worst-of Performance")
    plt.xlabel("Worst-of Performance at Maturity (% of initial spot)")
    plt.ylabel("Investor Return (%)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
