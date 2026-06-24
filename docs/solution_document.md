# Kizuna RiskTriage — Detailed Solution Document

## Calibrated Uncertainty Quantification for Supply Chain Risk Triage

**Team:** Kizuna | **Problem Statement:** PS2 | **Round:** 2

---

## 1. Problem Understanding

Supply chain managers make high-stakes inventory decisions on demand forecasts that hide risk. Point forecasts express false precision. Most confidence intervals are uncalibrated. The most damaging failure: uncertainty estimates calibrated on average but collapsing during high-volatility periods — the system is most overconfident exactly when wrong decisions are most expensive.

Standard conformal prediction assumes exchangeability that time-series demand violates. We frame this as supply chain risk triage: sort products by forecast trustworthiness and route each to an action.

## 2. Solution Architecture

Four-layer system:
1. **Probabilistic Forecasting** — LightGBM quantile regression (9 quantiles)
2. **Calibration Engine** — CQR + Adaptive Conformal Inference
3. **Risk Triage** — Low/Medium/High tiers from relative interval width
4. **Decision Support** — Newsvendor simulation + underestimation audit + dashboard

## 3. Technical Approach

- **Dataset:** M5 Walmart (synthetic equivalent for demo; real M5 for full eval)
- **Features:** Lags (1-28d), rolling stats (7-56d windows), calendar, events, volatility labels
- **Methods compared:** Raw QR → Split Conformal → CQR → CQR+ACI
- **Calibration metrics:** PICP, Winkler Score, Pinball Loss, conditional coverage
- **Risk tiers:** Validated by showing High-tier items have higher realized stockouts
- **Simulation:** Newsvendor order-up-to policy, robustness sweep across cost ratios

## 4. Prototype Design

- Python 3.11, LightGBM, custom conformal implementations
- Streamlit dashboard showing per-product forecasts + risk tiers + actions
- Full pipeline runs in <2 minutes on laptop

## 5. Feasibility Analysis

- No deep learning or GPU required
- MAPIE-compatible architecture
- Modular code, each component independently testable
- Known risks mitigated: intermittent demand (segmentation), arbitrary thresholds (sensitivity analysis), cost assumptions (robustness sweep)

## 6. Expected Impact

- Hits target service level that baseline misses during volatility
- Lower total inventory cost (holding + shortage)
- Reduced waste and embedded carbon from overstock
- Managers get trustworthy, prioritized action list instead of spreadsheet guesswork

## 7. Future Scope

- Multi-store/category scaling on full M5
- External data integration (weather, news)
- Online learning with incremental model updates
- Production deployment via FastAPI + ERP integration
