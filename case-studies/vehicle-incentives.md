---
title: "Case study: Vehicle Introducer Incentives"
layout: default
---

# Lendy Scenario Analysis

*Case study delivering decision support artefacts for a fictitious finance company*

---

## Contents


## Background Story: Why Lendy Needed Scenario Analysis

Lendy is a fictitious, non-bank financial services lender with approximately **$1.5B in Assets Under Management (AUM)**. The business has grown steadily through **introducer-led distribution** (brokers, dealers, referrers), which accounts for roughly **90% of all originations**.

The CEO has set an ambitious but deliberately non-linear growth ambition:

* **$3.5B AUM within 2 years**
* **$5.0B AUM by 2030**

Importantly, the CEO has ruled out a move to direct-to-consumer channels. Growth must come from:

* Better utilisation of introducer networks
* Portfolio mix decisions
* Pricing, risk appetite, and funding strategy

By late 2024, the executive team recognised a recurring problem:

> *Every strategy discussion collapsed into a single forecast, followed by debates about whether the assumptions were “too optimistic” or “too conservative”.*

Lendy introduced **enterprise scenario analysis** to shift the conversation away from prediction and toward **decision robustness**. The goal was not to find the “right answer”, but to understand:

* What needs to be true for the growth plan to work
* Where it might fail
* What guardrails leadership should put in place

This case study documents the **first full scenario analysis cycle**, which became the foundation for a suite of tools that drove informed decision making over individual opinion and point forecasts. A sample of **Decision Cockpit dashboards**, the primary tools are provided.

---

## Step 1: Decision Framing

**Decision statement**

> *How should Lendy grow from $1.5B to $5.0B AUM over the next 3–5 years while maintaining acceptable portfolio risk and funding resilience?*

**Why this decision matters now**

* Growth targets have been publicly signalled to investors
* Traditional funding is at risk when markets turn. Diversification reduces risk and resiliance.
* Introducer concentration is increasing

**Time horizon**

* Primary: 3–5 years (strategic)

**What success looks like**

* AUM exceeds $3.5B within 24 months
* Losses remain within stated risk appetite
* No single introducer group exceeds concentration thresholds

**What failure looks like**

* Growth achieved only by materially degrading credit quality i.e. quality remains critical despite a volume-based objective.
* Funding costs erode margins faster than volumes grow i.e. the ROE / WACC trade off degrades
* Foreseeable, but unmonitored, trends materially impact AUM i.e. inventives and other Lendy interventions negatively impact introducer behaviour  

**Stakeholders and incentives**

* Executive - Credit/Risk, Sales, Treasury, Strategy 
* Sales – advocate for introducers and market interventions that work. Responsible for volume.
* Strategy - distribute decisions, with trade offs, and ensure identified risks are incorporated into monitoring
* Credit & risk – focused on downside protection and early warning signals

The solution needed to align these perspectives in a single artefact.

---

## Step 2: System Boundary

**In scope**

* Portfolio-level AUM flows (origination, runoff)
* Asset classes:

  * Vehicle loans (70%)
  * Personal loans (10%)
  * Residential mortgages (10%)
  * Small business loans (10%)
* Portfolio-level loss rates
* Net interest margin and funding cost
* Introducer behaviour

**Out of scope (by design)**

* Borrower-level credit models
* Full regulatory capital modelling
* Direct-to-consumer channel economics
* Originate-to-distribute and portfolio modelling

**Controllable levers**

* Origination targets by asset class
* Introducer incentives and prioritisation
* Portfolio mix constraints

**External forces**

* Funding market conditions
* Macro-driven credit stress
* Introducer behaviour

---

## Step 3: Scenario Definition

### Scenario A: Base Case – “Disciplined Expansion”

Growth continues broadly in line with recent history. Introducers remain engaged, funding remains available at modestly higher spreads, and credit losses drift slightly upward but remain manageable.

This scenario reflects leadership’s *planning baseline*, not a forecast.

---

### Scenario B: Downside – “Growth with Friction”

Funding becomes more selective, requiring higher spreads. Introducers push lower-quality volume to maintain income, increasing portfolio loss rates. Growth is still possible, but trade-offs become visible.

This scenario tests whether Lendy’s growth plan is **fragile**.

---

### Scenario C: Upside – “Capacity Unlock”

Introducer productivity improves through better incentives and operational efficiency. Funding markets remain supportive, allowing growth without materially increasing loss rates.

This scenario tests **latent upside** rather than aggressive optimism.

---

## Step 4: Quantitative Assumptions

| Category                | Base Case | Downside | Upside   |
| ----------------------- | --------- | -------- | -------- |
| Avg Vehicle loan        | $10,000   | $8,000   | $12,000  |
| Vehicle loan growth     | 18% p.a.  | 10% p.a. | 22% p.a. |
| Other portfolios growth | 12% p.a.  | 6% p.a.  | 15% p.a. |
| Portfolio loss rate     | 1.2%      | 2.0%     | 1.0%     |
| Net interest margin     | Stable    | -80 bps  | +40 bps  |
| Cost of funds           | +50 bps   | +150 bps | Flat     |
| Top-5 introducer share  | 45%       | 55%      | 42%      |

**Assumptions most open to challenge**

* Loss rate sensitivity to growth pressure
* Funding cost volatility

---

## Step 5: Outcome Modelling

**Key outcomes modelled**

* AUM trajectory by asset class
* Portfolio mix evolution
* Gross interest income
* Credit losses

**Headline results**

* Base case reaches ~$3.7B AUM in year 2
* Downside reaches ~$3.1B with materially higher loss drag
* Upside exceeds $4.0B without breaching risk appetite

**Notable dynamics**

* Loss rate increases compound faster than margin compression
* Vehicle loan dominance amplifies both upside and downside

---

## Step 6: Trade-offs & Robustness

**Strategy options evaluated**

1. Vehicle-led growth
2. Balanced portfolio growth
3. Risk-constrained growth with tighter introducer limits

**Key insight**

* Vehicle-led growth maximises AUM but is fragile in the downside scenario
* Balanced growth sacrifices short-term AUM but is more robust

**Break-points**

* Loss rates above ~1.8% erase incremental growth benefits
* Funding spread increases above ~150 bps materially impair returns

---

## Step 7: Decision & Guardrails

**Strategic direction**

* Pursue growth anchored in vehicle lending
* Gradually rebalance portfolio mix toward non-vehicle assets

**Guardrails**

* Portfolio loss rate ≤ 1.6%
* Top-5 introducer share ≤ 50%
* Funding spread increase ≤ 125 bps

**Trigger points**

* Two consecutive quarters breaching loss thresholds
* Rapid introducer concentration growth

---

## Step 8: Communication & Refresh

**Executive message**

> Lendy can achieve its growth ambition, but only if growth is actively managed rather than passively forecast.

**Monitoring metrics**

* Monthly AUM growth vs scenario ranges
* Loss rates vs guardrails
* Introducer concentration trends

**Refresh cadence**

* Annual strategic refresh
* Event-driven refresh following funding or credit shocks

---

## How This Became a Decision Cockpit

This scenario analysis was operationalised into a **Decision Cockpit dashboard**:

* Scenario selector (Base / Downside / Upside)
* Growth and loss sliders
* Live AUM and margin projections
* Guardrail breach indicators

The dashboard does not replace judgment. It makes trade-offs visible.

This case study demonstrates how **scenario analysis, simple modelling, and data science practices** combine to support better strategic decisions—without pretending to predict the future.

### What the cockpit shows

**Levers**

* Expected origination uplift
* Incentive cost per loan
* Assumed deterioration in credit losses
* Average loan size

**Outputs**

* Expected profit
* Downside profit (P10)
* Probability of loss
* Trade-off curves showing diminishing returns to risk

**Guardrails**

* Maximum acceptable probability of loss
* Minimum acceptable downside profit
* Caps on incentive intensity and loss-rate drift