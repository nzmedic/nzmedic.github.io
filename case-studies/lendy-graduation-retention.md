---
title: "Case study: Graduation Risk & Retention Uplift (Vehicles)"
layout: default
---

# Graduation Risk & Retention Uplift (Vehicles)

*Case study demonstrating how scenario analysis, data science, and causal reasoning combine to support better retention decisions under uncertainty.*

---

## Contents

0. [Background Story](#background-story)  
1. [Step 1: Decision Framing](#step-1-decision-framing)  
2. [Step 2: System Boundary](#step-2-system-boundary)  
3. [Step 3: Scenario Definition](#step-3-scenario-definition)  
4. [Step 4: Data & Assumptions](#step-4-data--assumptions)  
5. [Step 5: Outcome Modelling](#step-5-outcome-modelling)  
6. [Step 6: Trade-offs & Robustness](#step-6-trade-offs--robustness)  
7. [Step 7: Decision & Guardrails](#step-7-decision--guardrails)  
8. [Step 8: Communication & Refresh](#step-8-communication--refresh)  
9. [How This Became a Decision Cockpit](#how-this-became-a-decision-cockpit)

---

## Background Story

*Why “graduation” became a strategic blind spot*

Lendy is a fictitious, non-bank specialist lender focused primarily on **vehicle finance**. Its core customers are often borrowers rebuilding credit or stabilising cash flow—customers who value access, speed, and flexibility over headline pricing.

Ironically, this success creates a structural problem.

As customers repay reliably and their credit profiles improve, many become attractive to **prime lenders** offering lower rates. These customers “graduate” out of Lendy’s portfolio just as they become:

* Lower risk  
* More predictable  
* More profitable  

This attrition does not show up cleanly in traditional reporting:

* Credit dashboards focus on **defaults**, not successful exits  
* Churn metrics treat all exits as equal  
* Portfolio reports lag the underlying behavioural shift  

By the time leadership notices the impact, the best assets have already left.

Lendy’s executive team recognised that this was not a **prediction problem**. It was a **decision problem**:  
how to intervene selectively, credibly, and at scale without destroying margin or fairness.

[Open the Decision Cockpit](/cockpits/lendy-graduation-retention/)

> Local run note: pipeline artefacts are generated into `cockpits/lendy-graduation-retention/outputs` (including `raw/`, `clean/`, and `eval/`) when you run `python -m projects.lendy-graduation-retention.pipeline`.

---

## Step 1: Decision Framing

**Decision statement**

> *How should Lendy proactively retain high-quality vehicle loan customers who are likely to refinance elsewhere, without over-spending or undermining pricing discipline?*

**Why this decision matters now**

* Competitive pressure in prime lending is increasing  
* Scaling AUM is anticipated to increase acquisition costs faster than margins  
* Losing prime loans silently erodes portfolio quality  

**Time horizon**

* Near-term: next 6–18 months  
* Strategic: portfolio quality over 3–5 years  

**What success looks like**

* Retention spend is targeted over broadly applied  
* Incremental retained balance exceeds the total cost of incentives  
* Along with AUM growth by retention increasing, portfolio risk profile improves  

**What failure looks like**

* Blanket offers that erode margin  
* Retaining customers who would have stayed anyway  
* Incentivising customers who are unprofitable long-term  

**Stakeholders and incentives**

* **Credit & Risk** – protect portfolio quality  
* **Pricing & Treasury** – preserve margin discipline  
* **Sales & Marketing** – advocate for competitive offers  
* **Strategy & Analytics** – balance value, risk, and robustness  

---

## Step 2: System Boundary

This analysis deliberately narrows the system to support clear decision-making.

**In scope**

* Vehicle loan portfolio only  
* Loan-month behavioural dynamics  
* Repricing and retention offers  
* Retained balance and net interest impact  

**Out of scope (by design)**

* Cross-sell or lifetime customer value  
* Detailed customer communications strategy  
* Legal or regulatory offer constraints  
* Non-vehicle products  

**Controllable levers**

* Who receives an offer  
* When an offer is made  
* Offer intensity (rate reduction / incentive size)  

**External forces and uncertainty**

* Prime lender competitiveness  
* Customer refinancing behaviour  
* Macroeconomic conditions  

The model is intentionally simpler than reality—by design.

---

## Step 3: Scenario Definition

This case study evaluates decisions across **plausible futures**, not forecasts.

### Scenario A: Base / Status Quo

Prime lenders compete steadily. Graduation risk follows recent history. Retention offers have moderate influence on customer behaviour.

This scenario anchors the analysis.

---

### Scenario B: More Competitive Prime Market

Prime lenders become more aggressive on pricing and refinancing. Graduation accelerates, especially among lower-risk borrowers.

This scenario tests fragility.

---

### Scenario C: Less Competitive Prime Market

Prime competition softens due to funding or risk appetite constraints. Graduation slows naturally.

This scenario tests whether intervention is still justified.

---

None of these scenarios are predictions. Each exists to test whether the **decision logic holds up**.

---

## Step 4: Data & Assumptions

This case study uses **synthetic data**, but not “clean demo data”.

The objective is to demonstrate **production-like decision workflows**, not perfect modelling conditions.

### Synthetic, but realistic

Synthetic loan-level data was generated to resemble a specialist vehicle lending portfolio, then intentionally degraded to reflect operational realities.

The data pipeline explicitly separates:

* **Raw datasets** – direct synthetic outputs, untouched  
* **Clean datasets** – model-ready tables produced through defined cleaning rules  

Both are retained and auditable.

### Messy data generation (incremental)

Messiness is introduced in controlled stages.

**Messy Level 1**

* Inconsistent categorical values (e.g. introducer name variants)  
* Delayed reporting (credit score lags)  
* Partial histories (truncated loan-month windows)  

**Messy Level 2**

* Missing-at-random income data, more common for lower stability and broker-originated loans  
* Outliers such as score shocks and balance jumps  
* Measurement noise (rounded incomes, noisy balances)  

### Why this matters

Executives rarely trust models that only work on perfect data.

This approach demonstrates that:

* Assumptions are visible  
* Data quality issues are logged, not hidden  
* Decisions remain stable under plausible messiness  

**Key assumption**

> *The messiness is plausible, not chaotic. It reflects how operational data actually behaves.*

---

## Step 5: Outcome Modelling

The modelling focuses on **timing, behaviour, and incremental impact**—not just risk scores.

### Graduation risk

Graduation is modelled as a **time-to-event** problem:

* When is a customer likely to refinance elsewhere?  
* How does that risk evolve month by month?  

This matters because intervention timing drives value.

### Retention uplift

Not all customers respond to offers.

The analysis distinguishes between:

* Customers who would stay anyway  
* Customers who will leave regardless  
* Customers whose behaviour changes *because* of an offer  

This causal framing is essential.

### Outputs produced

The models generate:

* Graduation probabilities and expected timing  
* Estimated retention uplift from offers  
* Incremental retained balance and net interest value  

**Importantly:** modelling is only half the story.

Validation artefacts are published alongside results to demonstrate whether the models are *useful*, not just accurate.

---

## Step 6: Trade-offs & Robustness

### Risk is not the same as value

Targeting only the highest graduation risk is sub-optimal.

Effective decisions require balancing:

* **Risk** – likelihood of leaving  
* **Persuadability** – likelihood of responding to an offer  
* **Value** – retained balance and margin  

High-risk, low-value customers are poor targets.

### Budget changes the answer

When retention budgets are constrained, the optimal action changes materially.

Two deliberately simple policies are evaluated:

* **Top-Uplift (Top-K)**  
  Offer retention incentives to the top X loans ranked by estimated uplift.

* **Risk-Filtered Uplift**  
  Restrict offers to high-risk loans, then rank by uplift within that group.

These are not “optimal” strategies. They are **first policies** designed to support executive discussion.

### Where the strategy breaks down

* Offers become unprofitable if applied too broadly  
* Poor calibration erodes trust  
* Over-fitting to a single scenario leads to fragile decisions  

Robustness matters more than precision.

---

## Step 7: Decision & Guardrails

**Recommended decision approach**

Adopt a **policy-based retention strategy** that targets customers with both meaningful graduation risk *and* demonstrable uplift.

**Guardrails leadership should monitor**

* Incremental retained balance vs spend  
* Net interest impact after incentives  
* Distribution of offers across risk bands  

**Trigger points**

* Retention spend exceeds incremental value  
* Model calibration deteriorates materially  
* Competitive dynamics shift outside scenario bounds  

This is governance, not optimisation.

---

## Step 8: Communication & Refresh

**Executive communication**

Results are communicated using:

* Scenario ranges, not point estimates  
* Policy comparisons, not model internals  
* Clear articulation of trade-offs  

**Refresh cadence**

* Quarterly data refresh  
* Annual assumption review  
* Event-driven updates following market shocks  

This framework is designed to evolve.

---

### Method & Validation

To support executive trust, the cockpit surfaces **evidence**, not just outputs.

**Risk model evidence**

* Time-split validation (train on earlier months, validate on later months)  
* C-index and horizon AUC (3 / 6 / 12 months)  
* Calibration bins (predicted vs observed outcomes)  
* Risk-decile lift to confirm meaningful separation  

**Uplift model evidence**

* Qini / AUUC summaries showing uplift outperforms random ranking  
* Comparison of naive vs adjusted treatment effects to illustrate bias  

**Policy evidence**

* Incremental retained balance under each policy  
* Incremental net interest impact  
* Sensitivity to budget constraints  

The emphasis is on **decision confidence**, not technical novelty.

---

## How This Became a Decision Cockpit

The analysis was operationalised into a **Decision Cockpit** for executives and senior leaders.

**What users can adjust**

* Scenario selection  
* Retention budget  
* Offer intensity assumptions  

**What insights are surfaced**

* Expected retained balance  
* Incremental value vs cost  
* Policy comparisons under different scenarios  

**Method & Validation panel**

A dedicated panel provides three headline tiles:

* Survival model performance metrics  
* Uplift model impact and incremental value  
* Bias check (naive vs adjusted comparison)  

A “Download metrics” action links directly to CSV artefacts.

**What the cockpit does not do**

* It does not recommend individual customers  
* It does not automate decisions  
* It does not claim predictive certainty  

---

This case study demonstrates that the goal is not better prediction accuracy.

It is **better decisions under uncertainty**, supported by transparent assumptions, credible evidence, and clear trade-offs.
