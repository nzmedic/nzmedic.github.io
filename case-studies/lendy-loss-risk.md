---
title: Lendy — Sub-portfolio Loss Risk
---

# Lendy: Sub-portfolio loss risk

**Cockpit:** [/cockpits/lendy-loss-risk/](/cockpits/lendy-loss-risk/)

## Background story

Lendy has grown quickly across four lending products: **vehicle**, **second-tier mortgages**, **personal loans**, and **small business**. While the business tracks overall arrears and losses, the executive team kept running into the same problem:

> “When credit conditions tighten, which sub-portfolio drives the losses — and when do those losses land?”

In other words, it wasn’t enough to know *total expected loss*. Lendy needed to understand **loss concentration** (by product) and **loss timing** (over months since origination) under different stress scenarios.

## Decision framing

**Decision:** How should Lendy adjust risk appetite, pricing, and growth focus across sub-portfolios given expected loss concentration and timing?

**Why it matters now:**
- AUM growth means small shifts in loss rate create large swings in total loss dollars.
- Different products behave differently under stress (default likelihood, severity, and timing).
- Treasury and risk teams need a consistent, scenario-based view to align funding strategy, provisioning, and targets.

## What this cockpit shows

The cockpit is designed for a *portfolio-level* view with a strategic lens. You can switch between scenarios and see how outcomes shift.

### Scenario selector
- `base`: current-state assumptions
- `mild_stress`: modest deterioration in credit performance
- `severe_stress`: larger shock with worse default + loss severity dynamics

### Key KPIs
- **Balance:** total outstanding balance across the four sub-portfolios
- **Expected Loss:** total expected loss under the selected scenario
- **Loss Rate:** expected loss divided by balance
- **Expected Default Balance:** expected balance that defaults (before recoveries)
- **Expected Default Count:** expected number of defaults

These KPIs are intended to answer: “How big is the problem under this scenario?”

### Loss by product (table)
A breakdown across:
- vehicle
- mortgage
- personal
- small_business

This answers: “Where are losses concentrated?”

Common patterns to look for:
- One product dominates expected loss even if it’s not the largest balance.
- Loss rate spikes more in one product under stress, revealing concentration risk.

### Loss over time (months since origination)
A view of expected loss and defaults by **months since origination**.

This answers: “When do losses arrive?”

Why timing matters:
- Provisioning and capital buffers depend on *when* losses are expected to land.
- Operational planning (collections capacity) needs lead time if defaults cluster in particular months.

## What levers exist (and what they mean)

This v1 cockpit focuses on **scenario selection** as the primary lever.

In the underlying modelling work, scenarios typically represent changes to assumptions like:
- Probability of default (PD) uplift
- Loss given default (LGD) uplift / recovery deterioration
- Timing curve shifts (defaults occur earlier/later)

The cockpit intentionally keeps the UI minimal: it’s built to support executive decision-making without turning the page into a full modelling workbench.

## Outputs and transparency

The cockpit includes download links for the underlying output files used to render the view:
- **KPIs (JSON)**
- **Loss by product (CSV)**
- **Loss over time (CSV)**

This is deliberate: stakeholders can validate the numbers, and analysts can reuse the outputs in other reporting.

## How to interpret results responsibly

This cockpit is a decision support tool:
- Use it to compare concentration and timing across scenarios.
- Treat outputs as *scenario-based expectations*, not point forecasts.
- Combine it with credit policy constraints, funding strategy, and operational capacity planning.

---


