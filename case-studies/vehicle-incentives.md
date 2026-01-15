//recall that github uses Jekyll to render markdown files
---
title: "Case study: Vehicle Introducer Incentives"
layout: default
---

# Case study: Vehicle Introducer Incentives

**Decision:**  
Should we increase introducer incentives to grow vehicle lending without breaching risk appetite or eroding margin?

[Open the interactive decision cockpit →](/cockpits/vehicle-incentives.html)

---

## Context

Vehicle lending growth was driven primarily through third-party introducers.  
Management wanted to increase originations, but prior experience showed that:

- Higher incentives can erode margin quickly
- Growth can introduce adverse selection
- Risk impacts are often visible only after volume ramps up

The challenge was to support a **growth decision with explicit downside visibility and guardrails**, rather than relying on point forecasts.

---

## Stakeholders and incentives

- **Executive sponsor** – accountable for profitable growth and capital efficiency  
- **Credit & risk** – focused on downside protection and early warning signals  
- **Analytics lead** – needed a pragmatic, reusable decision tool  
- **CTO / founder mindset** – preferred lightweight delivery with clear ROI  
- **Sales / channel** – incentivised to maximise volume, not risk-adjusted profit  

The solution needed to align these perspectives in a single artefact.

---

## Approach

The analysis was structured around decision-making rather than model complexity:

1. Frame the commercial question as a small set of controllable levers  
2. Quantify outcomes under uncertainty, not just expected values  
3. Make downside risk explicit and interpretable  
4. Translate risk appetite into simple policy guardrails  
5. Package the analysis as an interactive “decision cockpit”  

---

## What the cockpit shows

The cockpit is designed to support executive conversation, not technical inspection.

**Levers**
- Expected origination uplift
- Incentive cost per loan
- Assumed deterioration in credit losses
- Average loan size

**Outputs**
- Expected profit
- Downside profit (P10)
- Probability of loss
- Trade-off curves showing diminishing returns to risk

**Guardrails**
- Maximum acceptable probability of loss
- Minimum acceptable downside profit
- Caps on incentive intensity and loss-rate drift

---

## Governance and operating model

The output of the analysis was not “a model”, but a **decision plus an operating approach**:

- Pilot-scale deployment
- Weekly monitoring during ramp-up
- Pre-defined thresholds triggering intervention
- Clear actions when guardrails are breached (pause, tighten, redesign)

This allowed growth to proceed without relying on hindsight.

---

## Outcome

The result was a decision-ready artefact that:

- Made trade-offs explicit
- Reduced reliance on single-point forecasts
- Created shared understanding across stakeholders
- Could be reused for ongoing governance, not just initial approval

The initial version was intentionally lightweight; accuracy could be increased without changing the decision framework.

---

## Next iteration

Planned enhancements included:

- Replacing the synthetic engine with full Monte Carlo outputs  
- Adding scenario saving and comparison  
- Modelling channel mix and borrower quality feedback loops  
- Exporting one-page board summaries per scenario  

---

## What this demonstrates

- Executive-ready framing of complex decisions  
- Risk-aware thinking under uncertainty  
- Translation of analytics into policy and guardrails  
- Pragmatic delivery with a clear roadmap  

