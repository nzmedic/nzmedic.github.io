# Lendy Loss-Risk Decision Cockpit (MVP)

## Live cockpit (GitHub Pages)
- View the latest generated cockpit here: <your Pages URL>

## Build status
![Build & Deploy Cockpit to Pages](../../actions/workflows/pages.yml/badge.svg)


This repo implements a **minimum viable model stack** to support a sub-portfolio loss-risk cockpit for a specialist lender.

It intentionally uses simple, explainable components:
- **PD** (how many fail)
- **Timing curve** (when they fail)
- **LGD** (how costly failure is)
- **Scenario multipliers** (stress knobs)

The goal is to provide decision-grade visibility into:
- expected losses by sub-portfolio (vehicle, mortgage, etc.)
- loss emergence over time (proxy for arrears pressure)
- sensitivity to stress scenarios

## Quickstart

```bash
pip install -r requirements.txt

python -m scripts.run_cockpit --scenario base
python -m scripts.run_cockpit --scenario mild_stress
python -m scripts.run_cockpit --scenario severe_stress

