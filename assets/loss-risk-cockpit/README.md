# Lendy Loss-Risk Decision Cockpit (MVP)

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
python scripts/run_cockpit.py --scenario base
python scripts/run_cockpit.py --scenario mild_stress
python scripts/run_cockpit.py --scenario severe_stress
