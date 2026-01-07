// js/cockpit.js
import { computeOutputs } from "./model.js";

const els = {};
const state = {
  uplift_pct: 10,
  incentive_bps: 50,
  loss_delta_bps: 25,
  avg_loan: 10000
};

function money(x) {
  const sign = x < 0 ? "-" : "";
  const abs = Math.abs(x);
  return sign + "$" + abs.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function percent(x) {
  return (x * 100).toFixed(1) + "%";
}

function bind() {
  // sliders
  els.uplift = document.getElementById("uplift");
  els.incentive = document.getElementById("incentive");
  els.lossDelta = document.getElementById("lossDelta");
  els.avgLoan = document.getElementById("avgLoan");

  // readouts
  els.upliftVal = document.getElementById("upliftVal");
  els.incentiveVal = document.getElementById("incentiveVal");
  els.lossDeltaVal = document.getElementById("lossDeltaVal");
  els.avgLoanVal = document.getElementById("avgLoanVal");

  // KPIs
  els.kpiProfit = document.getElementById("kpiProfit");
  els.kpiP10 = document.getElementById("kpiP10");
  els.kpiProbLoss = document.getElementById("kpiProbLoss");
  els.kpiOriginations = document.getElementById("kpiOriginations");

  // chart divs
  els.profitChart = document.getElementById("profitChart");
  els.tradeoffChart = document.getElementById("tradeoffChart");

  // initial slider positions
  els.uplift.value = state.uplift_pct;
  els.incentive.value = state.incentive_bps;
  els.lossDelta.value = state.loss_delta_bps;
  els.avgLoan.value = state.avg_loan;

  // events
  els.uplift.addEventListener("input", () => { state.uplift_pct = Number(els.uplift.value); render(); });
  els.incentive.addEventListener("input", () => { state.incentive_bps = Number(els.incentive.value); render(); });
  els.lossDelta.addEventListener("input", () => { state.loss_delta_bps = Number(els.lossDelta.value); render(); });
  els.avgLoan.addEventListener("input", () => { state.avg_loan = Number(els.avgLoan.value); render(); });
}

function render() {
  // Safety checks (so “blank” becomes diagnosable)
  if (!window.Plotly) {
    console.error("Plotly not found. Check plotly script tag is loading.");
    return;
  }
  if (!els.profitChart || !els.tradeoffChart) {
    console.error("Chart div(s) not found:", {
      profitChart: els.profitChart,
      tradeoffChart: els.tradeoffChart
    });
    return;
  }

  // readouts
  els.upliftVal.textContent = `${state.uplift_pct}%`;
  els.incentiveVal.textContent = `${state.incentive_bps} bps`;
  els.lossDeltaVal.textContent = `${state.loss_delta_bps} bps`;
  els.avgLoanVal.textContent = money(state.avg_loan);

  const out = computeOutputs(state);

  // KPIs
  els.kpiProfit.textContent = money(out.expectedProfit);
  els.kpiP10.textContent = money(out.p10);
  els.kpiProbLoss.textContent = percent(out.probLoss);
  els.kpiOriginations.textContent = out.originations.toLocaleString();

  // --- Chart 1: Profit distribution histogram ---
  const hist = {
    x: out.dist,
    type: "histogram",
    nbinsx: 40,
    name: "Profit"
  };

  const layout1 = {
    margin: { l: 50, r: 20, t: 10, b: 40 },
    xaxis: { title: "Profit" },
    yaxis: { title: "Frequency" }
  };

  Plotly.react(els.profitChart, [hist], layout1, { displayModeBar: false, responsive: true });

  // --- Chart 2: Trade-off curve (profit vs loss delta) ---
  const xs = [];
  const ys = [];
  for (let d = 0; d <= 150; d += 5) {
    const tmp = { ...state, loss_delta_bps: d };
    const o = computeOutputs(tmp);
    xs.push(d);
    ys.push(o.expectedProfit);
  }

  const curve = { x: xs, y: ys, type: "scatter", mode: "lines", name: "Expected profit" };
  const point = {
    x: [state.loss_delta_bps],
    y: [out.expectedProfit],
    type: "scatter",
    mode: "markers",
    name: "Current setting"
  };

  const layout2 = {
    margin: { l: 60, r: 20, t: 10, b: 45 },
    xaxis: { title: "Loss-rate delta (bps)" },
    yaxis: { title: "Expected profit" }
  };

  Plotly.react(els.tradeoffChart, [curve, point], layout2, { displayModeBar: false, responsive: true });
}

document.addEventListener("DOMContentLoaded", () => {
  bind();
  render();
});
