// js/model.js
// Lightweight, deterministic model for V1 interactivity.
// TODO: Swap this later for precomputed Monte Carlo outputs or a real JS sim.

export function computeOutputs(state) {
  // Inputs
  const uplift = state.uplift_pct / 100;             // 0.00–0.30
  const incentiveBps = state.incentive_bps;          // 0–200
  const lossDeltaBps = state.loss_delta_bps;         // 0–150
  const avgLoan = state.avg_loan;                    // 5k–40k

  // Baseline assumptions (tunable later)
  const baseOriginations = 1200;                     // loans / period
  const baseMarginBps = 650;                         // net interest margin-ish
  const baseLossBps = 180;                           // baseline expected loss
  const fixedOpex = 220000;                          // fixed cost per period (demo)

  // Derived
  const originations = Math.round(baseOriginations * (1 + uplift));
  const volume = originations * avgLoan;

  const marginBps = baseMarginBps;
  const lossBps = baseLossBps + lossDeltaBps;

  // Profit components (very simplified)
  const grossMargin = volume * (marginBps / 10000);
  const expectedLoss = volume * (lossBps / 10000);
  const incentiveCost = volume * (incentiveBps / 10000);

  const expectedProfit = grossMargin - expectedLoss - incentiveCost - fixedOpex;

  // "Risk" proxy: higher uplift + higher loss delta => wider distribution
  const sigma = Math.max(60000, 90000 + 250000 * uplift + 1200 * lossDeltaBps);

  // Approximate P10 assuming normal distribution
  const p10 = expectedProfit - 1.2816 * sigma;

  // Approximate probability of loss under normal assumption
  // P(loss) = P(X < 0) = Phi((0 - mu)/sigma)
  const z = (0 - expectedProfit) / sigma;
  const probLoss = normalCdf(z);

  // Generate a synthetic distribution for plotting
  const dist = sampleNormal(expectedProfit, sigma, 1200);

  return {
    originations,
    volume,
    expectedProfit,
    p10,
    probLoss,
    dist
  };
}

// --- helpers ---
function normalCdf(x) {
  // Abramowitz and Stegun approximation
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-x * x / 2);
  let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  if (x > 0) p = 1 - p;
  return p;
}

function sampleNormal(mu, sigma, n) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    // Box-Muller
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    out[i] = mu + sigma * z0;
  }
  return out;
}
