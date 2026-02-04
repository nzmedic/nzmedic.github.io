import { money, percent } from "/js/ui.js";
import {
    fetchText,
    fixed,
    number,
    parseCSV,
    qs,
    setText,
    toNum,
} from "/cockpits/shared/cockpit-utils.js";

const scenarioSelect = qs("#scenarioSelect");
const offerBpsSelect = qs("#offerBpsSelect");
const loanSearch = qs("#loanSearch");

const budgetSlider = qs("#budgetSlider");
const budgetLabel = qs("#budgetLabel");
const budgetTypeCount = qs("#budgetTypeCount");
const budgetTypeCost = qs("#budgetTypeCost");

let selectedLoanId = null;

// Cached data for current scenario
let cache = {
    scenario: null,
    risk: [],
    uplift: [],
    frontier: [],
    metrics: [],
    explainGlobal: [],
    explainLocal: [],
};

/**
 * Build output paths for a given scenario.
 * @param {string} scenario - Scenario name used in output filenames.
 * @returns {{risk: string, uplift: string, frontier: string, metrics: string, explainGlobal: string, explainLocal: string}}
 * Output URLs.
 */
function paths(scenario) {
    return {
        risk: `/cockpits/lendy-graduation-retention/outputs/graduation_risk_by_loan_${scenario}.csv`,
        uplift: `/cockpits/lendy-graduation-retention/outputs/uplift_by_loan_${scenario}.csv`,
        frontier: `/cockpits/lendy-graduation-retention/outputs/frontier_${scenario}.csv`,
        metrics: `/cockpits/lendy-graduation-retention/outputs/model_metrics_${scenario}.csv`,
        explainGlobal: `/cockpits/lendy-graduation-retention/outputs/explainability_global_${scenario}.csv`,
        explainLocal: `/cockpits/lendy-graduation-retention/outputs/explainability_local_${scenario}.csv`,
    };
}

/**
 * Get the selected budget type from the UI.
 * @returns {"count"|"cost"} Budget type.
 */
function getBudgetType() {
    return budgetTypeCost.checked ? "cost" : "count";
}

/**
 * Sync the budget slider UI to the selected type.
 */
function syncBudgetUi() {
    const t = getBudgetType();

    if (t === "count") {
        budgetSlider.min = 100;
        budgetSlider.max = 2000;
        budgetSlider.step = 50;
        if (!Number.isFinite(Number(budgetSlider.value))) budgetSlider.value = 500;
        budgetLabel.textContent = `${number(budgetSlider.value)} offers`;
    } else {
        budgetSlider.min = 5000;
        budgetSlider.max = 150000;
        budgetSlider.step = 5000;
        if (!Number.isFinite(Number(budgetSlider.value))) budgetSlider.value = 50000;
        budgetLabel.textContent = `${money(budgetSlider.value)} cost`;
    }
}

/**
 * Update download links for the current scenario.
 * @param {string} scenario - Scenario name used in output filenames.
 */
function setDownloadLinks(scenario) {
    const p = paths(scenario);
    qs("#dlRiskCsv").href = p.risk;
    qs("#dlUpliftCsv").href = p.uplift;
    qs("#dlFrontierCsv").href = p.frontier;
    qs("#dlMetricsCsv").href = p.metrics;
    qs("#dlExplainGlobalCsv").href = p.explainGlobal;
    qs("#dlExplainLocalCsv").href = p.explainLocal;
}

/** Decision policy:
 * - filter to offer bps
 * - keep ite > 0 and not do_not_disturb
 * - rank by incremental_nii
 * - apply budget (count or cost)
 */
/**
 * Select target loans based on budget and offer settings.
 * @param {Array<Record<string, string>>} upliftRows - Uplift rows for the scenario.
 * @returns {Array<Record<string, string>>} Targeted loan rows.
 */
function chooseTargets(upliftRows) {
    const bps = toNum(offerBpsSelect.value, 100);
    const budgetType = getBudgetType();
    const budgetValue = toNum(budgetSlider.value, budgetType === "count" ? 500 : 50000);

    let rows = upliftRows
        .filter(r => toNum(r.treatment_bps) === bps)
        .filter(r => (r.segment || "") !== "do_not_disturb")
        .filter(r => toNum(r.ite_retention_12m, 0) > 0);

    rows.sort((a, b) => toNum(b.incremental_nii) - toNum(a.incremental_nii));

    if (budgetType === "count") return rows.slice(0, budgetValue);

    let cum = 0;
    const chosen = [];
    for (const r of rows) {
        const bal = toNum(r.balance);
        const cost = bal * (bps / 10000.0) * 1.0; // 1 year proxy
        if (cum + cost > budgetValue) break;
        chosen.push(r);
        cum += cost;
    }
    return chosen;
}

/**
 * Compute headline insight metrics and update the UI.
 * @param {Array<Record<string, string>>} riskRows - Risk rows for the scenario.
 * @param {Array<Record<string, string>>} upliftRows - Uplift rows for the scenario.
 * @param {Array<Record<string, string>>} chosen - Targeted loan rows.
 * @returns {{expGrads: number, riskBal: number, retainedAum: number, incNii: number, roi: number|null}}
 * Computed metrics.
 */
function computeInsights(riskRows, upliftRows, chosen) {
    const loans = riskRows.length;

    // Expected grads (12m) approximated as Σ p12
    const expGrads = riskRows.reduce((a, r) => a + toNum(r.prob_graduate_12m), 0);

    // At-risk balance proxy: Σ balance × p12
    const riskBal = riskRows.reduce((a, r) => a + toNum(r.balance) * toNum(r.prob_graduate_12m), 0);

    // Persuadable share (for current offer bps)
    const bps = toNum(offerBpsSelect.value, 100);
    const upliftBps = upliftRows.filter(r => toNum(r.treatment_bps) === bps);
    const persuadables = upliftBps.filter(r => (r.segment || "") === "persuadable");
    const persuadableCount = persuadables.length;

    const meanItePers = persuadables.length
        ? persuadables.reduce((a, r) => a + toNum(r.ite_retention_12m), 0) / persuadables.length
        : 0;

    const retainedAum = chosen.reduce((a, r) => a + toNum(r.incremental_retained_balance), 0);
    const incNii = chosen.reduce((a, r) => a + toNum(r.incremental_nii), 0);

    // Cost proxy for chosen (for ROI ratio)
    let costProxy = 0;
    for (const r of chosen) {
        costProxy += toNum(r.balance) * (bps / 10000.0) * 1.0;
    }
    const roi = costProxy > 0 ? (incNii / costProxy) : null;

    // Recommendation line
    const budgetType = getBudgetType();
    const budgetValue = toNum(budgetSlider.value);
    const budgetLabelText = budgetType === "count" ? `${number(budgetValue)} offers` : `${money(budgetValue)} cost`;

    setText("#recommendationLine",
        `With ${bps} bps and ${budgetLabelText}, target ${number(chosen.length)} loans to retain ${money(retainedAum)} AUM and ${money(incNii)} incremental NII (12m proxy).`
    );

    // KPI: Graduation pressure
    setText("#kpiPressureMain", `${number(Math.round(expGrads))} expected graduations`);
    setText("#kpiPressureSub", `${money(riskBal)} at-risk balance (Σ balance×P12) across ${number(loans)} loans`);

    // KPI: Opportunity
    setText("#kpiOpportunityMain", `${number(persuadableCount)} persuadables`);
    setText("#kpiOpportunitySub", `Mean ITE among persuadables: ${percent(meanItePers, 1)} (offer ${bps} bps)`);

    // KPI: ROI
    setText("#kpiRoiMain", `${money(incNii)} incremental NII`);
    setText("#kpiRoiSub", roi === null ? "ROI: —" : `ROI proxy: ${fixed(roi, 2)}x (NII / cost)`);

    return { expGrads, riskBal, retainedAum, incNii, roi };
}

/**
 * Render the uplift scatter plot.
 * @param {Array<Record<string, string>>} upliftRows - Uplift rows for the scenario.
 */
function renderUpliftScatter(upliftRows) {
    const bps = toNum(offerBpsSelect.value, 100);

    // Filter to current offer bps
    const rows = upliftRows.filter(r => toNum(r.treatment_bps) === bps);

    // Light sampling if huge (keeps Plotly responsive)
    const MAX_POINTS = 2500;
    const sampled = rows.length > MAX_POINTS
        ? rows.filter((_, i) => i % Math.ceil(rows.length / MAX_POINTS) === 0)
        : rows;

    // Group by segment
    const groups = new Map();
    for (const r of sampled) {
        const seg = (r.segment || "unknown");
        if (!groups.has(seg)) groups.set(seg, []);
        groups.get(seg).push(r);
    }

    // Marker size from balance (optional)
    const sizes = sampled.map(r => Math.sqrt(Math.max(0, toNum(r.balance, 0))) / 50); // heuristic
    const maxSize = Math.max(6, ...sizes);
    const minSize = 6;

    const traces = [];
    for (const [seg, g] of groups.entries()) {
        const xs = g.map(r => toNum(r.mu0_retention, null)).filter(v => v !== null);
        // To keep alignment of hover + arrays, rebuild from g and allow nulls:
        const x = g.map(r => toNum(r.mu0_retention, null));
        const y = g.map(r => toNum(r.ite_retention_12m, null));
        const text = g.map(r => `Loan ${r.loan_id}`);

        const markerSize = g.map(r => {
            const s = Math.sqrt(Math.max(0, toNum(r.balance, 0))) / 50;
            return Math.max(minSize, Math.min(maxSize, s));
        });
        //scattergl to plot actual points. Plotly WebGL smooths 1k+ points
        traces.push({
            name: seg.replaceAll("_", " "),
            type: "scattergl",
            mode: "markers",
            x,
            y,
            text,
            marker: { size: markerSize, opacity: 0.8 },
            hovertemplate:
                "%{text}<br>" +
                "μ0: %{x:.2%}<br>" +
                "ITE (12m): %{y:.2%}<extra></extra>"
        });
    }

    const layout = {
        margin: { l: 55, r: 10, t: 10, b: 45 },
        xaxis: { title: "Baseline retention (μ0)", tickformat: ".0%", range: [0.6, 1.0] },
        yaxis: { title: "ITE retention (12m)", tickformat: ".0%" },
        legend: { orientation: "h", x: 0, y: -0.2 },
        shapes: [
            // y=0 line for context
            {
                type: "line",
                x0: 0.6, x1: 1.0,
                y0: 0, y1: 0,
                line: { width: 1, dash: "dot" }
            }
        ]
    };

    Plotly.newPlot("upliftScatter", traces, layout, { displayModeBar: false, responsive: true });
}

/**
 * Render the targets table for the chosen loans.
 * @param {Array<Record<string, string>>} chosen - Targeted loan rows.
 * @param {Array<Record<string, string>>} riskRows - Risk rows for the scenario.
 */
function renderTargetsTable(chosen, riskRows) {
    const tbody = qs("#tblTargets tbody");
    tbody.innerHTML = "";

    if (!chosen.length) {
        tbody.innerHTML = `<tr><td colspan="6" class="text-muted">No targets under current settings.</td></tr>`;
        return;
    }

    const q = (loanSearch.value || "").trim();
    let rows = chosen.slice();
    if (q) rows = rows.filter(r => String(r.loan_id).includes(q));

    // join expected time to graduate from risk table for display
    const ettgByLoan = new Map(riskRows.map(r => [String(r.loan_id), r.expected_time_to_graduate_months]));

    const top = rows.slice(0, 60);
    for (const r of top) {
        const loanId = String(r.loan_id);
        const ettg = ettgByLoan.get(loanId);

        const tr = document.createElement("tr");
        tr.innerHTML = `
                <td class="mono">${loanId}</td>
                <td class="text-end">${money(r.balance)}</td>
                <td class="text-end">${(r.segment || "—").replaceAll("_", " ")}</td>
                <td class="text-end">${percent(r.ite_retention_12m, 1)}</td>
                <td class="text-end">${money(r.incremental_nii)}</td>
                <td class="text-end">${fixed(ettg, 1)}</td>
            `;

        tr.addEventListener("click", () => {
            selectedLoanId = loanId;
            renderLocalExplain(cache.explainLocal, cache.explainGlobal);
        });

        tbody.appendChild(tr);
    }
}

/**
 * Render local explainability details for the selected loan.
 * @param {Array<Record<string, string>>} localRows - Local explainability rows.
 */
function renderLocalExplain(localRows, globalRows = []) {
    const tbody = qs("#tblLocalExplain tbody");
    const label = qs("#selectedLoanLabel");

    if (!selectedLoanId) {
        label.textContent = "Select a loan from the target list.";
        tbody.innerHTML = `<tr><td colspan="3" class="text-muted">No loan selected.</td></tr>`;
        return;
    }

    let labelSuffix = "";

    let rows = localRows.filter(r => String(r.loan_id) === selectedLoanId && r.model_kind === "graduation_risk");
    if (!rows.length) rows = localRows.filter(r => String(r.loan_id) === selectedLoanId && r.model_kind === "uplift_surrogate");

    let formattedRows = [];
    if (rows.length) {
        formattedRows = rows.slice().sort((a, b) => toNum(a.rank) - toNum(b.rank)).slice(0, 10).map(r => ({
            rank: r.rank,
            feature: r.feature,
            contribution: r.contribution
        }));
    } else {
        const globals = globalRows
            .filter(r => r.model_kind === "graduation_risk")
            .slice()
            .sort((a, b) => toNum(b.importance) - toNum(a.importance))
            .slice(0, 10)
            .map((r, i) => ({
                rank: i + 1,
                feature: r.feature,
                contribution: r.importance
            }));

        if (globals.length) {
            formattedRows = globals;
            labelSuffix = " (global)";
        }
    }

    label.textContent = `Loan ${selectedLoanId} – top reason codes${labelSuffix}`;

    if (!formattedRows.length) {
        tbody.innerHTML = `<tr><td colspan="3" class="text-muted">No explainability rows found for this loan.</td></tr>`;
        return;
    }

    tbody.innerHTML = "";
    for (const r of formattedRows) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
                    <td class="mono">${r.rank || "—"}</td>
                    <td>${r.feature || "—"}</td>
                    <td class="text-end mono">${fixed(r.contribution, 4)}</td>
                `;
        tbody.appendChild(tr);
    }
}

/**
 * Render the graduation risk histogram.
 * @param {Array<Record<string, string>>} riskRows - Risk rows for the scenario.
 */
function renderRiskChart(riskRows) {
    const p12 = riskRows.map(r => toNum(r.prob_graduate_12m)).filter(v => Number.isFinite(v));
    const trace = {
        x: p12,
        type: "histogram",
        nbinsx: 20,
        hovertemplate: "P(graduate 12m): %{x:.2f}<br>Count: %{y}<extra></extra>"
    };

    const layout = {
        margin: { l: 40, r: 10, t: 10, b: 40 },
        xaxis: { title: "P(graduate in 12m)", tickformat: ".0%" },
        yaxis: { title: "Loans" },
        bargap: 0.05
    };

    Plotly.newPlot("riskChart", [trace], layout, { displayModeBar: false, responsive: true });
}

/**
 * Render the segment breakdown pie chart.
 * @param {Array<Record<string, string>>} upliftRows - Uplift rows for the scenario.
 */
function renderSegmentChart(upliftRows) {
    const bps = toNum(offerBpsSelect.value, 100);
    const rows = upliftRows.filter(r => toNum(r.treatment_bps) === bps);

    const counts = {};
    for (const r of rows) {
        const s = (r.segment || "unknown");
        counts[s] = (counts[s] || 0) + 1;
    }

    const labels = Object.keys(counts);
    const values = labels.map(k => counts[k]);

    const trace = {
        labels,
        values,
        type: "pie",
        hole: 0.55,
        textinfo: "label+percent",
        hovertemplate: "%{label}<br>%{value} loans (%{percent})<extra></extra>"
    };

    const layout = {
        margin: { l: 10, r: 10, t: 10, b: 10 },
        showlegend: false
    };

    Plotly.newPlot("segmentChart", [trace], layout, { displayModeBar: false, responsive: true });
}

/**
 * Render the uplift frontier chart.
 * @param {Array<Record<string, string>>} frontierRows - Frontier rows for the scenario.
 */
function renderFrontierChart(frontierRows) {
    const t = getBudgetType();
    const budgetValue = toNum(budgetSlider.value);

    const rows = frontierRows
        .filter(r => (r.budget_type || "") === t)
        .slice()
        .sort((a, b) => toNum(a.budget_value) - toNum(b.budget_value));

    const x = rows.map(r => toNum(r.budget_value));
    const y = rows.map(r => toNum(r.incremental_nii));

    const line = {
        x, y,
        type: "scatter",
        mode: "lines+markers",
        hovertemplate: `Budget: %{x}<br>Inc NII: %{y}<extra></extra>`
    };

    // Find nearest point to current budget for marker
    let idx = 0;
    let best = Infinity;
    for (let i = 0; i < x.length; i++) {
        const d = Math.abs(x[i] - budgetValue);
        if (d < best) { best = d; idx = i; }
    }

    const dot = {
        x: [x[idx]],
        y: [y[idx]],
        type: "scatter",
        mode: "markers",
        marker: { size: 12 },
        hovertemplate: `Selected<br>Budget: %{x}<br>Inc NII: %{y}<extra></extra>`
    };

    const layout = {
        margin: { l: 50, r: 10, t: 10, b: 45 },
        xaxis: { title: t === "count" ? "Offers made" : "Offer cost ($)" },
        yaxis: { title: "Incremental NII ($)" }
    };

    Plotly.newPlot("frontierChart", [line, dot], layout, { displayModeBar: false, responsive: true });
}

/**
 * Render headline model metrics.
 * @param {Array<Record<string, string>>} metricsRows - Metrics rows for the scenario.
 */
function renderHeadlineMetrics(metricsRows) {
    // Use whatever exists; fall back gracefully.
    // Expect columns: scenario_name, model_name, metric_name, metric_value, notes
    const byName = (model, metric) => {
        const r = metricsRows.find(x => x.model_name === model && x.metric_name === metric);
        return r ? r.metric_value : null;
    };

    // Survival
    const cindex = byName("survival", "c_index") ?? byName("graduation_timing", "c_index");
    const auc12 = byName("survival", "auc_12m") ?? byName("graduation_timing", "auc_12m");
    setText("#metricSurvival", cindex !== null ? `C-index ${fixed(cindex, 3)}` : "—");
    setText("#metricSurvivalSub", auc12 !== null ? `AUC@12m ${fixed(auc12, 3)}` : "See metrics CSV");

    // Uplift
    const auuc = byName("uplift", "auuc") ?? byName("uplift", "qini_auc");
    setText("#metricUplift", auuc !== null ? `AUUC ${fixed(auuc, 4)}` : "—");
    setText("#metricUpliftSub", "Incremental value under targeting");

    // Bias check
    const naive = byName("uplift", "naive_effect") ?? byName("bias", "naive_effect");
    const adj = byName("uplift", "adjusted_effect") ?? byName("bias", "adjusted_effect");
    if (naive !== null && adj !== null) {
        setText("#metricBias", `Naive ${fixed(naive, 4)} → Adjusted ${fixed(adj, 4)}`);
        setText("#metricBiasSub", "Corrects for biased offer assignment");
    } else {
        setText("#metricBias", "Bias demo");
        setText("#metricBiasSub", "See metrics CSV");
    }
}

/**
 * Render all cockpit visualizations and tables.
 */
function renderAll() {
    syncBudgetUi();

    const chosen = chooseTargets(cache.uplift);
    computeInsights(cache.risk, cache.uplift, chosen);

    // Charts
    renderRiskChart(cache.risk);
    renderUpliftScatter(cache.uplift);
    renderSegmentChart(cache.uplift);
    renderFrontierChart(cache.frontier);

    // Tables
    renderTargetsTable(chosen, cache.risk);
    renderLocalExplain(cache.explainLocal, cache.explainGlobal);

    // Metrics
    renderHeadlineMetrics(cache.metrics);
}

/**
 * Set the cockpit into a loading state.
 */
function setLoading() {
    setText("#recommendationLine", "Loading…");
    $("#tblTargets tbody").innerHTML = `<tr><td colspan="6" class="text-muted">Loading…</td></tr>`;
    $("#tblLocalExplain tbody").innerHTML = `<tr><td colspan="3" class="text-muted">No loan selected.</td></tr>`;
}

/**
 * Set the cockpit into an error state.
 */
function setError() {
    setText("#recommendationLine", "Failed to load scenario outputs. Check console + file paths.");
    $("#tblTargets tbody").innerHTML =
        `<tr><td colspan="6" class="text-danger">Failed to load scenario outputs. Check console + file paths.</td></tr>`;
}

/**
 * Load scenario outputs and render the cockpit.
 * @param {string} scenario - Scenario name to load.
 * @returns {Promise<void>} Resolves after rendering.
 */
async function loadScenario(scenario) {
    setLoading();
    setDownloadLinks(scenario);
    selectedLoanId = null;

    try {
        const p = paths(scenario);

        const [riskCsv, upliftCsv, frontierCsv, metricsCsv, explainGlobalCsv, explainLocalCsv] = await Promise.all([
            fetchText(p.risk),
            fetchText(p.uplift),
            fetchText(p.frontier),
            fetchText(p.metrics),
            fetchText(p.explainGlobal),
            fetchText(p.explainLocal),
        ]);

        cache = {
            scenario,
            risk: parseCSV(riskCsv),
            uplift: parseCSV(upliftCsv),
            frontier: parseCSV(frontierCsv),
            metrics: parseCSV(metricsCsv),
            explainGlobal: parseCSV(explainGlobalCsv),
            explainLocal: parseCSV(explainLocalCsv),
        };

        renderAll();
    } catch (e) {
        console.error(e);
        setError();
    }
}

document.addEventListener("DOMContentLoaded", () => {
    scenarioSelect.addEventListener("change", (e) => loadScenario(e.target.value));

    offerBpsSelect.addEventListener("change", () => renderAll());

    budgetSlider.addEventListener("input", () => renderAll());

    budgetTypeCount.addEventListener("change", () => renderAll());
    budgetTypeCost.addEventListener("change", () => renderAll());

    loanSearch.addEventListener("input", () => renderAll());

    syncBudgetUi();
    loadScenario(scenarioSelect.value || "base");
});
