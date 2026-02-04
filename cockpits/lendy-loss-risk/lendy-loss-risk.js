import { money, percent } from "/js/ui.js";
import {
    fetchJson,
    fetchText,
    number,
    parseCSV,
    qs,
    setText,
} from "/cockpits/shared/cockpit-utils.js";

const scenarioSelect = qs("#scenarioSelect");

/**
 * Build output paths for a given scenario.
 * @param {string} scenario - Scenario name used in output filenames.
 * @returns {{kpis: string, byProduct: string, overTime: string}} Output URLs.
 */
function paths(scenario) {
    return {
        kpis: `/cockpits/lendy-loss-risk/outputs/kpis_${scenario}.json`,
        byProduct: `/cockpits/lendy-loss-risk/outputs/loss_by_product_${scenario}.csv`,
        overTime: `/cockpits/lendy-loss-risk/outputs/loss_over_time_${scenario}.csv`,
    };
}

/**
 * Render KPI values from the JSON payload.
 * @param {Record<string, number>} k - KPI payload for the selected scenario.
 */
function renderKpis(k) {
    // Expected JSON schema:
    // balance, expected_loss, loss_rate, expected_default_balance, expected_default_count
    setText("#kpiBalance", money(k.balance));
    setText("#kpiExpectedLoss", money(k.expected_loss));
    setText("#kpiLossRate", percent(k.loss_rate, 2));
    setText("#kpiDefaultBal", money(k.expected_default_balance));
    setText("#kpiDefaultCount", number(k.expected_default_count));

    setText("#kpiBalanceHelp", "Total book balance");
    setText("#kpiExpectedLossHelp", "E[Loss] under scenario");
    setText("#kpiLossRateHelp", "E[Loss] / Balance");
    setText("#kpiDefaultBalHelp", "E[Default balance]");
    setText("#kpiDefaultCountHelp", "E[# defaults]");
}

/**
 * Render the "loss by product" table.
 * @param {Array<Record<string, string>>} rows - Parsed CSV rows.
 */
function renderLossByProduct(rows) {
    const tbody = qs("#tblLossByProduct tbody");
    tbody.innerHTML = "";

    if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-muted">No data.</td></tr>`;
        return;
    }

    // Your headers:
    // scenario_name,product,balance,expected_loss,expected_default_balance,expected_defaults_count,loss_rate
    for (const r of rows) {
        const product = r.product || "—";
        const tr = document.createElement("tr");
        tr.innerHTML = `
    <td class="text-capitalize">${product}</td>
    <td class="text-end">${money(r.balance)}</td>
    <td class="text-end">${money(r.expected_loss)}</td>
    <td class="text-end">${percent(r.loss_rate, 2)}</td>
    <td class="text-end">${number(r.expected_defaults_count)}</td>
    `;
        tbody.appendChild(tr);
    }
}

/**
 * Render the "loss over time" table.
 * @param {Array<Record<string, string>>} rows - Parsed CSV rows.
 */
function renderLossOverTime(rows) {
    const tbody = qs("#tblLossOverTime tbody");
    tbody.innerHTML = "";

    if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="4" class="text-muted">No data.</td></tr>`;
        return;
    }

    // Headers:    
    // scenario_name,months_since_origination,product,expected_loss_month,expected_defaults_count_month
    //
    // v1: aggregate across products to produce a portfolio-total timing curve by month
    const byMonth = new Map();

    for (const r of rows) {
        const m = Number(r.months_since_origination);
        if (!Number.isFinite(m)) continue;

        const el = Number(r.expected_loss_month) || 0;
        const dc = Number(r.expected_defaults_count_month) || 0;

        const cur = byMonth.get(m) || { expected_loss: 0, expected_default_count: 0 };
        cur.expected_loss += el;
        cur.expected_default_count += dc;
        byMonth.set(m, cur);
    }

    const months = Array.from(byMonth.keys()).sort((a, b) => a - b);

    for (const m of months) {
        const v = byMonth.get(m);
        const tr = document.createElement("tr");
        tr.innerHTML = `
    <td>${m}</td>
    <td class="text-end">${money(v.expected_loss)}</td>
    <td class="text-end">—</td>
    <td class="text-end">${number(v.expected_default_count)}</td>
    `;
        tbody.appendChild(tr);
    }
}

/**
 * Update download links for the current scenario.
 * @param {string} scenario - Scenario name used in output filenames.
 */
function setDownloadLinks(scenario) {
    const p = paths(scenario);

    qs("#dlLossByProduct").href = p.byProduct;
    qs("#dlLossOverTime").href = p.overTime;

    qs("#dlKpisJson").href = p.kpis;
    qs("#dlLossByProductCsv2").href = p.byProduct;
    qs("#dlLossOverTimeCsv2").href = p.overTime;
}

/**
 * Load scenario outputs and render tables/metrics.
 * @param {string} scenario - Scenario name to load.
 * @returns {Promise<void>} Resolves after rendering.
 */
async function loadScenario(scenario) {
    setDownloadLinks(scenario);

    qs("#tblLossByProduct tbody").innerHTML = `<tr><td colspan="5" class="text-muted">Loading…</td></tr>`;
    qs("#tblLossOverTime tbody").innerHTML = `<tr><td colspan="4" class="text-muted">Loading…</td></tr>`;

    try {
        const p = paths(scenario);
        const [kpis, byProdCsv, overTimeCsv] = await Promise.all([
            fetchJson(p.kpis),
            fetchText(p.byProduct),
            fetchText(p.overTime),
        ]);

        renderKpis(kpis);
        renderLossByProduct(parseCSV(byProdCsv));
        renderLossOverTime(parseCSV(overTimeCsv));
    } catch (e) {
        console.error(e);
        setText("#kpiBalance", "—");
        setText("#kpiExpectedLoss", "—");
        setText("#kpiLossRate", "—");
        setText("#kpiDefaultBal", "—");
        setText("#kpiDefaultCount", "—");

        qs("#tblLossByProduct tbody").innerHTML =
            `<tr><td colspan="5" class="text-danger">Failed to load scenario outputs. Check console + file paths.</td></tr>`;
        qs("#tblLossOverTime tbody").innerHTML =
            `<tr><td colspan="4" class="text-danger">Failed to load scenario outputs. Check console + file paths.</td></tr>`;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    scenarioSelect.addEventListener("change", (e) => loadScenario(e.target.value));
    loadScenario(scenarioSelect.value || "base");
});
