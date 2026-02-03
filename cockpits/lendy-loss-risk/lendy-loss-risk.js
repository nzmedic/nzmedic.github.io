import { money, percent } from "/js/ui.js";

const $ = (sel) => document.querySelector(sel);

const scenarioSelect = $("#scenarioSelect");

function paths(scenario) {
    return {
        kpis: `/cockpits/lendy-loss-risk/outputs/kpis_${scenario}.json`,
        byProduct: `/cockpits/lendy-loss-risk/outputs/loss_by_product_${scenario}.csv`,
        overTime: `/cockpits/lendy-loss-risk/outputs/loss_over_time_${scenario}.csv`,
    };
}

// Minimal numeric formatting for non-money counts
function number(x) {
    const n = Number(x);
    if (!Number.isFinite(n)) return "—";
    return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

// Minimal CSV parsing (controlled outputs)
function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter(Boolean);
    if (!lines.length) return [];
    const headers = lines[0].split(",").map(h => h.trim());
    return lines.slice(1).map(line => {
        const cols = line.split(",").map(c => c.trim());
        const row = {};
        headers.forEach((h, i) => row[h] = cols[i] ?? "");
        return row;
    });
}

async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`Failed to load ${url} (${res.status})`);
    return await res.json();
}

async function fetchText(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`Failed to load ${url} (${res.status})`);
    return await res.text();
}

function setText(sel, v) {
    $(sel).textContent = v;
}

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

function renderLossByProduct(rows) {
    const tbody = $("#tblLossByProduct tbody");
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

function renderLossOverTime(rows) {
    const tbody = $("#tblLossOverTime tbody");
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

function setDownloadLinks(scenario) {
    const p = paths(scenario);

    $("#dlLossByProduct").href = p.byProduct;
    $("#dlLossOverTime").href = p.overTime;

    $("#dlKpisJson").href = p.kpis;
    $("#dlLossByProductCsv2").href = p.byProduct;
    $("#dlLossOverTimeCsv2").href = p.overTime;
}

async function loadScenario(scenario) {
    setDownloadLinks(scenario);

    $("#tblLossByProduct tbody").innerHTML = `<tr><td colspan="5" class="text-muted">Loading…</td></tr>`;
    $("#tblLossOverTime tbody").innerHTML = `<tr><td colspan="4" class="text-muted">Loading…</td></tr>`;

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

        $("#tblLossByProduct tbody").innerHTML =
            `<tr><td colspan="5" class="text-danger">Failed to load scenario outputs. Check console + file paths.</td></tr>`;
        $("#tblLossOverTime tbody").innerHTML =
            `<tr><td colspan="4" class="text-danger">Failed to load scenario outputs. Check console + file paths.</td></tr>`;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    scenarioSelect.addEventListener("change", (e) => loadScenario(e.target.value));
    loadScenario(scenarioSelect.value || "base");
});
