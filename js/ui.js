// /js/ui.js
export function money(x, currency = "NZD") {
    const sign = x < 0 ? "-" : "";
    const abs = Math.abs(Number(x) || 0);
    return sign + "$" + abs.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

export function percent(x, dp = 1) {
    const n = Number(x);
    if (!Number.isFinite(n)) return "—";
    return (n * 100).toFixed(dp) + "%";
}

export function setBadge(el, status) {
    // status: "pass" | "warn" | "fail"
    el.classList.remove("text-bg-secondary", "text-bg-success", "text-bg-warning", "text-bg-danger");
    if (status === "pass") el.classList.add("text-bg-success");
    else if (status === "warn") el.classList.add("text-bg-warning");
    else if (status === "fail") el.classList.add("text-bg-danger");
    else el.classList.add("text-bg-secondary");
}
