/**
 * Query a single element in the DOM.
 * @param {string} selector - CSS selector for the element.
 * @param {Document|HTMLElement} [root=document] - Root node to query within.
 * @returns {HTMLElement|null} The matched element, if any.
 */
export function qs(selector, root = document) {
    return root.querySelector(selector);
}

/**
 * Format a value as an integer-like string with locale separators.
 * @param {number|string} value - Input value to format.
 * @returns {string} The formatted value or an em dash if invalid.
 */
export function number(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

/**
 * Format a value as a fixed precision number.
 * @param {number|string} value - Input value to format.
 * @param {number} [digits=2] - Number of decimal places to keep.
 * @returns {string} The formatted value or an em dash if invalid.
 */
export function fixed(value, digits = 2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    return n.toFixed(digits);
}

/**
 * Convert a value to a number with a fallback.
 * @param {number|string} value - Input value to convert.
 * @param {number|null} [fallback=0] - Value to return if conversion fails.
 * @returns {number|null} Parsed numeric value or fallback.
 */
export function toNum(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

/**
 * Parse a minimal CSV string into row objects.
 * @param {string} text - Raw CSV text (no quoted commas expected).
 * @returns {Array<Record<string, string>>} Parsed rows.
 */
export function parseCSV(text) {
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

/**
 * Fetch a JSON payload with no-store caching.
 * @param {string} url - URL to request.
 * @returns {Promise<unknown>} Parsed JSON response.
 */
export async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`Failed to load ${url} (${res.status})`);
    return await res.json();
}

/**
 * Fetch a text payload with no-store caching.
 * @param {string} url - URL to request.
 * @returns {Promise<string>} Text response.
 */
export async function fetchText(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`Failed to load ${url} (${res.status})`);
    return await res.text();
}

/**
 * Update text content for a selector.
 * @param {string} selector - CSS selector for the element.
 * @param {string} value - Text content to set.
 */
export function setText(selector, value) {
    const el = qs(selector);
    if (el) el.textContent = value;
}
