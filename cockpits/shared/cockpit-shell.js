/**
 * Build shared cockpit navigation markup.
 * @param {HTMLElement} container - Placeholder element to populate.
 */
function buildNav(container) {
    const brandHref = container.dataset.brandHref || "/";
    const brandLabel = container.dataset.brandLabel || "Decision Cockpits";
    const cockpitsHref = container.dataset.cockpitsHref || "/cockpits/";
    const caseStudyHref = container.dataset.caseStudy || "";
    const caseStudyLabel = container.dataset.caseStudyLabel || "Case study";

    const caseStudyLink = caseStudyHref
        ? `<a class="btn btn-sm btn-outline-light" href="${caseStudyHref}">${caseStudyLabel}</a>`
        : "";

    container.innerHTML = `
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand fw-semibold" href="${brandHref}">${brandLabel}</a>
            <div class="ms-auto d-flex gap-2">
                <a class="btn btn-sm btn-outline-light" href="${cockpitsHref}">Cockpits</a>
                ${caseStudyLink}
            </div>
        </div>
    </nav>
    `;
}

document.addEventListener("DOMContentLoaded", () => {
    const container = document.querySelector("[data-cockpit-nav]");
    if (!container) return;
    buildNav(container);
});
