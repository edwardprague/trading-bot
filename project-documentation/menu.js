/*
 * Project-docs sidebar menu.
 *
 * To ADD a page: drop a new {href, label} entry into FRACTAL_DOCS_MENU
 * below. Every docs page that includes <script src="menu.js"></script>
 * picks up the change on next reload — no per-page edits.
 *
 * To ADD A SUB-PAGE: add a `parent` field pointing to the parent's
 * href, e.g. { href: "subpage-1.html", label: "Sub page 1",
 *              parent: "dashboard.html" }. Sub-items render indented
 * under their parent. Source order within the same parent is preserved;
 * the parent itself can appear anywhere in the array — children are
 * grouped under it regardless. Nesting deeper than one level is not
 * currently styled.
 *
 * To REMOVE a page: delete its entry from the array. If you remove a
 * parent that has children, those children become unanchored — they'll
 * still render but as top-level items.
 *
 * To REORDER: rearrange the entries. (Top-level order = top-level
 * position; sub-item order = position among siblings sharing the same
 * parent.)
 *
 * Each docs page must contain a single <aside id="sidebar"></aside>
 * element somewhere in <body>; that's where the rendered <nav> lands.
 * The active item is detected from window.location.pathname's basename,
 * so the .active class is applied automatically — don't hand-mark it.
 */

var FRACTAL_DOCS_MENU = [
    { href: "index.html", label: "Overview" },
    { href: "research.html", label: "Research & Development" },
    { href: "lem.html", label: "Liquidity Engeneering Model", parent: "research.html" },
    { href: "ema.html", label: "EMA Analysis", parent: "research.html" },
    { href: "entry-ideas.html", label: "Entry Ideas", parent: "research.html" },
    { href: "key-findings.html", label: "Key Findings", parent: "research.html" },
    { href: "potential-dev.html", label: "Potential Dev", parent: "research.html" },
    { href: "dashboard.html", label: "Dashboard" },
    { href: "backtesting.html", label: "Backtesting", parent: "dashboard.html" },
    { href: "regimes.html", label: "Regimes", parent: "dashboard.html" },
    { href: "discovery.html", label: "Discovery", parent: "dashboard.html" },
    { href: "versions.html", label: "Versions", parent: "dashboard.html" },
    { href: "technical.html", label: "Technical Setup" },
];

(function () {
    function escapeHtml(s) {
        return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    function currentPageName() {
        // Strip query/hash; take the last path segment. Empty path (e.g.
        // visiting /project-documentation/) treated as index.html so the
        // Overview entry highlights correctly.
        var path = (window.location.pathname || "").split("?")[0].split("#")[0];
        var name = path.split("/").pop();
        return name || "index.html";
    }

    /* Build the ordered render list by walking the source array and
       slotting children directly after their parent. We preserve the
       source order of siblings, and the parent's source-order position
       within the top level. Orphans (children whose parent isn't in the
       array) fall through as top-level items so the page is still
       reachable. */
    function flattenWithChildren(menu) {
        var byParent = {}; // parent href -> [child entries, in source order]
        var topLevel = []; // entries that are top-level (no parent), in source order
        var hrefs = {};
        menu.forEach(function (it) {
            hrefs[it.href] = true;
        });

        menu.forEach(function (item) {
            if (item.parent && hrefs[item.parent]) {
                if (!byParent[item.parent]) byParent[item.parent] = [];
                byParent[item.parent].push(item);
            } else {
                topLevel.push(item);
            }
        });

        var out = [];
        topLevel.forEach(function (item) {
            out.push({ item: item, level: 0 });
            (byParent[item.href] || []).forEach(function (child) {
                out.push({ item: child, level: 1 });
            });
        });
        return out;
    }

    function render() {
        var aside = document.getElementById("sidebar");
        if (!aside) return;
        var current = currentPageName();
        var ordered = flattenWithChildren(FRACTAL_DOCS_MENU);
        var items = ordered
            .map(function (row) {
                var item = row.item;
                var classes = [];
                if (item.href === current) classes.push("active");
                if (row.level > 0) classes.push("sub");
                var cls = classes.length ? ' class="' + classes.join(" ") + '"' : "";
                return '<a href="' + escapeHtml(item.href) + '"' + cls + ">" + escapeHtml(item.label) + "</a>";
            })
            .join("");
        aside.innerHTML = '<nav class="sidebar-nav">' + items + "</nav>";
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", render);
    } else {
        render();
    }
})();
