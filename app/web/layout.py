"""Design tokens, page shell, and small reusable HTML components.

Every function here returns a plain ``str`` of HTML/CSS -- no templating
engine, matching the rest of the app's self-contained-f-string approach.
Callers are responsible for ``html.escape``-ing any untrusted text before it
reaches these functions; this module only assembles already-safe fragments.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
# A restrained, professional light theme (with a dark-mode remap) inspired by
# a personal reference project's clean SaaS aesthetic -- off-white surfaces,
# subtle borders, generous radius -- with the accent swapped to blue/indigo
# so it never collides with the risk-severity red/amber/green used below.
BASE_CSS = """
:root {
  --bg: #f7f8fa;
  --surface: #ffffff;
  --surface-2: #f2f4f7;
  --border: #e4e7ec;
  --border-strong: #d0d5dd;
  --text: #101828;
  --text-muted: #667085;
  --text-faint: #98a2b3;
  --primary: #2952cc;
  --primary-hover: #1f3fa3;
  --primary-tint: #eaf0ff;
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --shadow-sm: 0 1px 2px rgba(16, 24, 40, 0.05);
  --shadow-md: 0 4px 16px rgba(16, 24, 40, 0.08);

  --low-text: #067647; --low-bg: #ecfdf3; --low-border: #abefc6;
  --medium-text: #b54708; --medium-bg: #fffaeb; --medium-border: #fedf89;
  --high-text: #b42318; --high-bg: #fef3f2; --high-border: #fecdca;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0b0d12;
    --surface: #14171f;
    --surface-2: #1a1e28;
    --border: #262b36;
    --border-strong: #333a48;
    --text: #edeef1;
    --text-muted: #9aa1b0;
    --text-faint: #6b7280;
    --primary: #6d8dea;
    --primary-hover: #8ba4ef;
    --primary-tint: #1a2440;
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.45);

    --low-text: #4ade80; --low-bg: #0f2a1c; --low-border: #1e4d33;
    --medium-text: #fbbf24; --medium-bg: #2e2308; --medium-border: #57430f;
    --high-text: #f87171; --high-bg: #2e1414; --high-border: #5c2323;
  }
}

* { box-sizing: border-box; }

html, body { height: 100%; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", ui-sans-serif,
    system-ui, Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}

a { color: var(--primary); text-decoration: none; }
a:hover { text-decoration: underline; }

h1, h2, h3 { line-height: 1.25; letter-spacing: -0.01em; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 1.75rem 0 .75rem; color: var(--text); }
h3 { font-size: .95rem; margin: 0; }

code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .85em;
}

.topbar {
  position: sticky; top: 0; z-index: 10;
  display: flex; align-items: center; gap: 1.25rem;
  padding: .85rem 1.5rem;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}
.topbar .brand {
  font-weight: 700; font-size: 1.05rem; color: var(--text);
  display: flex; align-items: center; gap: .4rem;
}
.topbar .brand:hover { text-decoration: none; }
.topbar .brand .dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--primary); display: inline-block;
}
.topbar .tabs { display: flex; gap: .25rem; margin-left: auto; }
.topbar .tabs a {
  padding: .4rem .75rem; border-radius: var(--radius-sm);
  color: var(--text-muted); font-size: .88rem; font-weight: 500;
}
.topbar .tabs a:hover { background: var(--surface-2); text-decoration: none; }
.topbar .tabs a.active { background: var(--primary-tint); color: var(--primary); }
.topbar .subject {
  color: var(--text-muted); font-size: .88rem;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 32ch;
}

.container { max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  padding: 1.25rem 1.5rem;
}

.lede { color: var(--text-muted); margin: 0 0 1.5rem; }

.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: .4rem;
  height: 38px; padding: 0 1rem; border-radius: var(--radius-sm);
  font-size: .88rem; font-weight: 600; cursor: pointer;
  border: 1px solid var(--border-strong); background: var(--surface); color: var(--text);
  transition: transform .1s ease, background .15s ease, border-color .15s ease;
}
.btn:hover { background: var(--surface-2); text-decoration: none; }
.btn:active { transform: scale(.97); }
.btn-primary {
  background: var(--primary); border-color: var(--primary); color: #fff;
}
.btn-primary:hover { background: var(--primary-hover); border-color: var(--primary-hover); }

.dropzone {
  display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
  border: 1.5px dashed var(--border-strong); border-radius: var(--radius-lg);
  padding: 1.25rem 1.5rem; background: var(--surface-2);
}
.dropzone input[type=file] { flex: 1; min-width: 220px; font-size: .88rem; }

table.data { border-collapse: collapse; width: 100%; margin-top: .5rem; }
table.data th {
  text-align: left; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em;
  color: var(--text-faint); font-weight: 600; padding: .5rem .75rem; border-bottom: 1px solid var(--border);
}
table.data td {
  padding: .7rem .75rem; border-bottom: 1px solid var(--border); font-size: .9rem; vertical-align: middle;
}
table.data tr:last-child td { border-bottom: none; }
table.data tbody tr:hover { background: var(--surface-2); }
table.data .actions { display: flex; gap: .35rem; }
table.data .actions a {
  font-size: .8rem; font-weight: 600; padding: .3rem .6rem; border-radius: var(--radius-sm);
  color: var(--text-muted); border: 1px solid var(--border);
}
table.data .actions a:hover { background: var(--primary-tint); color: var(--primary); border-color: var(--primary-tint); text-decoration: none; }

.empty-state {
  text-align: center; color: var(--text-muted); padding: 2.5rem 1rem;
}

.badge {
  display: inline-flex; align-items: center; gap: .3rem;
  padding: .18rem .6rem; border-radius: 999px;
  font-size: .76rem; font-weight: 600; border: 1px solid transparent; white-space: nowrap;
}
.badge-low { color: var(--low-text); background: var(--low-bg); border-color: var(--low-border); }
.badge-medium { color: var(--medium-text); background: var(--medium-bg); border-color: var(--medium-border); }
.badge-high { color: var(--high-text); background: var(--high-bg); border-color: var(--high-border); }
.badge-neutral { color: var(--text-muted); background: var(--surface-2); border-color: var(--border); }

.tag-0 { color: #2952cc; background: #eaf0ff; }
.tag-1 { color: #c11574; background: #fdf2fa; }
.tag-2 { color: #067647; background: #ecfdf3; }
.tag-3 { color: #b54708; background: #fffaeb; }
.tag-4 { color: #5925dc; background: #f4f3ff; }
.tag-5 { color: #b42318; background: #fef3f2; }
.tag-6 { color: #026aa2; background: #f0f9ff; }
.tag-7 { color: #93370d; background: #fdf4e7; }

.clause-card {
  border: 1px solid var(--border); border-radius: var(--radius-md);
  padding: 1rem 1.25rem; margin: 0 0 .9rem; background: var(--surface);
}
.clause-card .clause-head {
  display: flex; align-items: center; justify-content: space-between; gap: .75rem; flex-wrap: wrap;
  margin-bottom: .4rem;
}
.clause-card .offsets { color: var(--text-faint); font-size: .78rem; margin: 0 0 .5rem; }
.clause-card pre {
  white-space: pre-wrap; word-wrap: break-word; margin: 0;
  background: var(--surface-2); border-radius: var(--radius-sm); padding: .7rem .85rem;
  font-family: inherit; font-size: .88rem; color: var(--text);
}

.callout {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 1rem 1.25rem; color: var(--text);
}

.finding {
  display: flex; gap: .6rem; align-items: flex-start;
  border: 1px solid var(--border); border-radius: var(--radius-md);
  padding: .8rem 1rem; margin-bottom: .6rem; background: var(--surface);
}
.finding .rationale { color: var(--text); font-size: .9rem; }
.finding .category { color: var(--text-faint); font-size: .78rem; }

.chat-log {
  border: 1px solid var(--border); border-radius: var(--radius-lg);
  padding: 1.25rem; min-height: 14rem; max-height: 55vh; overflow-y: auto;
  background: var(--surface); display: flex; flex-direction: column; gap: .9rem;
}
.chat-log:empty::before {
  content: "Ask a question about this contract to get a cited answer.";
  color: var(--text-faint); font-size: .9rem;
}
.msg { display: flex; }
.msg.you { justify-content: flex-end; }
.msg.bot { justify-content: flex-start; }
.bubble { max-width: 78%; border-radius: var(--radius-md); padding: .65rem .9rem; font-size: .92rem; }
.msg.you .bubble { background: var(--primary); color: #fff; border-bottom-right-radius: 4px; }
.msg.bot .bubble { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
.bubble pre {
  white-space: pre-wrap; word-wrap: break-word; margin: 0; font-family: inherit; font-size: inherit;
}
.cites { margin-top: .5rem; font-size: .78rem; color: var(--text-muted); line-height: 1.5; }

.chat-form { display: flex; gap: .6rem; margin-top: 1rem; }
.chat-form input[type=text] {
  flex: 1; padding: .7rem .9rem; border-radius: var(--radius-md);
  border: 1px solid var(--border-strong); font-size: .92rem; background: var(--surface); color: var(--text);
}
.chat-form input[type=text]:focus { outline: 2px solid var(--primary); outline-offset: 1px; }

@media (max-width: 640px) {
  .container { padding: 1.25rem 1rem 3rem; }
  .topbar { padding: .7rem 1rem; flex-wrap: wrap; }
  .topbar .subject { max-width: 100%; order: 3; }
  table.data th:nth-child(1), table.data td:nth-child(1) { display: none; }
}
"""

_TAG_PALETTE_SIZE = 8


def tag_class(name: str) -> str:
    """Deterministic tag-color class for a category name (stable across runs)."""
    index = sum(ord(ch) for ch in name) % _TAG_PALETTE_SIZE
    return f"tag-{index}"


def severity_class(severity: str) -> str:
    """Map a low/medium/high severity string onto its badge class (default neutral)."""
    return {"low": "badge-low", "medium": "badge-medium", "high": "badge-high"}.get(
        severity.lower(), "badge-neutral"
    )


def nav_bar(
    active: str = "",
    contract_id: str | None = None,
    contract_label: str | None = None,
) -> str:
    """Top bar: brand mark, optional contract-scoped tabs, optional subject label.

    ``active`` is one of "" (dashboard), "view", "chat", "report", or
    "compare" and controls which tab (if any) is highlighted. "Compare" is
    always shown, on every page -- it isn't scoped to one contract (it takes
    two fresh uploads), unlike the other three tabs. Escaping of
    ``contract_label`` is the caller's responsibility (it's already
    ``html.escape``-d contract data by the time it reaches here in every
    current call site).
    """
    def tab(key: str, label: str, href: str) -> str:
        cls = "active" if active == key else ""
        return f"<a class='{cls}' href='{href}'>{label}</a>"

    tabs = []
    if contract_id is not None:
        tabs.append(tab("view", "Clauses", f"/contracts/{contract_id}/view"))
        tabs.append(tab("chat", "Chat", f"/contracts/{contract_id}/chat"))
        tabs.append(tab("report", "Report", f"/contracts/{contract_id}/report"))
    tabs.append(tab("compare", "Compare", "/compare"))
    tabs_html = "<nav class='tabs'>" + "".join(tabs) + "</nav>"

    subject_html = (
        f"<span class='subject'>{contract_label}</span>" if contract_label else ""
    )

    return (
        "<div class='topbar'>"
        "<a class='brand' href='/'><span class='dot'></span>ContractLens</a>"
        f"{subject_html}"
        f"{tabs_html}"
        "</div>"
    )


def page(
    title: str,
    content: str,
    active: str = "",
    contract_id: str | None = None,
    contract_label: str | None = None,
) -> str:
    """Wrap ``content`` (already-assembled, already-escaped HTML) in the shared shell."""
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{title}</title>
<style>{BASE_CSS}</style></head><body>
{nav_bar(active, contract_id, contract_label)}
<div class='container'>
{content}
</div>
</body></html>"""
