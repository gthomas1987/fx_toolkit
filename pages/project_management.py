"""Project management page — two views, switchable at top.

Backed by `core.projects_store` (CSV under `data/`). Five statuses:
Backlog / In Progress / Blocked / Review / Done.

Views
-----
- **Table** (default): per-row layout with inline action buttons.
  Each row has priority ▲/▼ controls, ✏️ Edit (opens the dialog),
  and 📧 Chase (records a chase event — bumps chase_count and
  last_chased_at). A small "Last: 2d ago" caption appears under
  the action buttons when the project has been chased. Sort
  selector above the table picks #/Priority/Status/Type/Title/
  Last-chase with an ascending/descending toggle.
- **Kanban**: 5-column board, cards sorted by priority then due
  within each column, with per-card ◀ / ✏️ / ▶ buttons.

Both views share: page header + metric strip, filter row, the
new-project and edit dialogs, and the footer raw-data expander.
Switching views does NOT touch the CSV; it only changes layout.
"""

from datetime import date, datetime
from html import escape as _h

import pandas as pd
import streamlit as st

from core.projects_store import (
    PRIORITIES,
    PRIORITY_EMOJI,
    PRIORITY_RANK,
    STATUSES,
    TYPE_EMOJI,
    TYPES,
    add_project,
    change_priority,
    chase_project,
    delete_project,
    load_projects,
    move_status,
    summary_metrics,
    update_project,
)


# ---------------------------------------------------------------------------
# Styling — kanban-card CSS (only used by the Kanban view, but injected
# unconditionally because Streamlit doesn't conditionally remove CSS).
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
.kcard {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 4px;
}
.kcard.overdue { border-left: 3px solid #ff4d4d; }
.kcard .ktitle  { font-weight: 600; margin-bottom: 4px; line-height: 1.25; font-size: 0.95em; }
.kcard .kmeta   { font-size: 0.78em; color: rgba(255,255,255,0.65); margin-bottom: 2px; }
.kcard .knotes  { font-size: 0.76em; color: rgba(255,255,255,0.5);
                  margin-top: 4px; max-height: 3em; overflow: hidden;
                  font-style: italic; }
.kcard .khours  { font-size: 0.74em; color: rgba(255,255,255,0.55); margin-top: 3px; }
.kbadge {
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-size: 0.72em; background: rgba(255,255,255,0.08);
    margin-right: 4px;
}
.kdue { font-size: 0.78em; }
.kdue.late { color: #ff6b6b; font-weight: 600; }
.kcol-header {
    text-align: center; font-weight: 600; padding: 6px 8px;
    background: rgba(255,255,255,0.04); border-radius: 6px;
    margin-bottom: 8px; font-size: 0.9em;
}
.kempty { opacity: 0.4; text-align: center; padding: 16px; font-size: 0.85em; }
/* Tighten the spacing of the three action buttons under each card */
div[data-testid="column"] div[data-testid="stHorizontalBlock"] button { padding: 0 4px; }

/* ---- HTML project table (Table view) ---- */
.proj-table {
    width: 100%;
    border-collapse: collapse;
    color: #e2e8f0;
    font-size: 0.88em;
    margin: 8px 0 12px 0;
    table-layout: fixed;            /* respect column widths even with long content */
}
.proj-table th, .proj-table td {
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    vertical-align: top;
    overflow-wrap: break-word;
    word-break: break-word;
}
.proj-table th {
    background: rgba(255,255,255,0.04);
    font-weight: 600;
    font-size: 0.78em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #94a3b8;
    border-bottom: 1px solid rgba(255,255,255,0.15);
}
.proj-table tbody tr:hover { background: rgba(255,255,255,0.025); }
/* Column widths — Title and Description take the lion's share so they
   wrap rather than truncate. Notes is narrower because it's typically
   short. The number column is fixed pixels because "#12" doesn't need
   percentage scaling. */
.proj-table col.c-num       { width: 48px; }
.proj-table col.c-priority  { width: 9%; }
.proj-table col.c-title     { width: 28%; }
.proj-table col.c-status    { width: 8%; }
.proj-table col.c-type      { width: 9%; }
.proj-table col.c-progress  { width: 9%; }
.proj-table col.c-desc      { width: 25%; }
.proj-table col.c-notes     { width: 12%; }
.proj-table td.c-num {
    color: #64748b;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}
.proj-table td.c-title  { line-height: 1.4; }
.proj-table td.c-desc, .proj-table td.c-notes {
    line-height: 1.4;
    color: #cbd5e1;
    font-size: 0.94em;
}
/* Progress bar inside a cell */
.proj-table .pbar-bg {
    background: rgba(148, 163, 184, 0.22);
    height: 8px;
    border-radius: 4px;
    width: 100%;
    overflow: hidden;
    margin: 4px 0 2px 0;
}
.proj-table .pbar-fill {
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    height: 100%;
    border-radius: 4px;
}
/* ---- Inline-row buttons (Table view per-row controls) ---- */
/* Compact buttons so the priority ▲/▼, edit, and chase controls
   slot into table cells without dwarfing the row height. The
   `proj-row-buttons` wrapper class is applied to a parent st.container
   in render_table_view so these styles only target the table row
   buttons, not buttons elsewhere on the page. */
.proj-row-buttons div[data-testid="stButton"] > button {
    padding: 2px 8px !important;
    min-height: 28px !important;
    height: 28px !important;
    font-size: 0.85em !important;
    line-height: 1 !important;
}
.proj-row-buttons div[data-testid="stButton"] {
    margin-bottom: 0 !important;
}
.proj-row {
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 8px 4px 8px 4px;
}
.proj-row-header {
    background: rgba(255,255,255,0.04);
    border-bottom: 1px solid rgba(255,255,255,0.15);
    padding: 6px 4px;
    font-weight: 600;
    font-size: 0.78em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #94a3b8;
}
.proj-cell {
    line-height: 1.4;
    word-break: break-word;
    overflow-wrap: break-word;
    font-size: 0.9em;
}
.proj-cell-title { color: #e2e8f0; font-weight: 500; }
.proj-cell-muted { color: #94a3b8; }
.proj-cell-desc, .proj-cell-notes { color: #cbd5e1; font-size: 0.88em; }
.proj-num { color: #64748b; font-weight: 600; font-variant-numeric: tabular-nums; }
.proj-priority-label { display: inline-block; padding: 2px 0; font-size: 0.9em; }
.proj-pbar-bg {
    background: rgba(148, 163, 184, 0.22);
    height: 8px; border-radius: 4px; width: 100%;
    margin: 4px 0 2px 0; overflow: hidden;
}
.proj-pbar-fill {
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    height: 100%; border-radius: 4px;
}
.proj-chase-meta {
    font-size: 0.72em;
    color: #94a3b8;
    margin-top: 4px;
    font-style: italic;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header + metric strip
# ---------------------------------------------------------------------------

st.title("📋 Project Management")
st.caption("Track work given by the PM. CSV-backed at `data/projects.csv`.")

df = load_projects()
metrics = summary_metrics(df)

m = st.columns(5)
m[0].metric("In Progress", metrics["in_progress"])
m[1].metric("Backlog", metrics["backlog"])
m[2].metric("Blocked", metrics["blocked"])
m[3].metric("Overdue", metrics["overdue"])
m[4].metric(
    "Hours (active)",
    f"{metrics['hours_actual']:.0f} / {metrics['hours_est']:.0f}",
    help="Actual / estimated hours, across non-Done projects",
)

st.divider()


# ---------------------------------------------------------------------------
# Dialogs — new project, edit project. Defined here so both views can
# trigger them. Streamlit's @st.dialog decorator makes these modal.
# ---------------------------------------------------------------------------

@st.dialog("New project", width="large")
def new_project_dialog() -> None:
    title = st.text_input("Title*")
    description = st.text_area(
        "Description", height=80,
        placeholder="What is the PM asking for?",
    )
    c1, c2 = st.columns(2)
    type_ = c1.selectbox("Type", TYPES, index=TYPES.index("Ad-hoc"))
    priority = c2.selectbox("Priority", PRIORITIES,
                            index=PRIORITIES.index("Med"))
    c3, c4 = st.columns(2)
    date_req = c3.date_input("Requested", value=date.today())
    due = c4.date_input("Due", value=None)
    c5, c6 = st.columns(2)
    est = c5.number_input("Est hours", min_value=0.0, value=0.0, step=0.5)
    related = c6.text_input(
        "Related page",
        placeholder="e.g. Vol Dashboard / Option Pricer",
    )
    notes = st.text_area("Notes", height=80)

    if st.button("Create", type="primary", use_container_width=True):
        if not title.strip():
            st.error("Title is required.")
            return
        add_project(
            title=title,
            description=description,
            type_=type_,
            priority=priority,
            date_requested=date_req,
            due_date=due if due else None,
            est_hours=est,
            related_page=related,
            notes=notes,
        )
        st.rerun()


@st.dialog("Edit project", width="large")
def edit_project_dialog(project: dict) -> None:
    title = st.text_input("Title*", value=project["title"])
    description = st.text_area(
        "Description", value=project.get("description") or "", height=80,
    )
    c1, c2, c3 = st.columns(3)
    type_ = c1.selectbox(
        "Type", TYPES,
        index=TYPES.index(project["type"])
        if project["type"] in TYPES else 0,
    )
    priority = c2.selectbox(
        "Priority", PRIORITIES,
        index=PRIORITIES.index(project["priority"])
        if project["priority"] in PRIORITIES else 2,
    )
    status = c3.selectbox(
        "Status", STATUSES,
        index=STATUSES.index(project["status"])
        if project["status"] in STATUSES else 0,
    )
    c4, c5 = st.columns(2)
    date_req = c4.date_input(
        "Requested",
        value=project["date_requested"].date()
        if pd.notna(project["date_requested"]) else date.today(),
    )
    due = c5.date_input(
        "Due",
        value=project["due_date"].date()
        if pd.notna(project["due_date"]) else None,
    )
    c6, c7, c8 = st.columns(3)
    est = c6.number_input(
        "Est hours", min_value=0.0,
        value=float(project["est_hours"] or 0), step=0.5,
    )
    actual = c7.number_input(
        "Actual hours", min_value=0.0,
        value=float(project["actual_hours"] or 0), step=0.5,
    )
    progress = c8.slider(
        "Progress %", 0, 100, int(project["progress_pct"] or 0),
    )
    related = st.text_input(
        "Related page", value=project.get("related_page") or "",
    )
    notes = st.text_area(
        "Notes", value=project.get("notes") or "", height=100,
    )

    bcol1, bcol2 = st.columns([3, 1])
    save = bcol1.button("Save", type="primary", use_container_width=True)
    delete = bcol2.button("🗑️ Delete", use_container_width=True)

    if save:
        update_project(
            project["id"],
            title=title.strip() or "Untitled",
            description=description,
            type=type_,
            priority=priority,
            status=status,
            date_requested=pd.Timestamp(date_req),
            due_date=pd.Timestamp(due) if due else pd.NaT,
            est_hours=est,
            actual_hours=actual,
            progress_pct=progress,
            related_page=related,
            notes=notes,
        )
        st.rerun()
    if delete:
        delete_project(project["id"])
        st.rerun()


# ---------------------------------------------------------------------------
# View renderers
# ---------------------------------------------------------------------------

TODAY = pd.Timestamp(datetime.now().date())

# Placeholder rendered in empty Description/Notes cells. Defined as a
# module-level constant so f-strings don't need to embed it with
# escaped quotes — Python 3.10 / 3.11 reject backslashes inside the
# expression part of an f-string (PEP 701 relaxed this in 3.12).
_EMPTY_CELL = '<span style="opacity:0.3">—</span>'


def render_table_view(view: pd.DataFrame) -> None:
    """Per-row table with inline action buttons.

    Layout uses st.columns per row so action buttons (priority ▲/▼,
    edit, chase) can be wired directly to handlers. Title, Description
    and Notes render as wrapped HTML inside their cells.

    Columns: # / Priority+▲▼ / Title / Status / Type / Progress /
    Description / Notes / Edit+Chase. The bottom edit selector is
    gone — every row has its own ✏️ button.

    Project numbers come from the CSV row index +1. Stable across
    sorts and filters within a session; can shift after a delete
    (data store re-indexes on delete) — fine tradeoff for not
    introducing a stored project_num field.

    Built per-row rather than as one HTML table because action buttons
    can't be embedded in raw HTML — Streamlit needs real widgets to
    fire callbacks. Tradeoff: more components on the page, slightly
    slower reruns. For ~17-50 rows that's still well under 500ms.
    """
    if len(view) == 0:
        st.info("No projects match the current filters.")
        return

    # --- Sort controls ------------------------------------------------
    sort_options = {
        "Project #":  "_num",
        "Priority":   "_priority_rank",
        "Status":     "status",
        "Type":       "type",
        "Title":      "title",
        "Last chase": "last_chased_at",
    }
    sc1, sc2 = st.columns([2, 3])
    sort_by = sc1.selectbox(
        "Sort by",
        list(sort_options.keys()),
        index=0,
        key="pm_sort_by",
    )
    sort_asc = sc2.radio(
        "Order",
        ["Ascending", "Descending"],
        horizontal=True,
        index=0,
        key="pm_sort_order",
    ) == "Ascending"

    # --- Build display frame -----------------------------------------
    display = view.copy()
    display["_num"] = display.index + 1
    display["_priority_rank"] = (
        display["priority"].map(PRIORITY_RANK).fillna(99)
    )
    display = display.sort_values(
        sort_options[sort_by],
        ascending=sort_asc,
        kind="stable",
        na_position="last",
    )

    # Column weights tuned for a wide layout. Title and Description are
    # the widest because they hold wrapped multi-line text; the action
    # column on the right is wide enough to hold two compact buttons
    # side by side without wrapping.
    #
    # The order, top-to-bottom, of values in _COL_WEIGHTS matches the
    # column order in both the header and every data row.
    weights = [0.4, 1.5, 3.2, 0.9, 1.0, 1.0, 2.8, 1.3, 1.4]
    header_labels = [
        "#", "Priority", "Title", "Status", "Type",
        "Progress", "Description", "Notes", "Actions",
    ]

    # --- Header row ---------------------------------------------------
    header_cols = st.columns(weights)
    for col, label in zip(header_cols, header_labels):
        col.markdown(
            f'<div class="proj-row-header">{label}</div>',
            unsafe_allow_html=True,
        )

    # --- Data rows ----------------------------------------------------
    for _, r in display.iterrows():
        # The wrapper container scopes the .proj-row-buttons CSS to
        # *this* row's buttons only, so we don't override button styles
        # elsewhere on the page (e.g. the +New project button or the
        # edit dialog's Save/Delete).
        with st.container():
            st.markdown(
                '<div class="proj-row-buttons">',
                unsafe_allow_html=True,
            )
            cols = st.columns(weights)
            proj_id = r["id"]
            num     = int(r["_num"])
            prio    = r["priority"]
            prio_emoji = PRIORITY_EMOJI.get(prio, "⚪")
            type_disp = (
                f"{TYPE_EMOJI.get(r['type'], '📌')} {_h(str(r['type']))}"
            )
            progress = max(0, min(100, int(r.get("progress_pct") or 0)))

            # 1. Project number
            cols[0].markdown(
                f'<div class="proj-cell proj-num">#{num}</div>',
                unsafe_allow_html=True,
            )

            # 2. Priority + up/down buttons. Nested columns inside the
            # priority cell give us a [▲][label][▼] layout. ▲ moves up
            # the rank (more urgent = lower index); ▼ moves down. Both
            # buttons disable at the appropriate end so the user gets
            # immediate visual feedback that they've hit the clamp.
            with cols[1]:
                pup, plab, pdn = st.columns([1, 2.4, 1])
                up_disabled = prio == PRIORITIES[0]   # already Urgent
                dn_disabled = prio == PRIORITIES[-1]  # already Low
                if pup.button("▲", key=f"prio_up_{proj_id}",
                              disabled=up_disabled,
                              help="Increase priority"):
                    change_priority(proj_id, -1)
                    st.rerun()
                plab.markdown(
                    f'<div class="proj-priority-label">'
                    f'{prio_emoji} {_h(str(prio))}</div>',
                    unsafe_allow_html=True,
                )
                if pdn.button("▼", key=f"prio_dn_{proj_id}",
                              disabled=dn_disabled,
                              help="Decrease priority"):
                    change_priority(proj_id, +1)
                    st.rerun()

            # 3. Title — wrapped HTML
            cols[2].markdown(
                f'<div class="proj-cell proj-cell-title">'
                f'{_h(str(r["title"] or ""))}</div>',
                unsafe_allow_html=True,
            )

            # 4. Status
            cols[3].markdown(
                f'<div class="proj-cell">{_h(str(r["status"]))}</div>',
                unsafe_allow_html=True,
            )

            # 5. Type
            cols[4].markdown(
                f'<div class="proj-cell">{type_disp}</div>',
                unsafe_allow_html=True,
            )

            # 6. Progress bar
            cols[5].markdown(
                f'<div class="proj-cell">'
                f'<div class="proj-pbar-bg">'
                f'<div class="proj-pbar-fill" '
                f'style="width:{progress}%;"></div></div>'
                f'<div class="proj-cell-muted" style="font-size:0.8em;">'
                f'{progress}%</div></div>',
                unsafe_allow_html=True,
            )

            # 7. Description — wrapped
            desc = _h(str(r.get("description") or ""))
            cols[6].markdown(
                f'<div class="proj-cell proj-cell-desc">'
                f'{desc or _EMPTY_CELL}</div>',
                unsafe_allow_html=True,
            )

            # 8. Notes — wrapped
            notes = _h(str(r.get("notes") or ""))
            cols[7].markdown(
                f'<div class="proj-cell proj-cell-notes">'
                f'{notes or _EMPTY_CELL}</div>',
                unsafe_allow_html=True,
            )

            # 9. Actions — Edit + Chase, plus a "last chased Nd ago"
            # caption underneath when last_chased_at is set. The
            # caption lives in the same cell so the row's vertical
            # alignment stays clean.
            with cols[8]:
                ebtn, cbtn = st.columns(2)
                if ebtn.button(
                    "✏️",
                    key=f"edit_{proj_id}",
                    help="Edit project",
                    use_container_width=True,
                ):
                    edit_project_dialog(r.to_dict())
                if cbtn.button(
                    "📧",
                    key=f"chase_{proj_id}",
                    help="Chase / request update",
                    use_container_width=True,
                ):
                    chase_project(proj_id)
                    title_short = str(r["title"])[:40]
                    st.toast(f"Chased: {title_short}", icon="📧")
                    st.rerun()

                # Chase indicator
                last_chase = r.get("last_chased_at")
                chase_count = r.get("chase_count")
                if pd.notna(last_chase):
                    days = (TODAY - pd.Timestamp(last_chase).normalize()).days
                    when = (
                        "today" if days == 0
                        else "yesterday" if days == 1
                        else f"{days}d ago"
                    )
                    count_str = (
                        f" ·×{int(chase_count)}"
                        if pd.notna(chase_count) and int(chase_count) > 1
                        else ""
                    )
                    st.markdown(
                        f'<div class="proj-chase-meta">'
                        f'Last: {when}{count_str}</div>',
                        unsafe_allow_html=True,
                    )

            # Close .proj-row-buttons wrapper opened above
            st.markdown("</div>", unsafe_allow_html=True)
            # Thin separator between rows
            st.markdown(
                '<div style="border-bottom:1px solid rgba(255,255,255,0.06);'
                'margin: 2px 0 6px 0;"></div>',
                unsafe_allow_html=True,
            )


def render_kanban_view(view: pd.DataFrame) -> None:
    """Five-column board, cards sorted by priority then due within each."""

    def _due_html(due_ts) -> str:
        if pd.isna(due_ts):
            return ""
        days = (due_ts - TODAY).days
        due_str = due_ts.strftime("%b %d")
        if days < 0:
            return (f'<span class="kdue late">'
                    f'⚠ Due {due_str} ({-days}d late)</span>')
        if days == 0:
            return '<span class="kdue late">📅 Due today</span>'
        if days <= 3:
            return f'<span class="kdue late">📅 {due_str} ({days}d)</span>'
        return f'<span class="kdue">📅 {due_str} ({days}d)</span>'

    def _hours_html(est, actual) -> str:
        est = float(est or 0)
        actual = float(actual or 0)
        if est > 0:
            pct = min(999, actual / est * 100)
            warn = " ⚠" if pct > 100 else ""
            return (f'<div class="khours">⏱ {actual:.1f} / {est:.1f} h '
                    f'({pct:.0f}%){warn}</div>')
        if actual > 0:
            return f'<div class="khours">⏱ {actual:.1f} h logged</div>'
        return ""

    def render_card(project_row: pd.Series) -> None:
        p = project_row.to_dict()
        overdue = (
            pd.notna(p["due_date"])
            and p["due_date"] < TODAY
            and p["status"] != "Done"
        )
        type_tag = f"{TYPE_EMOJI.get(p['type'], '📌')} {p['type']}"
        related_html = (
            f' &middot; <span style="opacity:0.5">→ {p["related_page"]}</span>'
            if p.get("related_page") else ""
        )
        notes_preview = (p.get("notes") or "").strip().replace("\n", " ")
        if len(notes_preview) > 90:
            notes_preview = notes_preview[:90] + "…"
        notes_html = (
            f'<div class="knotes">{notes_preview}</div>'
            if notes_preview else ""
        )

        st.markdown(
            f"""
<div class="kcard{' overdue' if overdue else ''}">
    <div class="ktitle">{PRIORITY_EMOJI.get(p['priority'], '⚪')} {p['title']}</div>
    <div class="kmeta">
        <span class="kbadge">{type_tag}</span>{related_html}
    </div>
    <div class="kmeta">{_due_html(p['due_date'])}</div>
    {_hours_html(p['est_hours'], p['actual_hours'])}
    {notes_html}
</div>
""",
            unsafe_allow_html=True,
        )

        is_first = p["status"] == STATUSES[0]
        is_last = p["status"] == STATUSES[-1]
        b1, b2, b3 = st.columns(3)
        if b1.button("◀", key=f"L-{p['id']}", disabled=is_first,
                     use_container_width=True, help="Move left"):
            move_status(p["id"], -1)
            st.rerun()
        if b2.button("✏️", key=f"E-{p['id']}",
                     use_container_width=True, help="Edit"):
            edit_project_dialog(p)
        if b3.button("▶", key=f"R-{p['id']}", disabled=is_last,
                     use_container_width=True, help="Move right"):
            move_status(p["id"], 1)
            st.rerun()
        st.write("")  # spacer between cards

    def _sort_bucket(bucket: pd.DataFrame) -> pd.DataFrame:
        b = bucket.copy()
        b["_p"] = b["priority"].map(PRIORITY_RANK).fillna(99)
        b["_d"] = b["due_date"].fillna(pd.Timestamp.max)
        return b.sort_values(["_p", "_d"]).drop(columns=["_p", "_d"])

    cols = st.columns(len(STATUSES))
    for i, status in enumerate(STATUSES):
        with cols[i]:
            bucket = _sort_bucket(view[view["status"] == status])
            st.markdown(
                f'<div class="kcol-header">{status} '
                f'<span style="opacity:0.55">({len(bucket)})</span></div>',
                unsafe_allow_html=True,
            )
            if len(bucket) == 0:
                st.markdown('<div class="kempty">— empty —</div>',
                            unsafe_allow_html=True)
            else:
                for _, row in bucket.iterrows():
                    render_card(row)


# ---------------------------------------------------------------------------
# Filter row — shared by both views. Stays above the view toggle so it
# clearly applies to whichever view is currently selected.
# ---------------------------------------------------------------------------

fc1, fc2, fc3, fc4 = st.columns([2.5, 1.2, 1.2, 1.1])
search = fc1.text_input(
    "Search", placeholder="🔍 Search title, description, notes…",
    label_visibility="collapsed",
)
type_filter = fc2.multiselect(
    "Type filter", TYPES, label_visibility="collapsed",
    placeholder="Any type",
)
priority_filter = fc3.multiselect(
    "Priority filter", PRIORITIES, label_visibility="collapsed",
    placeholder="Any priority",
)
if fc4.button("➕ New project", type="primary", use_container_width=True):
    new_project_dialog()

view = df.copy()
if search:
    s = search.lower()
    view = view[
        view["title"].str.lower().str.contains(s, na=False)
        | view["notes"].str.lower().str.contains(s, na=False)
        | view["description"].str.lower().str.contains(s, na=False)
    ]
if type_filter:
    view = view[view["type"].isin(type_filter)]
if priority_filter:
    view = view[view["priority"].isin(priority_filter)]


# ---------------------------------------------------------------------------
# View toggle — Table is the default. The choice persists across reruns
# via the `pm_view_mode` session-state key.
# ---------------------------------------------------------------------------

tc1, _tc2 = st.columns([1.5, 5])
with tc1:
    view_mode = st.radio(
        "View",
        ["Table", "Kanban"],
        horizontal=True,
        index=0,
        key="pm_view_mode",
        label_visibility="collapsed",
    )

st.divider()

if view_mode == "Table":
    render_table_view(view)
else:
    render_kanban_view(view)


# ---------------------------------------------------------------------------
# Footer — full raw data + CSV download. Lives outside the view toggle
# so users can always pull the underlying file regardless of view.
# ---------------------------------------------------------------------------

with st.expander("📄 Raw data & export"):
    if len(df) == 0:
        st.info("No projects yet. Click **➕ New project** above to add one.")
    else:
        display_cols = [
            "title", "type", "priority", "status", "date_requested",
            "due_date", "est_hours", "actual_hours", "progress_pct",
            "related_page", "notes",
        ]
        st.dataframe(
            df[display_cols].sort_values(
                ["status", "priority", "due_date"]
            ),
            use_container_width=True,
            hide_index=True,
        )
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV", csv_bytes,
            file_name="projects.csv", mime="text/csv",
        )
