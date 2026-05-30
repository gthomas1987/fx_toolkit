"""Project management board for tracking PM-assigned work.

Kanban-style page backed by `core.projects_store` (CSV under `data/`).
Five status columns — Backlog / In Progress / Blocked / Review / Done.
Cards within a column are sorted by priority then due date.

UI patterns
-----------
- "➕ New project" button in the filter row opens a Streamlit dialog
  with the create form.
- Each card shows priority emoji + title, a type badge, due date
  (red if overdue), an hours-progress line (actual / est), and a
  notes preview. Three buttons below: `◀` (move left), `✏️` (edit
  dialog), `▶` (move right).
- The edit dialog has every field plus Save and Delete buttons.

The filter row above the board (search, type, priority) only changes
which cards are visible; it does not touch the underlying CSV.
"""

from datetime import date, datetime

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
    delete_project,
    load_projects,
    move_status,
    summary_metrics,
    update_project,
)


# ---------------------------------------------------------------------------
# Styling — subtle cards that work on the existing dark theme
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
# Dialogs — new and edit
# ---------------------------------------------------------------------------

@st.dialog("New project", width="large")
def new_project_dialog():
    title = st.text_input("Title*")
    description = st.text_area("Description", height=80,
                               placeholder="What is the PM asking for?")
    c1, c2 = st.columns(2)
    type_ = c1.selectbox("Type", TYPES, index=TYPES.index("Ad-hoc"))
    priority = c2.selectbox("Priority", PRIORITIES, index=PRIORITIES.index("Med"))
    c3, c4 = st.columns(2)
    date_req = c3.date_input("Requested", value=date.today())
    due = c4.date_input("Due", value=None)
    c5, c6 = st.columns(2)
    est = c5.number_input("Est hours", min_value=0.0, value=0.0, step=0.5)
    related = c6.text_input("Related page",
                            placeholder="e.g. Vol Dashboard / Option Pricer")
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
def edit_project_dialog(project: dict):
    title = st.text_input("Title*", value=project["title"])
    description = st.text_area(
        "Description", value=project.get("description") or "", height=80,
    )
    c1, c2, c3 = st.columns(3)
    type_ = c1.selectbox(
        "Type", TYPES,
        index=TYPES.index(project["type"]) if project["type"] in TYPES else 0,
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
# Filter row
# ---------------------------------------------------------------------------

fc1, fc2, fc3, fc4 = st.columns([2.5, 1.2, 1.2, 1.1])
search = fc1.text_input(
    "Search", placeholder="🔍 Search title, description, notes…",
    label_visibility="collapsed",
)
type_filter = fc2.multiselect(
    "Type filter", TYPES, label_visibility="collapsed", placeholder="Any type",
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

st.divider()


# ---------------------------------------------------------------------------
# Kanban board
# ---------------------------------------------------------------------------

TODAY = pd.Timestamp(datetime.now().date())


def _due_html(due_ts) -> str:
    if pd.isna(due_ts):
        return ""
    days = (due_ts - TODAY).days
    due_str = due_ts.strftime("%b %d")
    if days < 0:
        return f'<span class="kdue late">⚠ Due {due_str} ({-days}d late)</span>'
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
        return f'<div class="khours">⏱ {actual:.1f} / {est:.1f} h ({pct:.0f}%){warn}</div>'
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
    notes_html = f'<div class="knotes">{notes_preview}</div>' if notes_preview else ""

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
    """Sort a status bucket by priority then by due date (earliest first)."""
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
# Footer — raw data + export
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
