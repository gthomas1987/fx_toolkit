"""Project tracker data layer.

Backed by a single CSV file at `data/projects.csv`. The page goes
through `load_projects`, `add_project`, `update_project`,
`delete_project`, `move_status` — never touches the file directly.

Writes are atomic (tempfile in same dir + `os.replace`), so a
concurrent reader can never see a half-written file. Reads always
hit disk; fine at the expected scale (single user, dozens of rows).
If the file ever grows past ~1k rows or you start running multiple
concurrent writers, swap the storage layer to SQLite — every
function signature below is intentionally storage-agnostic so the
migration is a one-day job.

Conventions
-----------
- Dates stored as YYYY-MM-DD strings in CSV, parsed to pd.Timestamp
  on load. Hours and progress are floats. status/priority/type are
  constrained to the constants below.
- `id` is a 12-char uuid4 hex slice. Never re-used even after delete.
- `updated_at` is bumped on every write so the "recently changed"
  view actually means something.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd


DATA_PATH = Path("data/projects.csv")

STATUSES = ["Backlog", "In Progress", "Blocked", "Review", "Done"]
PRIORITIES = ["Urgent", "High", "Med", "Low"]
TYPES = ["Pricer", "Backtest", "Research", "Ad-hoc", "Data", "Other"]

PRIORITY_EMOJI = {"Urgent": "🔴", "High": "🟠", "Med": "🟡", "Low": "🟢"}
TYPE_EMOJI = {
    "Pricer": "💰", "Backtest": "📈", "Research": "🔬",
    "Ad-hoc": "⚡", "Data": "📊", "Other": "📌",
}
PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITIES)}

COLUMNS = [
    "id", "title", "description", "type", "priority", "status",
    "date_requested", "due_date", "est_hours", "actual_hours",
    "progress_pct", "notes", "related_page", "created_at", "updated_at",
]

DATE_COLS = ["date_requested", "due_date", "created_at", "updated_at"]
NUM_COLS = ["est_hours", "actual_hours", "progress_pct"]


def _empty_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=COLUMNS)
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c])
    return df


def load_projects() -> pd.DataFrame:
    """Load all projects. Returns empty frame if the CSV is missing."""
    if not DATA_PATH.exists():
        return _empty_df()
    df = pd.read_csv(DATA_PATH)
    # Forward-compat: tolerate older files missing newer columns.
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[COLUMNS]
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["title"] = df["title"].fillna("Untitled").astype(str)
    for c in ["description", "notes", "related_page"]:
        df[c] = df[c].fillna("").astype(str)
    return df


def save_projects(df: pd.DataFrame) -> None:
    """Atomically replace the CSV with the given DataFrame."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in DATE_COLS:
        out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
    fd, tmp = tempfile.mkstemp(dir=str(DATA_PATH.parent), suffix=".csv.tmp")
    try:
        os.close(fd)
        out.to_csv(tmp, index=False)
        os.replace(tmp, DATA_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def add_project(
    title: str,
    description: str = "",
    type_: str = "Ad-hoc",
    priority: str = "Med",
    status: str = "Backlog",
    date_requested: Optional[date] = None,
    due_date: Optional[date] = None,
    est_hours: float = 0.0,
    actual_hours: float = 0.0,
    progress_pct: float = 0.0,
    notes: str = "",
    related_page: str = "",
) -> str:
    """Create a project. Returns the new id."""
    now = datetime.now()
    new_id = uuid.uuid4().hex[:12]
    row = {
        "id": new_id,
        "title": (title or "").strip() or "Untitled",
        "description": description or "",
        "type": type_ if type_ in TYPES else "Other",
        "priority": priority if priority in PRIORITIES else "Med",
        "status": status if status in STATUSES else "Backlog",
        "date_requested": pd.Timestamp(date_requested or now.date()),
        "due_date": pd.Timestamp(due_date) if due_date else pd.NaT,
        "est_hours": float(est_hours or 0),
        "actual_hours": float(actual_hours or 0),
        "progress_pct": float(progress_pct or 0),
        "notes": notes or "",
        "related_page": related_page or "",
        "created_at": pd.Timestamp(now),
        "updated_at": pd.Timestamp(now),
    }
    df = load_projects()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_projects(df)
    return new_id


def update_project(project_id: str, **changes) -> None:
    """Update fields of a project in-place. Unknown fields are ignored."""
    df = load_projects()
    mask = df["id"] == project_id
    if not mask.any():
        return
    idx = df.index[mask][0]
    for k, v in changes.items():
        if k in COLUMNS and k != "id":
            df.at[idx, k] = v
    df.at[idx, "updated_at"] = pd.Timestamp(datetime.now())
    save_projects(df)


def delete_project(project_id: str) -> None:
    """Remove a project entirely. No-op if id not found."""
    df = load_projects()
    df = df[df["id"] != project_id].reset_index(drop=True)
    save_projects(df)


def move_status(project_id: str, direction: int) -> None:
    """Shift a project one status left (-1) or right (+1). Clamps at ends."""
    df = load_projects()
    mask = df["id"] == project_id
    if not mask.any():
        return
    idx = df.index[mask][0]
    cur = df.at[idx, "status"]
    if cur not in STATUSES:
        cur = STATUSES[0]
    new_pos = min(max(STATUSES.index(cur) + direction, 0), len(STATUSES) - 1)
    df.at[idx, "status"] = STATUSES[new_pos]
    df.at[idx, "updated_at"] = pd.Timestamp(datetime.now())
    save_projects(df)


def summary_metrics(df: pd.DataFrame) -> dict:
    """Headline metrics for the page strip."""
    today = pd.Timestamp(datetime.now().date())
    in_progress = df[df["status"] == "In Progress"]
    backlog = df[df["status"] == "Backlog"]
    blocked = df[df["status"] == "Blocked"]
    overdue = df[
        df["due_date"].notna()
        & (df["due_date"] < today)
        & (df["status"] != "Done")
    ]
    not_done = df[df["status"] != "Done"]
    return {
        "in_progress": int(len(in_progress)),
        "backlog": int(len(backlog)),
        "blocked": int(len(blocked)),
        "overdue": int(len(overdue)),
        "hours_actual": float(not_done["actual_hours"].fillna(0).sum()),
        "hours_est": float(not_done["est_hours"].fillna(0).sum()),
    }
