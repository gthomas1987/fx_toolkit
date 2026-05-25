"""FX option date schedule with per-currency holiday calendars.

A drop-in replacement for `core.calendar.compute_option_dates` that
respects USD/JPY/EUR/GBP/CHF/CAD/AUD/NZD holiday calendars. The
schedule follows the standard FX convention:

    trade_date
    -> spot_settlement   = trade + 2 business days
    -> option_settlement = spot_settlement + tenor (calendar months)
                              + adjust forward to next business day
                              if it falls on a weekend/holiday
    -> option_expiry     = option_settlement - 2 business days

# Per-pair calendars

For a pair like USDJPY, both USD AND JPY currencies must be open on
each business day (the combined calendar). The list of calendars per
pair:

    USDJPY, USDCAD, USDCHF, ...  : US + counterpart-country
    EURUSD, EURJPY, EURGBP, ...  : ECB-TARGET2 + counterpart-country
    GBPUSD, GBPJPY                : UK + counterpart-country
    CROSS pairs (no USD)          : both currencies' national calendars

USD = NYSE/Fed calendar (via `holidays.US`)
JPY = `holidays.JP`
EUR = TARGET2 / `holidays.ECB`
GBP = `holidays.GB`
CHF = `holidays.CH`
CAD = `holidays.CA`
AUD = `holidays.AU`
NZD = `holidays.NZ`

Currencies not in this list fall back to weekdays-only.

# Why this matters

The pricing impact is small for vanilla options (~1.5% on a 1M tenor
from 1-day T difference) but it matters for matching Bloomberg quote
sheets exactly. For backtesting on long horizons holidays affect both
entry and exit dates so the impact cancels out — this is primarily a
quote-matching feature, not a pricing accuracy feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import calendar as _cal


try:
    import holidays as _holidays
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _holidays = None
    _HOLIDAYS_AVAILABLE = False


# =============================================================================
# Per-currency holiday calendar accessors
# =============================================================================
# Maps ISO currency code → holidays.HolidayBase factory.
# Each entry returns a callable: years -> HolidayBase instance.
_CCY_CAL_MAP: dict[str, "callable | None"] = {}

if _HOLIDAYS_AVAILABLE:
    _CCY_CAL_MAP = {
        "USD": lambda yrs: _holidays.US(years=yrs),
        "JPY": lambda yrs: _holidays.JP(years=yrs),
        # ECB = TARGET2 calendar — used for euro-area cash settlement
        "EUR": lambda yrs: _holidays.ECB(years=yrs),
        "GBP": lambda yrs: _holidays.GB(years=yrs),
        "CHF": lambda yrs: _holidays.CH(years=yrs),
        "CAD": lambda yrs: _holidays.CA(years=yrs),
        "AUD": lambda yrs: _holidays.AU(years=yrs),
        "NZD": lambda yrs: _holidays.NZ(years=yrs),
        "CNY": lambda yrs: _holidays.CN(years=yrs),
        "CNH": lambda yrs: _holidays.HK(years=yrs),   # offshore CNH uses HK
        "HKD": lambda yrs: _holidays.HK(years=yrs),
        "SGD": lambda yrs: _holidays.SG(years=yrs),
        "KRW": lambda yrs: _holidays.KR(years=yrs),
        "TWD": lambda yrs: _holidays.TW(years=yrs),
        "INR": lambda yrs: _holidays.IN(years=yrs),
        "THB": lambda yrs: _holidays.TH(years=yrs),
        "MXN": lambda yrs: _holidays.MX(years=yrs),
        "BRL": lambda yrs: _holidays.BR(years=yrs),
        "ZAR": lambda yrs: _holidays.ZA(years=yrs),
        "NOK": lambda yrs: _holidays.NO(years=yrs),
        "SEK": lambda yrs: _holidays.SE(years=yrs),
    }


# Tiny LRU cache so we don't rebuild a holidays.HolidayBase every call.
_HOL_CACHE: dict[tuple, set] = {}


def _get_holiday_set(ccy: str, years: range) -> set:
    """Return a set of holiday dates for a currency over a year range.

    Cached by (ccy, years_tuple) so repeated calls are fast.
    """
    key = (ccy, tuple(years))
    if key in _HOL_CACHE:
        return _HOL_CACHE[key]
    if ccy not in _CCY_CAL_MAP or _CCY_CAL_MAP[ccy] is None:
        out: set = set()
    else:
        cal = _CCY_CAL_MAP[ccy](list(years))
        out = set(cal.keys())
    _HOL_CACHE[key] = out
    return out


def _pair_calendars(pair: str) -> tuple[str, ...]:
    """Currencies whose holiday calendars must be considered for `pair`.

    For a pair like USDJPY we need BOTH USD and JPY to be open.
    """
    if len(pair) != 6:
        raise ValueError(f"Pair must be 6 chars (e.g. USDJPY), got {pair!r}")
    return (pair[:3], pair[3:])


def _is_business_day(d: date, ccys: tuple[str, ...], years: range) -> bool:
    """A date is a business day for `ccys` iff:
    1. It's Mon-Fri, AND
    2. It's not a holiday in ANY of the provided currencies.
    """
    if d.weekday() >= 5:
        return False
    for ccy in ccys:
        if d in _get_holiday_set(ccy, years):
            return False
    return True


def _add_bd(d: date, n: int, ccys: tuple[str, ...]) -> date:
    """Add n business days, accounting for the given currencies'
    holiday calendars."""
    out = d
    if n == 0:
        return out
    step = 1 if n > 0 else -1
    # Build a year range covering ±1 year around d (holidays.HolidayBase
    # caches per year, so a wide range is cheap).
    years = range(d.year - 1, d.year + 3)
    for _ in range(abs(n)):
        out += timedelta(days=step)
        while not _is_business_day(out, ccys, years):
            out += timedelta(days=step)
    return out


def _next_bd(d: date, ccys: tuple[str, ...]) -> date:
    """Roll d forward to the next business day if it's not one already."""
    years = range(d.year - 1, d.year + 3)
    out = d
    while not _is_business_day(out, ccys, years):
        out += timedelta(days=1)
    return out


def _add_calendar_months(d: date, months: int) -> date:
    """Calendar-month roll with end-of-month clamping."""
    new_month = d.month + months
    new_year = d.year + (new_month - 1) // 12
    new_month = ((new_month - 1) % 12) + 1
    last_day = _cal.monthrange(new_year, new_month)[1]
    return date(new_year, new_month, min(d.day, last_day))


def _add_tenor(d: date, tenor_label: str) -> date:
    """Add a tenor like '1W', '1M', '1Y' (calendar increment)."""
    s = tenor_label.upper().strip()
    if s.endswith("Y"):
        return _add_calendar_months(d, int(s[:-1]) * 12)
    if s.endswith("M"):
        return _add_calendar_months(d, int(s[:-1]))
    if s.endswith("W"):
        return d + timedelta(weeks=int(s[:-1]))
    if s.endswith("D"):
        return d + timedelta(days=int(s[:-1]))
    raise ValueError(f"Unrecognised tenor label: {tenor_label!r}")


# =============================================================================
# Public API
# =============================================================================
@dataclass
class FxOptionDates:
    """Result of `compute_option_dates_for_pair`. Same field names as
    `core.calendar.OptionDates` so it's a drop-in replacement.
    """
    trade_date: date
    spot_settlement: date
    option_settlement: date
    option_expiry: date
    T_years: float
    tenor_label: str
    pair: str
    calendars_used: tuple[str, ...]


def compute_option_dates_for_pair(
        trade_date: date, tenor_label: str, pair: str,
) -> FxOptionDates:
    """Compute the FX option date schedule for a specific pair, using
    that pair's combined holiday calendar.

    Parameters
    ----------
    trade_date : date
        The trade / valuation date.
    tenor_label : str
        Tenor label, e.g. '1W', '1M', '1Y'.
    pair : str
        Six-letter currency pair, e.g. 'USDJPY'. Determines which
        holiday calendars to use.

    Returns
    -------
    FxOptionDates with all key dates and T_years.

    Notes
    -----
    Bloomberg's FX option date schedule is slightly subtle:
      - spot_settlement = trade + 2 BD on the COMBINED pair calendar
      - delivery       = spot + tenor (calendar months), next-BD-roll
                          on the COMBINED pair calendar
      - expiry         = delivery - 2 BD on the **non-USD leg's**
                          calendar (for USD-pairs); for non-USD
                          crosses, the FOREIGN (first) currency's
                          calendar.

    The expiry rule reflects that the option fixes in the foreign
    market (e.g. Tokyo for USDJPY) — the USD-leg settlement can
    happen after the option expires. This matches Bloomberg's
    behaviour for the USDJPY 15-May test case where Juneteenth
    (19-Jun) is a US holiday but not a Tokyo holiday: expiry =
    18-Jun (BBG) vs 17-Jun (combined-calendar rule).

    Fallback: if `holidays` is not installed, weekday-only rolls
    (same as `core.calendar`).
    """
    if not _HOLIDAYS_AVAILABLE:
        from core.calendar import compute_option_dates
        od = compute_option_dates(trade_date, tenor_label)
        return FxOptionDates(
            trade_date=od.trade_date,
            spot_settlement=od.spot_settlement,
            option_settlement=od.option_settlement,
            option_expiry=od.option_expiry,
            T_years=od.T_years,
            tenor_label=tenor_label,
            pair=pair,
            calendars_used=("weekday_only",),
        )

    ccy_for, ccy_dom = _pair_calendars(pair)

    # Combined calendar for spot + delivery
    combined = (ccy_for, ccy_dom)
    spot_settle = _add_bd(trade_date, 2, combined)
    delivery_raw = _add_tenor(spot_settle, tenor_label)
    delivery = _next_bd(delivery_raw, combined)

    # Expiry: use the non-USD leg's calendar. If neither leg is USD,
    # use the foreign (first) currency's calendar. Rationale: the FX
    # option fixes in the foreign market, and US holidays alone
    # shouldn't prevent a non-US fix (which is what Bloomberg does).
    if ccy_for == "USD":
        expiry_cals = (ccy_dom,)
    elif ccy_dom == "USD":
        expiry_cals = (ccy_for,)
    else:
        expiry_cals = (ccy_for,)
    expiry = _add_bd(delivery, -2, expiry_cals)
    T_years = (expiry - trade_date).days / 365.0

    return FxOptionDates(
        trade_date=trade_date,
        spot_settlement=spot_settle,
        option_settlement=delivery,
        option_expiry=expiry,
        T_years=T_years,
        tenor_label=tenor_label,
        pair=pair,
        calendars_used=combined + (f"expiry_on:{','.join(expiry_cals)}",),
    )


# Convenience: keep the no-pair function for backwards compat with code
# that doesn't yet know the pair (uses weekday-only).
def compute_option_dates_no_pair(trade_date: date,
                                  tenor_label: str) -> FxOptionDates:
    """No-pair variant: weekday-only rolls. Use this when the pair
    isn't known. For BBG-exact matching, prefer
    `compute_option_dates_for_pair`."""
    from core.calendar import compute_option_dates
    od = compute_option_dates(trade_date, tenor_label)
    return FxOptionDates(
        trade_date=od.trade_date,
        spot_settlement=od.spot_settlement,
        option_settlement=od.option_settlement,
        option_expiry=od.option_expiry,
        T_years=od.T_years,
        tenor_label=tenor_label,
        pair="(unknown)",
        calendars_used=("weekday_only",),
    )
