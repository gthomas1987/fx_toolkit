"""FX option date conventions: T+2 spot, settlement-based tenor, expiry = settle - 2bd.

Standard FX option date schedule:
    trade_date
    -> spot_settlement = trade_date + 2 business days        (T+2 spot)
    -> option_settlement = spot_settlement + tenor           (calendar)
    -> option_expiry = option_settlement - 2 business days   (so a spot trade
                                                              on expiry settles
                                                              on the option's
                                                              cash settlement
                                                              date)

Tenor labels:
    'NW' = N weeks (calendar, e.g. '6W' = 42 days)
    'NM' = N calendar months (with end-of-month clamping)
    '1Y' = 12 months

Business-day handling: simple Mon-Fri (no holiday calendar). Sufficient for
typical FX flow pricing. Holidays affect dates by 1-2 calendar days at most
and have negligible impact on pricing relative to vol uncertainty.
"""
from __future__ import annotations
from datetime import date, timedelta
from dataclasses import dataclass
import calendar as _cal


def add_business_days(d: date, n: int) -> date:
    """Add n business days (Mon-Fri only)."""
    if n == 0:
        return d
    direction = 1 if n > 0 else -1
    n_abs = abs(n)
    while n_abs > 0:
        d = d + timedelta(days=direction)
        if d.weekday() < 5:
            n_abs -= 1
    return d


def next_business_day(d: date) -> date:
    """If d falls on a weekend, push forward to next Monday."""
    while d.weekday() >= 5:
        d = d + timedelta(days=1)
    return d


def add_calendar_months(d: date, months: int) -> date:
    """Add calendar months, clamping to the last valid day of the target month."""
    new_month = d.month + months
    new_year = d.year + (new_month - 1) // 12
    new_month = ((new_month - 1) % 12) + 1
    last_day = _cal.monthrange(new_year, new_month)[1]
    return date(new_year, new_month, min(d.day, last_day))


@dataclass
class OptionDates:
    trade_date: date
    spot_settlement: date
    option_settlement: date
    option_expiry: date
    T_years: float
    tenor_label: str


def compute_option_dates(trade_date: date, tenor_label: str) -> OptionDates:
    """Compute the full FX option date schedule for a trade.

    Parameters
    ----------
    trade_date : date
        The trade / valuation date.
    tenor_label : str
        Tenor as 'NW' (weeks) or 'NM' (months) or '1Y'.

    Returns
    -------
    OptionDates with spot_settlement, option_settlement, option_expiry and
    T_years (= (expiry - trade_date) / 365 — used as the option vol horizon).

    Examples
    --------
    >>> compute_option_dates(date(2026, 4, 27), '1M')
    # spot=29 Apr 2026 (Wed), settle=29 May 2026 (Fri), expiry=27 May 2026 (Wed)
    """
    spot_settle = add_business_days(trade_date, 2)

    if tenor_label.endswith('W'):
        weeks = int(tenor_label[:-1])
        opt_settle_raw = spot_settle + timedelta(days=weeks * 7)
    elif tenor_label.endswith('M'):
        months = int(tenor_label[:-1])
        opt_settle_raw = add_calendar_months(spot_settle, months)
    elif tenor_label == '1Y':
        opt_settle_raw = add_calendar_months(spot_settle, 12)
    else:
        raise ValueError(f"Unknown tenor label: {tenor_label}")

    # Modified-following: if settlement falls on weekend, push to next BD
    opt_settle = next_business_day(opt_settle_raw)
    opt_expiry = add_business_days(opt_settle, -2)
    T_years = (opt_expiry - trade_date).days / 365.0

    return OptionDates(
        trade_date=trade_date,
        spot_settlement=spot_settle,
        option_settlement=opt_settle,
        option_expiry=opt_expiry,
        T_years=T_years,
        tenor_label=tenor_label,
    )
