from datetime import date, datetime, time, timedelta
import pandas as pd
import httpx
from tqdm import tqdm
from pathlib import Path
import asyncio

DATADIR = Path(__file__).parent.parent / "Data" / "daily"
DATADIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "https://api.carbonintensity.org.uk"
FW = "/intensity/{dt}/fw{horizon_type}"

MAX_CONCURRENT = 10


async def fetch_one(
    client: httpx.AsyncClient, dt: datetime, sem: asyncio.Semaphore, horizon_type: str
) -> tuple[datetime, dict]:
    async with sem:
        dt_str = dt.isoformat(timespec="minutes") + "Z"
        url = f"{BASE_URL}{FW.format(dt=dt_str, horizon_type=horizon_type)}"
        r = await client.get(url)
        r.raise_for_status()
        return (dt, r.json())


async def fetch_day(
    client: httpx.AsyncClient, day: date, sem: asyncio.Semaphore, horizon_type: str
):
    """Fetch all 48 half-hour forecast windows for a given calendar day."""
    midnight = datetime.combine(day, time.min)
    times = [midnight + timedelta(minutes=30 * step) for step in range(48)]
    tasks = [fetch_one(client, t, sem, horizon_type) for t in times]
    return await asyncio.gather(*tasks)


def date_range(start: date, end: date):
    curr = start
    while curr <= end:
        yield curr
        curr += timedelta(days=1)


def month_range(start: date, end: date):
    curr = start
    while curr <= end:
        if curr.month == 12:
            next_month = curr.replace(year=curr.year + 1, month=1, day=1)
        else:
            next_month = curr.replace(month=curr.month + 1, day=1)
        yield curr, min(next_month - timedelta(days=1), end)
        curr = next_month


async def fetch_batch(
    start: date, end: date, data_path: Path, horizon_type: str = "48h"
):
    data_path.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    if horizon_type != "48h" and horizon_type != "24h":
        raise ValueError("Horizon type must be '24h' or '48h'")

    default_forecast, default_actual = 100, 100
    async with httpx.AsyncClient(timeout=30.0) as client:
        for day in tqdm(list(date_range(start, end))):
            if _partition_path(data_path, day, horizon_type).exists():
                continue
            res = await fetch_day(client, day, sem, horizon_type=horizon_type)
            df = process_day(res)
            df = repair_day(
                df, day, default_forecast, default_actual, horizon_type=horizon_type
            )
            last = df[df["issued_at"] == df["issued_at"].max()].iloc[1]
            default_forecast = int(last["forecast"])
            default_actual = int(last["actual"])
            save_day(df, data_path, horizon_type)


def repair_day(
    df_day: pd.DataFrame,
    day: date,
    default_forecast: int,
    default_actual: int,
    horizon_type: str,
) -> pd.DataFrame:
    """
    df_day:     single day's raw data
    day:        the date being processed
    df_history: all previously saved data (for cross-day ffill fallback)
    """
    expected_issued = pd.date_range(
        start=pd.Timestamp(day), periods=48, freq="30min", tz="UTC"
    )

    rows = []

    if horizon_type == "48h":
        num_periods = 97
    elif horizon_type == "24h":
        num_periods = 49
    else:
        raise ValueError("horizon type must be '48h' or '24h'")

    for issued_at in expected_issued:
        periods = pd.date_range(
            start=issued_at - pd.Timedelta("30min"),
            periods=num_periods,
            freq="30min",
            tz="UTC",
        )
        rows.append(pd.DataFrame({"issued_at": issued_at, "period_start": periods}))
    skeleton = pd.concat(rows, ignore_index=True)
    skeleton["period_end"] = skeleton["period_start"] + pd.Timedelta("30min")

    # Merge actual data onto skeleton; unmatched (issued_at, period_start) pairs → NaN
    df_day = skeleton.merge(
        df_day[["issued_at", "period_start", "forecast", "actual"]],
        on=["issued_at", "period_start"],
        how="left",
    )

    # Fill row 0 with last known / default forecast
    if pd.isna(df_day.iloc[0]["forecast"]):
        df_day.iloc[0, df_day.columns.get_loc("forecast")] = default_forecast
    if pd.isna(df_day.iloc[0]["actual"]):
        df_day.iloc[0, df_day.columns.get_loc("actual")] = default_actual

    # propogate the ffill for the first forecast
    mask = df_day["issued_at"] == pd.Timestamp(day, tz="UTC")
    df_day.loc[mask, ["forecast", "actual"]] = df_day.loc[
        mask, ["forecast", "actual"]
    ].ffill()

    # Ffill pattern, first make sure that the first periods are all filled
    df_day = df_day.sort_values(["period_start", "issued_at"])
    df_day[["forecast", "actual"]] = df_day.groupby("period_start")[
        ["forecast", "actual"]
    ].ffill()

    # Then make sure that the periods after the first (for each issue is filled)
    df_day = df_day.sort_values(["issued_at", "period_start"])
    df_day[["forecast", "actual"]] = df_day.groupby("issued_at")[
        ["forecast", "actual"]
    ].ffill()

    return df_day


def process_day(res: list) -> pd.DataFrame:
    """
    Takes the list of tuples returned by fetch_day and returns a cleaned DataFrame.

    res: list of (issued_at: datetime, packet: {"data": list[ForecastDict]})

    ForecastDict:
    {
        "from": str (ISO 8601),
        "to":   str (ISO 8601),
        "intensity": {"forecast": int | None, "actual": int | None, "index": str}
    }

    Output columns: issued_at | period_start | period_end | forecast | actual
    """
    rows = []
    for issued_at, packet in res:
        for entry in packet["data"]:
            intensity = entry["intensity"]
            rows.append(
                {
                    "issued_at": issued_at,
                    "period_start": entry["from"],
                    "period_end": entry["to"],
                    "forecast": intensity.get("forecast", pd.NA),
                    "actual": intensity.get("actual", pd.NA),
                }
            )
    df = pd.DataFrame(rows)
    df["issued_at"] = pd.to_datetime(df["issued_at"], utc=True)
    df["period_start"] = pd.to_datetime(df["period_start"], utc=True)
    df["period_end"] = pd.to_datetime(df["period_end"], utc=True)
    df["forecast"] = df["forecast"].astype("Int16")
    df["actual"] = df["actual"].astype("Int16")
    return df.drop_duplicates(subset=["issued_at", "period_start"])


def save_day(df: pd.DataFrame, data_path: Path, horizon_type: str):
    """
    Saves one day's DataFrame as a parquet partition: data_path/YYYY-MM-DD.parquet
    Derived from the date of the earliest issued_at in the DataFrame.
    """
    out = _partition_path(
        data_path, df["issued_at"].min().date(), horizon_type=horizon_type
    )
    df.to_parquet(out, index=False)


def _partition_path(data_path: Path, day: date, horizon_type: str) -> Path:
    return data_path / f"{day}-{horizon_type}.parquet"


async def _run(horizon_type: str):
    start = date(2025, 1, 1)
    end = date(2026, 1, 1)
    for batch_start, batch_end in month_range(start, end):
        print(f"Fetching {batch_start} -> {batch_end}")
        await fetch_batch(batch_start, batch_end, DATADIR, horizon_type=horizon_type)


if __name__ == "__main__":
    asyncio.run(_run("24h"))
    asyncio.run(_run("48h"))
