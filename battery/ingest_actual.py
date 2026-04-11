"""A secondary ingest pipeline that simply extracts the actual values at each timestep from the existing parquet files."""

import pandas as pd
from pathlib import Path
from datetime import date, timedelta


def actual_year_to_parquet(
    data_path: Path,
    output_path: Path,
    start: date = date(2025, 1, 1),
) -> pd.DataFrame:
    curr = start
    end = start + timedelta(days=365)
    days = []
    while curr <= end:
        if curr.day == 1:
            print(f"scanning {curr.month}...")
        file_path = data_path / f"{curr}-24h.parquet"
        df = pd.read_parquet(file_path)
        first_issue = df[df["issued_at"] == df["issued_at"].min()]
        actual = first_issue[["period_start", "period_end", "actual"]]
        days.append(actual)
        curr += timedelta(days=1)

    df_out = pd.concat(days)
    df_out = df_out.drop_duplicates(subset=["period_start"], keep="first")
    df_out.to_parquet(output_path / f"{start.year}.parquet")
    return df_out


if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent.parent / "Data" / "daily"
    OUTPUT_PATH = Path(__file__).parent.parent / "Data" / "actual"
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    actual_year_to_parquet(DATA_PATH, OUTPUT_PATH)
    print(f"saved actual to {OUTPUT_PATH}")
