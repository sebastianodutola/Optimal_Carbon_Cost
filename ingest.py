import httpx
from datetime import datetime, date, timezone, time, timedelta
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

DATAPATH = Path(__file__).parent.joinpath("intensity.parquet")
BASE_URL = "https://api.carbonintensity.org.uk"
FW24 = "/intensity/{date}/fw24h"

now = datetime.now()


def fetch_next(start=now, days=2):
    print(f"Fetching forcast data from {start}...")
    dfs = []
    for day in range(days):
        date = start + timedelta(days=day)
        date = date.isoformat(timespec="minutes") + "Z"
        url = f"{BASE_URL}{FW24.format(date=date)}"
        response = httpx.get(url)
        response.raise_for_status()
        dfs.append(pd.DataFrame(response.json()["data"]))
    df = pd.concat(dfs)
    df.drop_duplicates(subset=["from"], inplace=True)
    return df


if __name__ == "__main__":
    d = date(2026, 1, 1)
    days = 30
    df = fetch_next(start=datetime.combine(date=d, time=time(0)), days=days)
    table = pa.Table.from_pandas(df)
    print(f"Writing to {DATAPATH}...")
    pq = pq.write_table(table, DATAPATH)
    print(f"Finished Writing.")
