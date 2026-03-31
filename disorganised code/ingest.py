import httpx
from datetime import datetime, date, timezone, time, timedelta
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

DATAPATH = Path(__file__).parent.joinpath("intensity.parquet")
BASE_URL = "https://api.carbonintensity.org.uk"
FW48 = "/intensity/{date}/fw48h"

now = datetime.now()


def fetch_range(start: datetime, days: int = 2):
    seen: set[str] = set()
    out: list[pd.DataFrame] = []

    print(f"Fetching forcast data from {start}...")
    with httpx.Client(timeout=30.0) as client:
        for day in tqdm(range(0, days, 2)):
            date = start + timedelta(days=day)
            date = date.isoformat(timespec="minutes") + "Z"
            url = f"{BASE_URL}{FW48.format(date=date)}"

            response = httpx.get(url)
            response.raise_for_status()
            df = pd.DataFrame(response.json()["data"])

            # dedup
            mask = ~df["from"].isin(seen)
            df = df[mask]
            seen.update(df["from"].to_list())

            out.append(df)
    res = pd.concat(out)
    return res


if __name__ == "__main__":
    d = date(2026, 2, 27)
    days = 2
    df = fetch_range(start=datetime.combine(date=d, time=time(0)), days=days)
    table = pa.Table.from_pandas(df)
    print(f"Writing to {DATAPATH}...")
    pq = pq.write_table(table, DATAPATH)
    print(f"Finished Writing.")
    print("Description of saved DataFrame")
    print("------------------------------")
    print(df.describe())
