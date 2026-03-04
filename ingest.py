import httpx
from datetime import datetime, date, timezone, time
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

DATAPATH = Path(__file__).parent.joinpath("intensity.parquet")
BASE_URL = "https://api.carbonintensity.org.uk"
FW48 = "/intensity/{start}/fw48h"

now = datetime.now()


def fetch_next(dt=now):
    start = dt.isoformat(timespec="minutes") + "Z"
    print(f"Fetching forcast data from {start}...")
    url = f"{BASE_URL}{FW48.format(start=start)}"
    response = httpx.get(url)
    response.raise_for_status()
    return response


if __name__ == "__main__":
    d = date(2026, 1, 1)
    response = fetch_next(dt=datetime.combine(date=d, time=time(0)))
    df = pd.DataFrame(response.json()["data"])
    table = pa.Table.from_pandas(df)
    print(f"Writing to {DATAPATH}...")
    pq = pq.write_table(table, DATAPATH)
    print(f"Finished Writing.")
