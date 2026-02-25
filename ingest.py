import httpx
from datetime import datetime
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

DATAPATH = Path(__file__).parent.joinpath("intensity.parquet")
BASE_URL = "https://api.carbonintensity.org.uk"
FW48 = "/intensity/{start}/fw48h"

now = datetime.now()


def fetch_next(dt=now):
    start = dt.isoformat() + "Z"
    print(f"Fetching forcast data from {start}...")
    url = f"{BASE_URL}{FW48.format(start=start)}"
    response = httpx.get(url)
    response.raise_for_status()
    return response


response = fetch_next()
df = pd.DataFrame(response.json()["data"])
table = pa.Table.from_pandas(df)
print(f"Writing to {DATAPATH}...")
pq = pq.write_table(table, DATAPATH)
print(f"Finished Writing.")
