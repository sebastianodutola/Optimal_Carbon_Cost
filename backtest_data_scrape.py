from datetime import date, time, datetime
from ingest import fetch_range
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

START_DATE = date(2017, 9, 26)
START_DT = datetime.combine(START_DATE, time=time(0))
CURRENT_DATE = date(2026, 3, 3)
DATAPATH = Path(__file__).parent.joinpath("Data/backtest_data.py")

days = (CURRENT_DATE - START_DATE).days  # 3080

if __name__ == "__main__":
    df = fetch_range(START_DT, days)
    table = pa.Table.from_pandas(df)
    print(f"Writing {days} days to {DATAPATH}")
    pq.write_table(table, DATAPATH)
    print("Finished writing...")
    print("--------------------------------------")
    print("Summary of written data")
    print("--------------------------------------")
    print(df.describe())
