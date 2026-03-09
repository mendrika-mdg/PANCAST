import sys, csv, os, datetime
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(1, "/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/data-preparation/run_nowcast")
import generate_geotiff

root = "/gws/ssde/j25b/swift/UKCEH_nowcast_portal"
userid = "mendrika"
modelname = "pancast"

lats = np.load(
    "/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/geolocation/pancast_lats_africa.npy"
)
lons = np.load(
    "/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/geolocation/pancast_lons_africa.npy"
)

assert lats.shape == lons.shape == (2015, 2187)

testProd = generate_geotiff.UKCEH_PortalProduct(
    userid,
    modelname,
    root,
    lats,
    lons
)

TO_GEOTIFF = "/work/scratch-nopw2/mendrika/pancast-live/log/backprod/ready_geotiff.csv"
NOWCASTS_FOLDER = "/work/scratch-nopw2/mendrika/pancast-live"
PROCESSED_FILES = "/work/scratch-nopw2/mendrika/pancast-live/log/geotiff/backprod_processed_times.csv"
MISSED_FILES = "/work/scratch-nopw2/mendrika/pancast-live/log/geotiff/backprod_missed_times.csv"

LEAD_TIMES_MIN = [30, 60, 90, 120]


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def is_already_processed(time_dict: dict) -> bool:
    if not Path(PROCESSED_FILES).exists():
        return False
    with open(PROCESSED_FILES, newline="") as f:
        reader = csv.DictReader(f)
        return any(all(row[k] == time_dict[k] for k in time_dict) for row in reader)


def append_time(time_dict: dict, log_path: str, mode: str = "a") -> None:
    ensure_parent_dir(log_path)
    path = Path(log_path)
    write_header = mode == "w" or not path.exists() or path.stat().st_size == 0

    with open(log_path, mode, newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["year", "month", "day", "hour", "minute"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(time_dict)


if __name__ == "__main__":

    print("Loading back-production GeoTIFF queue")

    try:
        df = pd.read_csv(TO_GEOTIFF)
    except Exception:
        raise OSError("backprod ready_geotiff.csv does not exist")

    for i in range(len(df)):

        year, month, day, hour, minute = df.loc[i][
            ["year", "month", "day", "hour", "minute"]
        ]

        time_dict = {
            "year": str(year),
            "month": f"{int(month):02d}",
            "day": f"{int(day):02d}",
            "hour": f"{int(hour):02d}",
            "minute": f"{int(minute):02d}",
        }

        print(f"Time to process: {time_dict}")

        if is_already_processed(time_dict):
            print("Already processed — skipping")
            continue

        nowcast_origin_name = (
            f"{time_dict['year']}{time_dict['month']}{time_dict['day']}_"
            f"{time_dict['hour']}{time_dict['minute']}"
        )

        nowcast_origin = datetime.datetime(
            int(year), int(month), int(day), int(hour), int(minute)
        )

        predictions = []
        missing = False

        for lt in LEAD_TIMES_MIN:
            fpath = (
                f"{NOWCASTS_FOLDER}/nowcasts_t{lt:03d}/"
                f"nowcast_t{lt:03d}_from_{nowcast_origin_name}.npy"
            )

            if not Path(fpath).exists():
                print(f"Missing nowcast: {fpath}")
                append_time(time_dict, MISSED_FILES)
                missing = True
                break

            predictions.append(np.load(fpath))

        if missing:
            continue

        print("Generating GeoTIFFs")

        for pred, lt in zip(predictions, LEAD_TIMES_MIN):
            assert pred.shape == lats.shape
            testProd.generate_portal_geotiff(
                pred,
                nowcast_origin,
                lt
            )

        append_time(time_dict, PROCESSED_FILES)
        print("Done")
