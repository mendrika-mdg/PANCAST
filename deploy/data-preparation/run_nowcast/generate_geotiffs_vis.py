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
    "/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/geolocation/nrt_lats_africa.npy"
)
lons = np.load(
    "/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/geolocation/nrt_lons_africa.npy"
)

assert lats.shape == lons.shape == (2015, 2186)

testProd = generate_geotiff.UKCEH_PortalProduct(
    userid,
    modelname,
    root,
    lats,
    lons,
    irregular=True
)

TO_GEOTIFF = "/work/scratch-nopw2/mendrika/pancast-live/log/ready_geotiff.csv"
NOWCASTS_FOLDER = "/work/scratch-nopw2/mendrika/pancast-live"
PROCESSED_FILES = "/work/scratch-nopw2/mendrika/pancast-live/log/geotiff/processed_times.csv"
MISSED_FILES = "/work/scratch-nopw2/mendrika/pancast-live/log/geotiff/missed_times.csv"

LEAD_TIMES_MIN = [30, 60, 90, 120]

def is_already_processed(time_dict: dict) -> bool:
    if not Path(PROCESSED_FILES).exists():
        return False
    with open(PROCESSED_FILES, newline="") as f:
        reader = csv.DictReader(f)
        return any(all(row[k] == time_dict[k] for k in time_dict) for row in reader)

def append_time(time_dict: dict, log_path: str, mode: str = "a") -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = mode == "w" or not path.exists() or path.stat().st_size == 0

    with open(log_path, mode, newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["year", "month", "day", "hour", "minute"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(time_dict)


def align_prediction_to_nrt(pred, lats, lons):
    ny, nx = lats.shape

    if pred.shape == (ny, nx):
        return pred

    if pred.shape == (ny, nx + 1):
        return pred[:, :nx]

    raise ValueError(
        f"Incompatible PANCAST shape {pred.shape} "
        f"for NRT grid {lats.shape}"
    )


if __name__ == "__main__":

    print("Getting file to be processed")

    try:
        df = pd.read_csv(TO_GEOTIFF)
    except Exception:
        raise OSError("ready_geotiff.csv does not exist")

    year, month, day, hour, minute = df.loc[0][
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
        print("File already processed")
        sys.exit(0)

    nowcast_origin_name = (
        f"{time_dict['year']}{time_dict['month']}{time_dict['day']}_"
        f"{time_dict['hour']}{time_dict['minute']}"
    )

    nowcast_origin = datetime.datetime(
        int(year), int(month), int(day), int(hour), int(minute)
    )

    print("Loading nowcasts")

    predictions = []
    for lt in LEAD_TIMES_MIN:
        fpath = (
            f"{NOWCASTS_FOLDER}/nowcasts_t{lt:03d}/"
            f"nowcast_t{lt:03d}_from_{nowcast_origin_name}.npy"
        )

        if not Path(fpath).exists():
            print(f"Missing nowcast file: {fpath}")
            append_time(time_dict, MISSED_FILES, mode="a")
            sys.exit(1)

        p = np.load(fpath)
        p = align_prediction_to_nrt(p, lats, lons)
        predictions.append(p)
    
    print("Generating GeoTIFFs")

    for pred, lt in zip(predictions, LEAD_TIMES_MIN):
        assert np.isfinite(pred).all(), "NaNs or infs in prediction"
        testProd.generate_portal_geotiff(
            pred,
            nowcast_origin,
            lt
        )

    print("Keeping record")
    append_time(time_dict, PROCESSED_FILES, mode="a")

    print("Done")
