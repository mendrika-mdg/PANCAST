import os, csv, sys, torch
import numpy as np
from pathlib import Path
from netCDF4 import Dataset
from datetime import datetime, timedelta, UTC

sys.path.insert(1, "/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/data-preparation")
from pancast_input_preparation import (
    get_time, update_hour, process_file,
    CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
    CONTEXT_LON_MIN, CONTEXT_LON_MAX,
)

sys.path.insert(1, "/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/streamlit")
from utils_preprocessed_64 import (
    load_models,
    scale_input,
    ensemble_predict,
    rescale_after_threshold,
    smooth_prediction,
    gamma_boost,
)

NB_X0 = 100

OUTPUT_FOLDER = "/work/scratch-nopw2/mendrika/pancast-live"
PROCESSED_FILES = f"{OUTPUT_FOLDER}/log/backprod/processed_times.csv"
READY_FOR_GEOTIFF = f"{OUTPUT_FOLDER}/log/backprod/ready_geotiff.csv"
MISSED_FILES = f"{OUTPUT_FOLDER}/log/backprod/missed_times.csv"

LEAD_TIMES_MIN = [30, 60, 90, 120]
LAGS_BEFORE_T0_MIN = [120, 90, 60, 30, 0]


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


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


def dict_to_filename(time_dict: dict) -> str:
    path_core = (
        f"/gws/ssde/j25b/swift/rt_cores/"
        f"{time_dict['year']}/{time_dict['month']}/{time_dict['day']}/"
        f"{time_dict['hour']}{time_dict['minute']}"
    )
    return (
        f"{path_core}/Convective_struct_extended_"
        f"{time_dict['year']}{time_dict['month']}{time_dict['day']}"
        f"{time_dict['hour']}{time_dict['minute']}_000.nc"
    )


def file_exists(time_dict: dict) -> bool:
    return Path(dict_to_filename(time_dict)).exists()


def all_past_files_exist(time_dict: dict) -> bool:
    dt = datetime(
        int(time_dict["year"]),
        int(time_dict["month"]),
        int(time_dict["day"]),
        int(time_dict["hour"]),
        int(time_dict["minute"]),
        tzinfo=UTC,
    )

    for lag in LAGS_BEFORE_T0_MIN:
        t = dt - timedelta(minutes=lag)
        fdict = {
            "year": str(t.year),
            "month": f"{t.month:02d}",
            "day": f"{t.day:02d}",
            "hour": f"{t.hour:02d}",
            "minute": f"{t.minute:02d}",
        }
        if not file_exists(fdict):
            return False

    return True


def generate_nowcasts(time_t0: dict, models: dict) -> None:

    year = time_t0["year"]
    month = time_t0["month"]
    day = time_t0["day"]
    hour = time_t0["hour"]
    minute = time_t0["minute"]

    nowcast_origin = f"{year}{month}{day}_{hour}{minute}"
    input_path = f"{OUTPUT_FOLDER}/inputs_t0/input-{nowcast_origin}.pt"

    ensure_parent_dir(input_path)

    file_before_t0 = [
        update_hour(time_t0, hours_to_add=0, minutes_to_add=-lag)["path"]
        for lag in LAGS_BEFORE_T0_MIN
    ]

    if not all(os.path.exists(f) for f in file_before_t0):
        append_time(time_t0, MISSED_FILES)
        return

    tensors = []
    for f in file_before_t0:
        t = process_file(
            f,
            NB_X0,
            CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
            CONTEXT_LON_MIN, CONTEXT_LON_MAX,
        )
        if t is None:
            append_time(time_t0, MISSED_FILES)
            return
        tensors.append(t)

    input_tensor = torch.stack(tensors, dim=0)

    if input_tensor.shape != (5, NB_X0, 13):
        raise ValueError(f"Unexpected input tensor shape {tuple(input_tensor.shape)}")

    torch.save(input_tensor, input_path)

    input_processed = scale_input(input_tensor).unsqueeze(0)

    for lt in LEAD_TIMES_MIN:
        pred = ensemble_predict(models[lt], input_processed).cpu().numpy()
        pred = rescale_after_threshold(pred, floor=0.12)
        pred = gamma_boost(pred, gamma=0.6)
        pred = smooth_prediction(pred, sigma=0.8)

        out_path = (
            f"{OUTPUT_FOLDER}/nowcasts_t{lt:03d}/"
            f"nowcast_t{lt:03d}_from_{nowcast_origin}.npy"
        )
        ensure_parent_dir(out_path)
        np.save(out_path, pred)

    append_time(time_t0, PROCESSED_FILES)
    append_time(time_t0, READY_FOR_GEOTIFF, mode="w")


if __name__ == "__main__":

    print("Loading models")
    models = {lt: load_models(lt) for lt in LEAD_TIMES_MIN}

    start = datetime(2026, 1, 14, 0, 0, tzinfo=UTC)
    end   = datetime(2026, 1, 15, 11, 0, tzinfo=UTC)

    dt = start
    while dt <= end:
        time_t0 = {
            "year": str(dt.year),
            "month": f"{dt.month:02d}",
            "day": f"{dt.day:02d}",
            "hour": f"{dt.hour:02d}",
            "minute": f"{dt.minute:02d}",
        }

        print(f"Nowcast origin: {time_t0}")

        if file_exists(time_t0) and all_past_files_exist(time_t0):
            generate_nowcasts(time_t0, models)
        else:
            append_time(time_t0, MISSED_FILES)

        dt += timedelta(minutes=15)
