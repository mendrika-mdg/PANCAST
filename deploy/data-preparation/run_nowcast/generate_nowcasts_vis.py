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
    lats, lons
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

PROCESSED_FILES = "/work/scratch-nopw2/mendrika/pancast-live/log/processed_times.csv"
READY_FOR_GEOTIFF = "/work/scratch-nopw2/mendrika/pancast-live/log/ready_geotiff.csv"
MISSED_FILES = "/work/scratch-nopw2/mendrika/pancast-live/log/missed_times.csv"
OUTPUT_FOLDER = "/work/scratch-nopw2/mendrika/pancast-live/"

LEAD_TIMES_MIN = [30, 60, 90, 120]
LAGS_BEFORE_T0_MIN = [120, 90, 60, 30, 0]

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
        writer = csv.DictWriter(f, fieldnames=["year", "month", "day", "hour", "minute"])
        if write_header:
            writer.writeheader()
        writer.writerow(time_dict)

def round_to_nearest_15(dt: datetime) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    minutes = (dt.minute + 7) // 15 * 15
    if minutes == 60:
        dt += timedelta(hours=1)
        minutes = 0
    return dt.replace(minute=minutes)

def get_time_dict() -> dict:
    dt = datetime.now(UTC) + timedelta(minutes=-30)
    dt = round_to_nearest_15(dt)
    return {
        "year": str(dt.year),
        "month": f"{dt.month:02d}",
        "day": f"{dt.day:02d}",
        "hour": f"{dt.hour:02d}",
        "minute": f"{dt.minute:02d}",
    }

def dict_to_filename(time_dict: dict) -> str:
    path_core = f"/gws/ssde/j25b/swift/rt_cores/{time_dict['year']}/{time_dict['month']}/{time_dict['day']}/{time_dict['hour']}{time_dict['minute']}"
    return f"{path_core}/Convective_struct_extended_{time_dict['year']}{time_dict['month']}{time_dict['day']}{time_dict['hour']}{time_dict['minute']}_000.nc"

def file_exists(time_dict: dict) -> bool:
    return Path(dict_to_filename(time_dict)).exists()

def all_past_files_exist(time_dict: dict) -> bool:
    dt = datetime(
        year=int(time_dict["year"]),
        month=int(time_dict["month"]),
        day=int(time_dict["day"]),
        hour=int(time_dict["hour"]),
        minute=int(time_dict["minute"]),
        tzinfo=UTC
    )

    required = []
    for lag in LAGS_BEFORE_T0_MIN:
        t = dt - timedelta(minutes=lag)
        required.append({
            "year": str(t.year),
            "month": f"{t.month:02d}",
            "day": f"{t.day:02d}",
            "hour": f"{t.hour:02d}",
            "minute": f"{t.minute:02d}",
        })

    return all(file_exists(f) for f in required)

def generate_nowcasts(time_t0: dict, models: dict) -> None:

    year = time_t0["year"]
    month = time_t0["month"]
    day = time_t0["day"]
    hour = time_t0["hour"]
    minute = time_t0["minute"]

    file_t0 = dict_to_filename(time_t0)

    nowcast_origin = f"{year}{month}{day}_{hour}{minute}"
    input_path = f"{OUTPUT_FOLDER}/inputs_t0/input-{nowcast_origin}.pt"

    ensure_parent_dir(input_path)
    ensure_parent_dir(PROCESSED_FILES)
    ensure_parent_dir(READY_FOR_GEOTIFF)
    ensure_parent_dir(MISSED_FILES)

    file_before_t0 = [
        update_hour(time_t0, hours_to_add=0, minutes_to_add=-lag)["path"]
        for lag in LAGS_BEFORE_T0_MIN
    ]

    if not all(os.path.exists(f) for f in file_before_t0):
        print(f"Missing files, saving logs to {MISSED_FILES}")
        append_time(time_dict=time_t0, mode="a", log_path=MISSED_FILES)
        return

    with Dataset(file_t0, "r") as data_t0:
        pmax_lat = data_t0["Pmax_lat"][:]
        pmax_lon = data_t0["Pmax_lon"][:]

    if pmax_lat.size == 0 or pmax_lon.size == 0:
        print("No cores at t0, skipping nowcast generation")
        append_time(time_dict=time_t0, mode="a", log_path=MISSED_FILES)
        return

    tensors = []
    for f in file_before_t0:
        tensors.append(
            process_file(
                f,
                NB_X0,
                CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
                CONTEXT_LON_MIN, CONTEXT_LON_MAX
            )
        )

    input_tensor = torch.stack(tensors, dim=0)

    if tuple(input_tensor.shape) != (5, NB_X0, 13):
        raise ValueError(f"Unexpected input tensor shape: {tuple(input_tensor.shape)}")

    torch.save(input_tensor, input_path)

    input_processed = scale_input(input_tensor).unsqueeze(0)

    for lt in LEAD_TIMES_MIN:
        pred = ensemble_predict(models[lt], input_processed).cpu().numpy()
        pred = rescale_after_threshold(pred, floor=0.12) #0.12 is the best
        pred = gamma_boost(pred, gamma=0.85) #0.8 is the best
        pred = smooth_prediction(pred, sigma=0.8)

        out_path = f"{OUTPUT_FOLDER}/nowcasts_t{lt:03d}/nowcast_t{lt:03d}_from_{nowcast_origin}.npy"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, pred)

    append_time(time_dict=time_t0, mode="a", log_path=PROCESSED_FILES)
    append_time(time_dict=time_t0, mode="w", log_path=READY_FOR_GEOTIFF)
    print(f"File processed, saving logs to {PROCESSED_FILES} and sending signal to {READY_FOR_GEOTIFF}")

if __name__ == "__main__":
    print("Getting NRT information")
    time_t0 = get_time_dict()

    print(f"Nowcast origin: {time_t0}")

    if is_already_processed(time_t0):
        print(f"Nowcasts from {time_t0} already produced")
        sys.exit(0)

    print(f"File exists: {file_exists(time_t0)}")
    print(f"All past files exist: {all_past_files_exist(time_t0)}")

    if file_exists(time_t0) and all_past_files_exist(time_t0):
        print("Loading models")
        models = {lt: load_models(lt) for lt in LEAD_TIMES_MIN}

        print("Generating nowcasts")
        generate_nowcasts(time_t0, models)
    else:
        print("Missing file(s)")
