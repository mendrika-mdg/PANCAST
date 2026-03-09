import os
import sys
import torch
import numpy as np      
from netCDF4 import Dataset  
from scipy.ndimage import label
from datetime import datetime, timedelta
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics

def get_time(file_path):
    """
    Extract zero-padded time components from a file path like:
    /.../2025/02/05/1345/Convective_struct_extended_202502051345_000.nc

    Returns:
        dict with keys: 'year', 'month', 'day', 'hour', 'minute'
    """
    basename = os.path.basename(file_path)

    # Extract the datetime string from the filename
    parts = basename.split("_")
    
    timestamp = parts[-2]
    if len(timestamp) != 12:
        raise ValueError(f"Invalid timestamp format in filename: {timestamp}")

    return {
        "year":   timestamp[0:4],
        "month":  timestamp[4:6],
        "day":    timestamp[6:8],
        "hour":   timestamp[8:10],
        "minute": timestamp[10:12]
    }


def update_hour(date_dict, hours_to_add, minutes_to_add):
    """
    Add hours to a datetime dictionary and return the updated dict and a generated file path.

    Args:
        date_dict     (dict): Keys: 'year', 'month', 'day', 'hour', 'minute' as strings, e.g. "01", "23"
        hours_to_add   (int): Number of hours to add.
        minutes_to_add (int): Number of minutes to add.

    Returns:
        dict: {
            'time': updated time dictionary with zero-padded strings,
            'path': file path in format /.../YYYYMMDDHHMM_000.nc
        }
    """
    # Parse original time
    time_obj = datetime(
        int(date_dict["year"]),
        int(date_dict["month"]),
        int(date_dict["day"]),
        int(date_dict["hour"]),
        int(date_dict["minute"])
    )

    # Add hours and minutes
    updated = time_obj + timedelta(hours=hours_to_add, minutes=minutes_to_add)

    # Create updated dictionary with padded strings
    new_date_dict = {
        "year":   f"{updated.year:04d}",
        "month":  f"{updated.month:02d}",
        "day":    f"{updated.day:02d}",
        "hour":   f"{updated.hour:02d}",
        "minute": f"{updated.minute:02d}"
    }

    # Build file path safely using single quotes inside f-strings
    path_core = f"/gws/ssde/j25b/swift/rt_cores/{new_date_dict['year']}/{new_date_dict['month']}/{new_date_dict['day']}/{new_date_dict['hour']}{new_date_dict['minute']}"
    file_path = f"{path_core}/Convective_struct_extended_{new_date_dict['year']}{new_date_dict['month']}{new_date_dict['day']}{new_date_dict['hour']}{new_date_dict['minute']}_000.nc"

    return {'time': new_date_dict, 'path': file_path}


def extract_box(matrix, y, x, box_size=3):
    half = box_size // 2
    y_min = max(y - half, 0)
    y_max = min(y + half + 1, matrix.shape[0])
    x_min = max(x - half, 0)
    x_max = min(x + half + 1, matrix.shape[1])
    return matrix[y_min:y_max, x_min:x_max]


def create_storm_database(data_t, lats, lons):
    tir_t = data_t["cores"][y_min:y_max+1, x_min:x_max+1].data
    temp_t = tir_t < 0

    Pmax_lat = np.asarray(data_t["Pmax_lat"][:]).ravel()
    Pmax_lon = np.asarray(data_t["Pmax_lon"][:]).ravel()

    valid = (
        (Pmax_lon >= CONTEXT_LON_MIN) & (Pmax_lon <= CONTEXT_LON_MAX) &
        (Pmax_lat >= CONTEXT_LAT_MIN) & (Pmax_lat <= CONTEXT_LAT_MAX)
    )

    Pmax_lat = Pmax_lat[valid]
    Pmax_lon = Pmax_lon[valid]

    labeled_array, _ = label(temp_t)
    core_labels = np.unique(labeled_array[labeled_array != 0])

    dict_storm_size = {lab: np.sum(labeled_array == lab) * 9 for lab in core_labels}

    lats_crop = np.asarray(lats[y_min:y_max+1, x_min:x_max+1])
    lons_crop = np.asarray(lons[y_min:y_max+1, x_min:x_max+1])

    dict_storm_extent = {}
    for lab in core_labels:
        mask = labeled_array == lab

        lat_vals = lats_crop[mask]
        lon_vals = lons_crop[mask]

        if lat_vals.size == 0 or lon_vals.size == 0:
            continue

        if np.all(np.isnan(lat_vals)) or np.all(np.isnan(lon_vals)):
            continue

        dict_storm_extent[lab] = {
            "lat_min": float(np.nanmin(lat_vals)),
            "lat_max": float(np.nanmax(lat_vals)),
            "lon_min": float(np.nanmin(lon_vals)),
            "lon_max": float(np.nanmax(lon_vals)),
        }

    dict_storm_temperature = {}
    for lab in core_labels:
        mask = labeled_array == lab
        tir_core = tir_t[mask]
        yx_indices = np.argwhere(mask)

        if yx_indices.size == 0:
            continue

        y, x = yx_indices[np.argmin(tir_core)]
        box = extract_box(tir_t, y, x)
        dict_storm_temperature[lab] = float(np.nanmean(box))

    storm_database = {}
    for lat, lon in zip(Pmax_lat, Pmax_lon):
        if np.isnan(lat) or np.isnan(lon):
            continue

        try:
            y_idx, x_idx = snflics.to_yx(lat, lon, lats_crop, lons_crop)
            if y_idx is None or x_idx is None:
                continue
        except (IndexError, TypeError):
            continue

        lab = labeled_array[y_idx, x_idx]

        if lab == 0:
            continue

        if lab in storm_database:
            continue

        if lab not in dict_storm_extent:
            continue

        ext = dict_storm_extent[lab]
        storm_database[int(lab)] = {
            "lat": float(lat),
            "lon": float(lon),
            "lat_min": ext["lat_min"],
            "lat_max": ext["lat_max"],
            "lon_min": ext["lon_min"],
            "lon_max": ext["lon_max"],
            "tir": dict_storm_temperature.get(lab, float("nan")),
            "size": dict_storm_size[lab],
            "mask": 1
        }

    return storm_database


def generate_fictional_storm(context_lat_min, context_lat_max,
                             context_lon_min, context_lon_max):
    # Generate a dummy non-convective storm entry with mask=0
    lat = np.random.uniform(context_lat_min, context_lat_max)
    lon = np.random.uniform(context_lon_min, context_lon_max)

    storm = {
        "lat": lat,
        "lon": lon,
        "lat_min": lat,
        "lat_max": lat,
        "lon_min": lon,
        "lon_max": lon,
        "tir": 30.0,
        "size": 0.0,
        "mask": 0
    }

    return ("artificial", storm)

def pad_observed_storms(storm_db, nb_x0,
                        context_lat_min, context_lat_max,
                        context_lon_min, context_lon_max):
    # Ensure a fixed number of storm cores by truncating or padding

    storm_list = list(storm_db.items())

    if len(storm_list) >= nb_x0:
        sorted_db = sorted(storm_list, key=lambda item: item[1]["tir"])
        return sorted_db[:nb_x0]

    needed = nb_x0 - len(storm_list)
    for _ in range(needed):
        storm_list.append(
            generate_fictional_storm(
                context_lat_min=context_lat_min,
                context_lat_max=context_lat_max,
                context_lon_min=context_lon_min,
                context_lon_max=context_lon_max
            )
        )

    return storm_list


def transform_to_array(data):
    # Transform list of storms into an array of local per-core features

    result = []
    for _, entry in data:
        lat = float(entry["lat"])
        lon = float(entry["lon"])
        lat_min = float(entry.get("lat_min", lat))
        lat_max = float(entry.get("lat_max", lat))
        lon_min = float(entry.get("lon_min", lon))
        lon_max = float(entry.get("lon_max", lon))
        tir = float(entry["tir"])
        size = float(entry["size"])
        mask = int(entry["mask"])

        # [lat, lon, lat_min, lat_max, lon_min, lon_max, tir, size, mask]
        result.append([
            lat, lon,
            lat_min, lat_max,
            lon_min, lon_max,
            tir, size,
            mask
        ])

    return np.array(result, dtype=np.float32)


def load_geodata():
    geodata = Dataset(
        "/gws/ssde/j25b/swift/rt_cores/geoloc_grids/nxny2268_2080_nxnyds164580_blobdx0.04491576_arean41_n27_27_79.nc",
        mmap_mode="r"
    )
    return geodata["lats_mid"], geodata["lons_mid"]

lats, lons = load_geodata()

lats = np.where(lats == -999.999, np.nan, lats)
lons = np.where(lons == -999.999, np.nan, lons)

y_min, y_max = 48, 2062
x_min, x_max = 77, 2262

CONTEXT_LAT_MIN = -35
CONTEXT_LAT_MAX = 24
CONTEXT_LON_MIN = -18
CONTEXT_LON_MAX = 51

def process_file(file_t, nb_x0,
                 CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
                 CONTEXT_LON_MIN, CONTEXT_LON_MAX,
                 lats=lats, lons=lons):

    try:
        with Dataset(file_t, "r") as data_t:
            x0_lat = np.asarray(data_t["Pmax_lat"][:])
            x0_lon = np.asarray(data_t["Pmax_lon"][:])

            if x0_lat.size == 0 or x0_lon.size == 0:
                return None

            t_time = get_time(file_t)
            t_month = int(t_time["month"])
            t_hour = int(t_time["hour"])
            t_minute = int(t_time["minute"])

            month_angle = 2 * np.pi * (t_month - 1) / 12.0
            tod_angle = 2 * np.pi * (t_hour + t_minute / 60.0) / 24.0

            time_features = torch.tensor(
                [
                    np.sin(month_angle), np.cos(month_angle),
                    np.sin(tod_angle), np.cos(tod_angle)
                ],
                dtype=torch.float32
            ).unsqueeze(0).repeat(nb_x0, 1)

            storm_database = create_storm_database(data_t, lats, lons)

            X_features = pad_observed_storms(
                storm_database, nb_x0,
                CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
                CONTEXT_LON_MIN, CONTEXT_LON_MAX
            )

            core_features_np = transform_to_array(X_features)

            if np.isnan(core_features_np).any():
                print(f"NaNs in features for {file_t}")
                return None

            core_features = torch.from_numpy(core_features_np)

            input_tensor = torch.cat([time_features, core_features], dim=1)

            return input_tensor

    except Exception as e:
        print(f"Error processing {file_t}: {e}")
        return None
