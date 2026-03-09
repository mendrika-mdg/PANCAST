import torch

class PreProcessor:
    
    def __init__(self, scaler_path: str):
        scaler = torch.load(scaler_path, map_location="cpu", weights_only=False)

        self.mean = torch.tensor(scaler["mean"], dtype=torch.float32)
        self.scale = torch.tensor(scaler["scale"], dtype=torch.float32)

        # scale columns: lat, lon, lat_min, lat_max, lon_min, lon_max, tir, size
        self.COLS_TO_SCALE = list(range(4, 12))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x feature order:
        0  month_sin
        1  month_cos
        2  tod_sin
        3  tod_cos
        4  lat
        5  lon
        6  lat_min
        7  lat_max
        8  lon_min
        9  lon_max
        10 tir
        11 size
        12 mask

        x shape: (T, N, 13) or (N, 13)
        """

        is_batched = x.dim() == 3

        if not is_batched:
            x = x.unsqueeze(0)

        T, N, F = x.shape

        flat = x.view(T * N, F)
        flat[:, self.COLS_TO_SCALE] = (
            flat[:, self.COLS_TO_SCALE] - self.mean
        ) / self.scale

        x_scaled = flat.view(T, N, F)

        return x_scaled if is_batched else x_scaled.squeeze(0)

    @torch.no_grad()
    def process_single_input(x_raw: torch.Tensor, scaler_path: str) -> torch.Tensor:
        processor = PreProcessor(scaler_path)
        return processor(x_raw)
