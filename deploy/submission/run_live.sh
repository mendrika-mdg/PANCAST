#!/bin/bash

sleep 5

echo "Nowcast pipeline started at $(date)"
echo "Activating virtual environment..."
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

echo "Running nowcast generation script..."
python /home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/data-preparation/run_nowcast/generate_nowcasts_vis.py

echo "Deactivating DL virtual environment..."
deactivate 

echo "Loading Jaspy environment..."
module load jaspy

echo "Generating GeoTIFFs..."
python /home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/data-preparation/run_nowcast/generate_geotiffs_vis.py

echo "Nowcast pipeline finished at $(date)"
