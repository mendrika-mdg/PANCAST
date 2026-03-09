#!/bin/bash
#SBATCH --job-name=pancast-live
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --account=wiser-ewsa
#SBATCH -o /work/scratch-nopw2/mendrika/PANCAST/submission_logs/output/%j.out
#SBATCH -e /work/scratch-nopw2/mendrika/PANCAST/submission_logs/error/%j.err

set -e

LOCKFILE=/home/users/mendrika/Object-Based-LSTMConv/pancast_live.lock

if [ -f "$LOCKFILE" ]; then
    echo "Another PANCAST job is running. Exiting."
    exit 1
fi

touch $LOCKFILE
trap "rm -f $LOCKFILE" EXIT

echo "Nowcast pipeline started at $(date)"
echo "Activating virtual environment..."
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

echo "Running nowcast generation script..."
python /home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/data-preparation/run_nowcast/generate_nowcasts_vis.py

echo "Deactivating DL virtual environment..."
deactivate 

source /etc/profile
echo "Loading Jaspy environment..."
module load jaspy

echo "Generating GeoTIFFs..."
python /home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/data-preparation/run_nowcast/generate_geotiffs_vis.py

echo "Nowcast pipeline finished at $(date)"