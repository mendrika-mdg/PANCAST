#!/bin/bash
#SBATCH --job-name=backprod
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/error/%j.err


echo "Nowcast pipeline started at $(date)"
echo "Activating virtual environment..."
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

echo "Running nowcast generation script..."
python /home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/backproduction/backprod_generate_nowcasts_vis.py

echo "Deactivating DL virtual environment..."
deactivate 

echo "Loading Jaspy environment..."
module load jaspy

echo "Generating GeoTIFFs..."
python /home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/backproduction/backprod_generate_geotiffs.py

echo "Nowcast pipeline finished at $(date)"
