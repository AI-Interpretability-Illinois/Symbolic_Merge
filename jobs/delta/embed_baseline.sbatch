#!/bin/bash
#SBATCH --job-name=sym_embed
#SBATCH --reservation=sup-30781
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodelist=gpua047
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/biro/xiaocong/main_table/delta_jobs/logs/embed_%j.log
# Off-the-shelf embedding baseline (BAAI/bge-m3) on the same views, candidate pools and metrics
# as the main table, with the same mean-centring and PC1-removal controls.
set -e
cd /u/xiaocong/Symbolic_Merge
export PATH=/u/xiaocong/anaconda3/envs/modularity/bin:$PATH PYTHONPATH=/projects/biro/xiaocong/pylib
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
LOCAL=/tmp/sym_embed_$SLURM_JOB_ID
mkdir -p $LOCAL
trap 'rm -rf $LOCAL' EXIT
cp -r /projects/biro/xiaocong/models/bge-m3 $LOCAL/model
echo "copied to $LOCAL: $(du -sh $LOCAL | cut -f1) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 tools/embed_baseline.py --model_dir $LOCAL/model --output_dir /projects/biro/xiaocong/main_table/embed_baseline/bge-m3 --device cuda
echo "ALL DONE $(date)"
