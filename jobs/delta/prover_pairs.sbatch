#!/bin/bash
#SBATCH --job-name=sym_pairs
#SBATCH --reservation=sup-30781
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodelist=gpua047
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/biro/xiaocong/main_table/delta_jobs/logs/pairs_%j.log
# All ordered pairs among the four miniF2F provers (the nine not yet run; Lean 4 -> X exist), with
# Gemma-2-9B layer 20 and the main-table SAE, so the merge / transitivity test can run over formal
# systems. Then the XLCoST exclude-self (no free hit) re-scoring for the two languages whose Gemma
# representations are on /projects (Java, C++); the other five are scored on timan108.
set -e
cd /u/xiaocong/Symbolic_Merge
export PATH=/u/xiaocong/anaconda3/envs/modularity/bin:$PATH PYTHONPATH=/projects/biro/xiaocong/pylib
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export SYMBOLIC_MERGE_PILE_DENSITY=/projects/biro/xiaocong/pile_density_l0_114.json
LOCAL=/tmp/sym_pairs_$SLURM_JOB_ID
mkdir -p $LOCAL
trap 'rm -rf $LOCAL' EXIT
cp -r /projects/biro/shared/models/gemma-2-9b $LOCAL/model
cp /projects/biro/shared/sae/gemma-scope-9b-pt-res-131k/layer_20/width_131k/average_l0_114/params.npz $LOCAL/sae.npz
echo "copied to $LOCAL: $(du -sh $LOCAL | cut -f1) $(date)"
DATA=/projects/biro/xiaocong/data/formal/minif2f_pairs
M=/projects/biro/xiaocong/main_table/minif2f_pairs
METRICS=dense,dense_centered,dense_pc1,sae_mean,sae_idf
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for pair in $(ls $DATA); do
  O=$M/$pair
  [ -f $O/eval/scores.json ] && continue
  python3 xlcost/run_xlcost_sae.py --dataset_file $DATA/$pair/all.jsonl --output_dir $O \
      --model_path $LOCAL/model --sae_path $LOCAL/sae.npz --layer 20 --device cuda
  python3 xlcost/eval_xlcost_sae.py --run_dir $O --output_dir $O/eval --device cuda --metrics $METRICS
  echo "done $pair $(date)"
done
# XLCoST exclude-self protocol on the runs kept on /projects
for L in Java Cpp; do
  R=/projects/biro/xiaocong/main_table/xlcost/${L}_program
  [ -f $R/eval_exclude_self/scores.json ] && continue
  python3 xlcost/eval_xlcost_sae.py --run_dir $R --output_dir $R/eval_exclude_self --device cuda --metrics $METRICS --exclude_self
  echo "done exclude-self $L $(date)"
done
echo "ALL DONE $(date)"
