#!/bin/bash
#SBATCH --job-name=sym_s5_libidf
#SBATCH --reservation=sup-30781
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/projects/biro/xiaocong/main_table/delta_jobs/logs/s5_libidf_%j.log
# S5 of the pre-registered extras: idf from a held-out same-system library instead of the test candidate
# pool. Encodes 3,000 sampled set.mm theorems and 3,000 HOL Light library terms with the main-table model
# and SAE (formal/make_library.py wrote them in the XLCoST row format), then re-scores the miniF2F
# Lean 4 -> Metamath and -> HOL Light runs with the library as the reference collection.
set -e
cd /u/xiaocong/Symbolic_Merge
export PATH=/u/xiaocong/anaconda3/envs/modularity/bin:$PATH PYTHONPATH=/projects/biro/xiaocong/pylib
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export SYMBOLIC_MERGE_PILE_DENSITY=/projects/biro/xiaocong/pile_density_l0_114.json
M=/projects/biro/xiaocong/main_table
LIB=/projects/biro/xiaocong/data/formal/libraries
SAE=/projects/biro/shared/sae/gemma-scope-9b-pt-res-131k/layer_20/width_131k/average_l0_114/params.npz
OUT=$M/library_idf
METRICS=dense,dense_centered,dense_pc1,sae_mean,sae_idf
LOCAL=/tmp/sym_s5_$SLURM_JOB_ID
mkdir -p $LOCAL $OUT
trap 'rm -rf $LOCAL' EXIT
cp -r /projects/biro/shared/models/gemma-2-9b $LOCAL/model
cp $SAE $LOCAL/sae.npz
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
for sys in Metamath HOLLight; do
  R=$OUT/library_$sys
  if [ ! -d $R/representations/candidate ]; then
    rm -rf $R
    python3 xlcost/run_xlcost_sae.py --dataset_file $LIB/$sys/library.jsonl --output_dir $R --model_path $LOCAL/model --sae_path $LOCAL/sae.npz --layer 20 --device cuda
    rm -rf $R/representations/query      # the two sides are the same statements; keep one
    echo "encoded library $sys $(date)"
  fi
  # re-score the main-table run with the library as the reference collection
  python3 xlcost/eval_xlcost_sae.py --run_dir $M/minif2f/lean4_$sys --background_run $R --output_dir $OUT/lean4_${sys}_libidf --device cuda --metrics $METRICS
  echo "done $sys $(date)"
done
echo "ALL DONE $(date)"
