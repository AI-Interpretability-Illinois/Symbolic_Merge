#!/bin/bash
#SBATCH --job-name=sym_g27b
#SBATCH --reservation=sup-30781
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodelist=gpua047
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --output=/projects/biro/xiaocong/main_table/delta_jobs/logs/g27b_%j.log
# Third model size: Gemma-2-27B (46 layers, d=4608) split over two A100-40GB, with the Gemma Scope
# 27B residual SAEs (131k features) at the three released layers (10, 22, 34), plus two extra
# sparsities at layer 22. miniF2F Lean 4 -> HOL Light and -> Metamath, as in the other sweeps.
set -e
cd /u/xiaocong/Symbolic_Merge
export PATH=/u/xiaocong/anaconda3/envs/modularity/bin:$PATH PYTHONPATH=/projects/biro/xiaocong/pylib
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
unset SYMBOLIC_MERGE_PILE_DENSITY   # Pile densities index the 9B layer-20 dictionary only
LOCAL=/tmp/sym_g27b_$SLURM_JOB_ID
mkdir -p $LOCAL/sae
trap 'rm -rf $LOCAL' EXIT
cp -r /projects/biro/xiaocong/models/gemma-2-27b $LOCAL/model
S=/projects/biro/xiaocong/sae/gemma-scope-27b-pt-res
for f in layer_10/width_131k/average_l0_106 layer_22/width_131k/average_l0_82 layer_34/width_131k/average_l0_72 \
         layer_22/width_131k/average_l0_48 layer_22/width_131k/average_l0_150; do
  mkdir -p $LOCAL/sae/$f; cp $S/$f/params.npz $LOCAL/sae/$f/
done
echo "copied to $LOCAL: $(du -sh $LOCAL | cut -f1) $(date)"
DATA=/projects/biro/xiaocong/data/formal/minif2f_xsys
M=/projects/biro/xiaocong/main_table/ablation
METRICS=dense,dense_centered,dense_pc1,sae_mean,sae_idf
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
ablate() {  # <config name> <layer> <sae subdir>
  local name=$1 layer=$2 sae=$LOCAL/sae/$3/params.npz
  for task in HOLLight Metamath; do
    O=$M/$task/$name
    [ -f $O/eval/scores.json ] && continue
    for attempt in 1 2; do
      rm -rf $O
      if timeout 2400 python3 xlcost/run_xlcost_sae.py --dataset_file $DATA/$task/all.jsonl --output_dir $O \
             --model_path $LOCAL/model --sae_path $sae --layer $layer --device split --max_gpu_memory 30GiB,34GiB &&
         timeout 900 python3 xlcost/eval_xlcost_sae.py --run_dir $O --output_dir $O/eval --device cuda --metrics $METRICS; then
        echo "done $task $name $(date)"; break
      fi
      echo "attempt $attempt failed or timed out: $task $name $(date)"
    done
  done
}
ablate g27b_L22_w131k 22 layer_22/width_131k/average_l0_82        # middle layer first (22 of 46)
ablate g27b_L10_w131k 10 layer_10/width_131k/average_l0_106
ablate g27b_L34_w131k 34 layer_34/width_131k/average_l0_72
ablate g27b_L22_w131k_l0_48 22 layer_22/width_131k/average_l0_48
ablate g27b_L22_w131k_l0_150 22 layer_22/width_131k/average_l0_150
echo "ALL DONE $(date)"
