#!/bin/bash
#SBATCH --job-name=sym_llama_isa
#SBATCH --reservation=sup-30781
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodelist=gpua047
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/biro/xiaocong/main_table/delta_jobs/logs/llama_isa_%j.log
# Completes the Llama-3.1-8B main table: miniF2F Lean 4 -> Isabelle (HOL Light and Metamath were
# run in the ablation). Same SAE and layer as the other Llama runs.
set -e
cd /u/xiaocong/Symbolic_Merge
export PATH=/u/xiaocong/anaconda3/envs/modularity/bin:$PATH PYTHONPATH=/projects/biro/xiaocong/pylib
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
unset SYMBOLIC_MERGE_PILE_DENSITY
LOCAL=/tmp/sym_isa_$SLURM_JOB_ID
mkdir -p $LOCAL/sae
trap 'rm -rf $LOCAL' EXIT
cp -r /work/hdd/biro/xiaocong/models/Llama-3.1-8B $LOCAL/model
cp -r /work/hdd/biro/xiaocong/sae/llama-scope-lxr-32x/Llama3_1-8B-Base-L15R-32x $LOCAL/sae/
echo "copied to $LOCAL: $(du -sh $LOCAL | cut -f1) $(date)"
SAE=$LOCAL/sae/Llama3_1-8B-Base-L15R-32x/checkpoints/final.safetensors
O=/projects/biro/xiaocong/main_table/ablation/Isabelle/llama_L15_w131k
if [ ! -f $O/eval/scores.json ]; then
  python3 xlcost/run_xlcost_sae.py --dataset_file /projects/biro/xiaocong/data/formal/minif2f_xsys/Isabelle/all.jsonl \
      --output_dir $O --model_path $LOCAL/model --sae_path $SAE --layer 15 --device cuda
  python3 xlcost/eval_xlcost_sae.py --run_dir $O --output_dir $O/eval --device cuda \
      --metrics dense,dense_centered,dense_pc1,sae_mean,sae_idf
fi
echo "ALL DONE $(date)"
