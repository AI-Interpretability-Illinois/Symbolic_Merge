#!/bin/bash
#SBATCH --job-name=sym_llama_xlcost
#SBATCH --reservation=sup-30781
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodelist=gpua047
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/biro/xiaocong/main_table/delta_jobs/logs/llama_xlcost_%j.log
# Second model family on XLCoST program-level code-to-code search, all seven query languages:
# Llama-3.1-8B + Llama Scope L15R-32x (131k). Official protocol and metrics via eval_xlcost_sae.py.
set -e
cd /u/xiaocong/Symbolic_Merge
export PATH=/u/xiaocong/anaconda3/envs/modularity/bin:$PATH PYTHONPATH=/projects/biro/xiaocong/pylib
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
unset SYMBOLIC_MERGE_PILE_DENSITY
LOCAL=/tmp/sym_llx_$SLURM_JOB_ID
mkdir -p $LOCAL/sae
trap 'rm -rf $LOCAL' EXIT
cp -r /work/hdd/biro/xiaocong/models/Llama-3.1-8B $LOCAL/model
cp -r /work/hdd/biro/xiaocong/sae/llama-scope-lxr-32x/Llama3_1-8B-Base-L15R-32x $LOCAL/sae/
echo "copied to $LOCAL: $(du -sh $LOCAL | cut -f1) $(date)"
SAE=$LOCAL/sae/Llama3_1-8B-Base-L15R-32x/checkpoints/final.safetensors
DATA=/projects/biro/xiaocong/data/XLCoST_data/retrieval/code2code_search/program_level
M=/projects/biro/xiaocong/main_table/xlcost_llama
METRICS=dense,dense_centered,dense_pc1,sae_mean,sae_idf
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for L in C PHP Java "C++" Python "C#" Javascript; do   # smallest first, so partial results arrive early
  case "$L" in "C++") name=Cpp;; "C#") name=Csharp;; *) name=$L;; esac   # output dirs use the names of the Gemma runs
  O=$M/${name}_program
  [ -f $O/eval/scores.json ] && continue
  python3 xlcost/run_xlcost_sae.py --dataset_file "$DATA/$L/test.jsonl" --output_dir $O \
      --model_path $LOCAL/model --sae_path $SAE --layer 15 --device cuda
  python3 xlcost/eval_xlcost_sae.py --run_dir $O --output_dir $O/eval --device cuda --metrics $METRICS
  echo "done $L $(date)"
done
echo "ALL DONE $(date)"
