#!/bin/bash
#SBATCH --job-name=sym_llama_abl2
#SBATCH --reservation=sup-30781
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodelist=gpua047
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/biro/xiaocong/main_table/ablation/logs/llama_%j.log
# Second model family: Llama-3.1-8B + Llama Scope residual SAEs (LXR), miniF2F HOL Light + Metamath.
# v2: the first job deadlocked loading an SAE straight from /work/hdd, so this copies the
# model and SAEs to node-local disk once, bounds every run with a timeout, and retries once.
set -e
cd /u/xiaocong/Symbolic_Merge
export PATH=/u/xiaocong/anaconda3/envs/modularity/bin:$PATH PYTHONPATH=/projects/biro/xiaocong/pylib
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
LOCAL=/tmp/sym_llama_$SLURM_JOB_ID
mkdir -p $LOCAL/sae
trap 'rm -rf $LOCAL' EXIT
cp -r /work/hdd/biro/xiaocong/models/Llama-3.1-8B $LOCAL/model
for d in /work/hdd/biro/xiaocong/sae/llama-scope-lxr-8x/Llama3_1-8B-Base-L15R-8x \
         /work/hdd/biro/xiaocong/sae/llama-scope-lxr-32x/Llama3_1-8B-Base-L{8,12,15,20,24}R-32x; do
  cp -r $d $LOCAL/sae/
done
echo "copied to $LOCAL: $(du -sh $LOCAL | cut -f1) $(date)"
DATA=/projects/biro/xiaocong/data/formal/minif2f_xsys
M=/projects/biro/xiaocong/main_table/ablation
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ablate() {  # <config name> <layer> <sae dir name>
  local name=$1 layer=$2 sae=$LOCAL/sae/$3/checkpoints/final.safetensors
  for task in HOLLight Metamath; do
    O=$M/$task/$name
    [ -f $O/eval/scores.json ] && continue
    for attempt in 1 2; do
      rm -rf $O
      if timeout 1500 python3 xlcost/run_xlcost_sae.py --dataset_file $DATA/$task/all.jsonl --output_dir $O \
             --model_path $LOCAL/model --sae_path $sae --layer $layer --device cuda &&
         timeout 900 python3 xlcost/eval_xlcost_sae.py --run_dir $O --output_dir $O/eval --device cuda \
             --metrics dense,dense_centered,dense_pc1,sae_mean,sae_idf; then
        echo "done $task $name $(date)"; break
      fi
      echo "attempt $attempt failed or timed out: $task $name $(date)"
    done
  done
}
# layer 15 of 32 is the depth closest to Gemma-2-9B layer 20 of 42; 32x = 131,072 features
ablate llama_L15_w131k 15 Llama3_1-8B-Base-L15R-32x
ablate llama_L15_w32k 15 Llama3_1-8B-Base-L15R-8x
for L in 8 12 20 24; do
  ablate llama_L${L}_w131k $L Llama3_1-8B-Base-L${L}R-32x
done
