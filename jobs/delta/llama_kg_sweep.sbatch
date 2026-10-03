#!/bin/bash
#SBATCH --job-name=sym_llama_kg
#SBATCH --reservation=sup-30781
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodelist=gpua047
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/biro/xiaocong/main_table/delta_jobs/logs/llama_kg_%j.log
# Is Llama's tie with PC1-removed dense on knowledge-graph and cross-lingual terms a layer choice?
# Llama-3.1-8B layer sweep on Common-KG and MultiFarm with the Llama Scope residual SAEs
# (131k at layers 8/12/15/20/24, 32k at 15), native views, one model pass per store caching five layers.
set -e
cd /u/xiaocong/Symbolic_Merge
export PATH=/u/xiaocong/anaconda3/envs/modularity/bin:$PATH PYTHONPATH=/projects/biro/xiaocong/pylib
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
unset SYMBOLIC_MERGE_PILE_DENSITY
LOCAL=/tmp/sym_llkg_$SLURM_JOB_ID
mkdir -p $LOCAL/sae
trap 'rm -rf $LOCAL' EXIT
cp -r /work/hdd/biro/xiaocong/models/Llama-3.1-8B $LOCAL/model
W=/work/hdd/biro/xiaocong/sae
declare -A SAEP
for L in 8 12 15 20 24; do
  cp -r $W/llama-scope-lxr-32x/Llama3_1-8B-Base-L${L}R-32x $LOCAL/sae/; SAEP[llama_L${L}_w131k]=$LOCAL/sae/Llama3_1-8B-Base-L${L}R-32x/checkpoints/final.safetensors
done
cp -r $W/llama-scope-lxr-8x/Llama3_1-8B-Base-L15R-8x $LOCAL/sae/; SAEP[llama_L15_w32k]=$LOCAL/sae/Llama3_1-8B-Base-L15R-8x/checkpoints/final.safetensors
echo "copied to $LOCAL: $(du -sh $LOCAL | cut -f1) $(date)"
M=/projects/biro/xiaocong/main_table
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

store() {  # <contexts dir> <output root> <name:context_dir> ...
  local ctx=$1 out=$2; shift 2
  if [ ! -f $out/hidden_llama_sweep/done ]; then
    python3 bioml/cache_bioml_hidden.py --context_run_dirs $ctx --output_dir $out/hidden_llama_sweep \
        --layers 8,12,15,20,24 --contexts 0 --label_last --model_path $LOCAL/model --device cuda
    touch $out/hidden_llama_sweep/done
  fi
  for cfg in "${!SAEP[@]}"; do
    local layer=${cfg#llama_L}; layer=${layer%%_*}
    local run=$out/sweep/$cfg
    if [ ! -f $run/results.json ]; then
      python3 bioml/encode_bioml_cached.py --cache_dir $out/hidden_llama_sweep --layer $layer --sae_path ${SAEP[$cfg]} \
          --output_dir $run --device cuda
    fi
    for spec in "$@"; do
      local name=${spec%%:*} qctx=${spec#*:}
      [ -f $run/analysis_$name/ranking_summary.csv ] && continue
      python3 bioml/analyze_bioml_idf.py --sae_run_dirs $run --context_run_dirs $qctx \
          --background all --fast --universal_density none --output_dir $run/analysis_$name
      echo "done $run/analysis_$name $(date)"
    done
    rm -rf $run/representations
  done
  rm -rf $out/hidden_llama_sweep
}
store $M/commonkg/native/nell_dbpedia  $M/commonkg/native_scored_nell_dbpedia  all:$M/commonkg/native/nell_dbpedia
store $M/commonkg/native/yago_wikidata $M/commonkg/native_scored_yago_wikidata all:$M/commonkg/native/yago_wikidata
store $M/multifarm/native $M/multifarm/native_scored \
      views_cn-en:$M/multifarm/native/views/cn-en views_en-ru:$M/multifarm/native/views/en-ru views_ar-en:$M/multifarm/native/views/ar-en
echo "ALL DONE $(date)"
