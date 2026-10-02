#!/bin/bash
#SBATCH --job-name=sym_llama_tasks
#SBATCH --reservation=sup-30781
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodelist=gpua047
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/projects/biro/xiaocong/main_table/delta_jobs/logs/llama_tasks_%j.log
# Second model family for the main table: Llama-3.1-8B + Llama Scope L15R-32x (131k features)
# on every native-context task except XLCoST (separate job): Common-KG, MultiFarm, Bio-ML,
# Anatomy, Valentine. Model and SAE are staged to node-local disk (cold reads from /work/hdd stall).
set -e
cd /u/xiaocong/Symbolic_Merge
export PATH=/u/xiaocong/anaconda3/envs/modularity/bin:$PATH PYTHONPATH=/projects/biro/xiaocong/pylib
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
unset SYMBOLIC_MERGE_PILE_DENSITY   # Pile densities index Gemma Scope features, not Llama Scope
LOCAL=/tmp/sym_llt_$SLURM_JOB_ID
mkdir -p $LOCAL/sae
trap 'rm -rf $LOCAL' EXIT
cp -r /work/hdd/biro/xiaocong/models/Llama-3.1-8B $LOCAL/model
cp -r /work/hdd/biro/xiaocong/sae/llama-scope-lxr-32x/Llama3_1-8B-Base-L15R-32x $LOCAL/sae/
echo "copied to $LOCAL: $(du -sh $LOCAL | cut -f1) $(date)"
SAE=$LOCAL/sae/Llama3_1-8B-Base-L15R-32x/checkpoints/final.safetensors
LAYER=15
TAG=llama_L15_w131k
M=/projects/biro/xiaocong/main_table
METRICS=dense,dense_centered,dense_pc1,sae_mean,sae_idf
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# store <contexts dir> <output root> <name:context_dir> ... : cache layer 15 once, encode, analyse per query set
store() {
  local ctx=$1 out=$2; shift 2
  if [ ! -f $out/$TAG/results.json ]; then
    python3 bioml/cache_bioml_hidden.py --context_run_dirs $ctx --output_dir $out/hidden_llama \
        --layers $LAYER --contexts 0 --label_last --model_path $LOCAL/model --device cuda
    python3 bioml/encode_bioml_cached.py --cache_dir $out/hidden_llama --layer $LAYER --sae_path $SAE \
        --output_dir $out/$TAG --device cuda
    rm -rf $out/hidden_llama
  fi
  for spec in "$@"; do
    local name=${spec%%:*} qctx=${spec#*:}
    [ -f $out/$TAG/analysis_$name/ranking_summary.csv ] && continue
    python3 bioml/analyze_bioml_idf.py --sae_run_dirs $out/$TAG --context_run_dirs $qctx \
        --background all --fast --lexical --universal_density none --output_dir $out/$TAG/analysis_$name
    echo "done $out/$TAG/analysis_$name $(date)"
  done
}
store $M/commonkg/native/nell_dbpedia  $M/commonkg/native_scored_nell_dbpedia  all:$M/commonkg/native/nell_dbpedia
store $M/commonkg/native/yago_wikidata $M/commonkg/native_scored_yago_wikidata all:$M/commonkg/native/yago_wikidata
store $M/multifarm/native $M/multifarm/native_scored \
      views_cn-en:$M/multifarm/native/views/cn-en views_en-ru:$M/multifarm/native/views/en-ru views_ar-en:$M/multifarm/native/views/ar-en
store $M/anatomy/native $M/anatomy/native_scored all:$M/anatomy/native
store $M/bioml/native $M/bioml/native_scored views_valid:$M/bioml/native/views/valid views_train:$M/bioml/native/views/train

# Valentine, whole-column layout (the main-table layout)
O=$M/valentine_column_llama
if [ ! -f $O/eval/scores.json ]; then
  python3 schema/run_valentine_sae.py --data_root /projects/biro/xiaocong/data/Valentine-datasets --output_dir $O \
      --model_path $LOCAL/model --sae_path $SAE --layer $LAYER --preset column --max_length 1024 --device cuda --overwrite
  python3 schema/eval_valentine_sae.py --run_dir $O --output_dir $O/eval --metrics $METRICS
  echo "done valentine $(date)"
fi
echo "ALL DONE $(date)"
