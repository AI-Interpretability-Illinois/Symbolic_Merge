#!/bin/bash
#SBATCH --job-name=sym_s1_random
#SBATCH --reservation=sup-30781
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/biro/xiaocong/main_table/delta_jobs/logs/s1_random_%j.log
# S1 of the pre-registered extras (paper/PREREG_extras_2026-10-04.md): a random sparse dictionary with
# the width (131k) and per-token sparsity of the Gemma Scope SAE, on the same layer-20 activations of
# Gemma-2-9B, with the same idf weighting, on all 16 main-table rows. Submit with
#   sbatch --dependency=afterany:<job> jobs/delta/s1_random_dict.sbatch   (SEED=0 by default)
set -e
cd /u/xiaocong/Symbolic_Merge
export PATH=/u/xiaocong/anaconda3/envs/modularity/bin:$PATH PYTHONPATH=/projects/biro/xiaocong/pylib
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
unset SYMBOLIC_MERGE_PILE_DENSITY        # Pile densities index the real dictionary, not a random one
SEED=${SEED:-0}
M=/projects/biro/xiaocong/main_table
DATA=/projects/biro/xiaocong/data
G=/projects/biro/shared/sae/gemma-scope-9b-pt-res-131k/layer_20/width_131k/average_l0_114/params.npz
R=$M/random_dict/seed$SEED
RD=/projects/biro/xiaocong/sae/random/random_w131k_seed$SEED.npz
METRICS=dense,dense_centered,dense_pc1,sae_mean,sae_idf
LOCAL=/tmp/sym_s1_$SLURM_JOB_ID
mkdir -p $LOCAL $R $(dirname $RD)
trap 'rm -rf $LOCAL' EXIT
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# 1. the random dictionary, calibrated to the real SAE's L0 on cached symbol tokens
if [ ! -f $RD ]; then
  python3 tools/random_dictionary.py --out $RD --real_sae $G --layer 20 --seed $SEED --device cuda \
      --calib_caches $M/commonkg/native_scored_nell_dbpedia/hidden_sweep,$M/commonkg/native_scored_yago_wikidata/hidden_sweep,$M/multifarm/native_scored/hidden_sweep
fi
cp $RD $LOCAL/random.npz
cp -r /projects/biro/shared/models/gemma-2-9b $LOCAL/model
echo "staged $(du -sh $LOCAL | cut -f1) $(date)"

run_xl() {  # <dataset file> <output dir>
  [ -f $2/eval/scores.json ] && return
  rm -rf $2
  python3 xlcost/run_xlcost_sae.py --dataset_file "$1" --output_dir $2 --model_path $LOCAL/model --sae_path $LOCAL/random.npz --layer 20 --device cuda
  python3 xlcost/eval_xlcost_sae.py --run_dir $2 --output_dir $2/eval --device cuda --metrics $METRICS
  echo "done $2 $(date)"
}
# 2. miniF2F
for task in Isabelle Metamath HOLLight; do run_xl $DATA/formal/minif2f_xsys/$task/all.jsonl $R/minif2f/lean4_$task; done
# 3. knowledge graphs and cross-lingual terms, from the cached hidden states
kg() {  # <store with hidden_sweep> <name:query-context dir> ...
  local out=$1 run=$1/random_seed$SEED; shift
  [ -f $run/results.json ] || python3 bioml/encode_bioml_cached.py --cache_dir $out/hidden_sweep --layer 20 --sae_path $LOCAL/random.npz --output_dir $run --device cuda
  for spec in "$@"; do
    local name=${spec%%:*} qctx=${spec#*:}
    [ -f $run/analysis_$name/ranking_summary.csv ] && continue
    python3 bioml/analyze_bioml_idf.py --sae_run_dirs $run --context_run_dirs $qctx --background all --fast --universal_density none --output_dir $run/analysis_$name
    echo "done $run/analysis_$name $(date)"
  done
}
kg $M/commonkg/native_scored_nell_dbpedia  all:$M/commonkg/native/nell_dbpedia
kg $M/commonkg/native_scored_yago_wikidata all:$M/commonkg/native/yago_wikidata
kg $M/multifarm/native_scored views_cn-en:$M/multifarm/native/views/cn-en views_en-ru:$M/multifarm/native/views/en-ru views_ar-en:$M/multifarm/native/views/ar-en
# 4. Valentine, whole-column layout as in the main table
V=$R/valentine_column
if [ ! -f $V/eval/scores.json ]; then
  rm -rf $V
  python3 schema/run_valentine_sae.py --data_root $DATA/Valentine-datasets --output_dir $V --model_path $LOCAL/model --sae_path $LOCAL/random.npz \
      --layer 20 --device cuda --preset column --views 16 --max_rows 2000 --max_length 1024
  python3 schema/eval_valentine_sae.py --run_dir $V --output_dir $V/eval --metrics $METRICS
  echo "done valentine $(date)"
fi
# 5. XLCoST, smallest language first
for L in C PHP Java "C++" Python "C#" Javascript; do
  case "$L" in "C++") name=Cpp;; "C#") name=Csharp;; *) name=$L;; esac
  run_xl "$DATA/XLCoST_data/retrieval/code2code_search/program_level/$L/test.jsonl" $R/xlcost/${name}_program
done
echo "ALL DONE $(date)"
