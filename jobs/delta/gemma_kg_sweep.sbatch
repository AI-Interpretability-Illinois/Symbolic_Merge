#!/bin/bash
#SBATCH --job-name=sym_kg_sweep
#SBATCH --reservation=sup-30781
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodelist=gpua047
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/biro/xiaocong/main_table/delta_jobs/logs/kg_sweep_%j.log
# Layer and SAE-width ablation beyond formal mathematics: Gemma-2-9B on Common-KG (NELL->DBpedia,
# YAGO->Wikidata) and MultiFarm (zh/ru/ar->en) with native views. One model pass caches ten layers
# per store; then one encode + analysis per (layer, SAE). Widths at layer 20 use each width's
# canonical release (262k and the matched-L0 16k/1M files are not on Delta and /work/hdd is full).
set -e
cd /u/xiaocong/Symbolic_Merge
export PATH=/u/xiaocong/anaconda3/envs/modularity/bin:$PATH PYTHONPATH=/projects/biro/xiaocong/pylib
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export SYMBOLIC_MERGE_PILE_DENSITY=/projects/biro/xiaocong/pile_density_l0_114.json
LOCAL=/tmp/sym_kgs_$SLURM_JOB_ID
mkdir -p $LOCAL/sae
trap 'rm -rf $LOCAL' EXIT
cp -r /projects/biro/shared/models/gemma-2-9b $LOCAL/model
G=/projects/biro/shared/sae/gemma-scope-9b-pt-res-131k
W=/work/hdd/biro/xiaocong/sae/gemma-scope-9b-pt-res
declare -A SAEP
# the same SAE files as the miniF2F layer/width sweep (ablation_minif2f.csv), so the axes are comparable
stage() {  # <config> <source params.npz>
  mkdir -p $LOCAL/sae/$1; cp $2 $LOCAL/sae/$1/params.npz; SAEP[$1]=$LOCAL/sae/$1/params.npz
}
stage L5_w131k  $G/layer_5/width_131k/average_l0_94/params.npz
stage L9_w131k  $W/layer_9/width_131k/average_l0_75/params.npz
stage L10_w131k $G/layer_10/width_131k/average_l0_84/params.npz
stage L15_w131k $G/layer_15/width_131k/average_l0_103/params.npz
stage L20_w131k $G/layer_20/width_131k/average_l0_114/params.npz
stage L25_w131k $G/layer_25/width_131k/average_l0_96/params.npz
stage L30_w131k $G/layer_30/width_131k/average_l0_95/params.npz
stage L31_w131k $W/layer_31/width_131k/average_l0_92/params.npz
stage L35_w131k $G/layer_35/width_131k/average_l0_94/params.npz
stage L40_w131k $G/layer_40/width_131k/average_l0_94/params.npz
stage L20_w16k  $W/layer_20/width_16k/average_l0_68/params.npz
stage L20_w32k  $W/layer_20/width_32k/average_l0_57/params.npz
stage L20_w65k  $W/layer_20/width_65k/average_l0_55/params.npz
stage L20_w524k $W/layer_20/width_524k/average_l0_48/params.npz
stage L20_w1m   $W/layer_20/width_1m/average_l0_101/params.npz
echo "copied to $LOCAL: $(du -sh $LOCAL | cut -f1) $(date)"
M=/projects/biro/xiaocong/main_table
LAYERS=5,9,10,15,20,25,30,31,35,40
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# store <contexts dir> <output root> <name:context_dir> ...
store() {
  local ctx=$1 out=$2; shift 2
  if [ ! -f $out/hidden_sweep/done ]; then
    python3 bioml/cache_bioml_hidden.py --context_run_dirs $ctx --output_dir $out/hidden_sweep \
        --layers $LAYERS --contexts 0 --label_last --model_path $LOCAL/model --device cuda
    touch $out/hidden_sweep/done
  fi
  for cfg in "${!SAEP[@]}"; do
    local layer=${cfg#L}; layer=${layer%%_*}
    local run=$out/sweep/$cfg
    if [ ! -f $run/results.json ]; then
      python3 bioml/encode_bioml_cached.py --cache_dir $out/hidden_sweep --layer $layer --sae_path ${SAEP[$cfg]} \
          --output_dir $run --device cuda
    fi
    local dens="--universal_density none"; [ "$cfg" = L20_w131k ] && dens=""
    for spec in "$@"; do
      local name=${spec%%:*} qctx=${spec#*:}
      [ -f $run/analysis_$name/ranking_summary.csv ] && continue
      python3 bioml/analyze_bioml_idf.py --sae_run_dirs $run --context_run_dirs $qctx \
          --background all --fast $dens --output_dir $run/analysis_$name
      echo "done $run/analysis_$name $(date)"
    done
    rm -rf $run/representations   # keep only the scores; the cache can regenerate them
  done
}
store $M/commonkg/native/nell_dbpedia  $M/commonkg/native_scored_nell_dbpedia  all:$M/commonkg/native/nell_dbpedia
store $M/commonkg/native/yago_wikidata $M/commonkg/native_scored_yago_wikidata all:$M/commonkg/native/yago_wikidata
store $M/multifarm/native $M/multifarm/native_scored \
      views_cn-en:$M/multifarm/native/views/cn-en views_en-ru:$M/multifarm/native/views/en-ru views_ar-en:$M/multifarm/native/views/ar-en
echo "ALL DONE $(date)"
