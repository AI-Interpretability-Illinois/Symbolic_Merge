#!/bin/bash
#SBATCH --job-name=sym_seeds
#SBATCH --reservation=sup-30781
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodelist=gpua047
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/projects/biro/xiaocong/main_table/delta_jobs/logs/seeds_%j.log
# Seed study for the only stochastic step, the sampling of views: Common-KG classes (16 of their
# instances) with three extra seeds, and Valentine columns (16 of their cell values) with two extra
# seeds. Gemma-2-9B, layer 20, Gemma Scope 131k (L0 114), i.e. the main-table setting.
set -e
cd /u/xiaocong/Symbolic_Merge
export PATH=/u/xiaocong/anaconda3/envs/modularity/bin:$PATH PYTHONPATH=/projects/biro/xiaocong/pylib
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export SYMBOLIC_MERGE_PILE_DENSITY=/projects/biro/xiaocong/pile_density_l0_114.json
LOCAL=/tmp/sym_seed_$SLURM_JOB_ID
mkdir -p $LOCAL
trap 'rm -rf $LOCAL' EXIT
cp -r /projects/biro/shared/models/gemma-2-9b $LOCAL/model
cp /projects/biro/shared/sae/gemma-scope-9b-pt-res-131k/layer_20/width_131k/average_l0_114/params.npz $LOCAL/sae.npz
echo "copied to $LOCAL: $(du -sh $LOCAL | cut -f1) $(date)"
M=/projects/biro/xiaocong/main_table
METRICS=dense,dense_centered,dense_pc1,sae_mean,sae_idf
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for seed in 1 2 3; do
  V=$M/commonkg/native_seed$seed
  [ -d $V/nell_dbpedia ] || python3 native/build_native_views.py commonkg --data_dir /projects/biro/xiaocong/data/commonkg --output_dir $V --seed $seed
  for c in nell_dbpedia yago_wikidata; do
    out=$M/commonkg/seed_study/seed${seed}_$c
    [ -f $out/analysis/ranking_summary.csv ] && continue
    python3 bioml/cache_bioml_hidden.py --context_run_dirs $V/$c --output_dir $out/hidden \
        --layers 20 --contexts 0 --label_last --model_path $LOCAL/model --device cuda
    python3 bioml/encode_bioml_cached.py --cache_dir $out/hidden --layer 20 --sae_path $LOCAL/sae.npz \
        --output_dir $out --device cuda
    python3 bioml/analyze_bioml_idf.py --sae_run_dirs $out --context_run_dirs $V/$c \
        --background all --fast --output_dir $out/analysis
    rm -rf $out/hidden
    echo "done commonkg seed $seed $c $(date)"
  done
done

for seed in 7 13; do
  O=$M/valentine_column_seed$seed
  [ -f $O/eval/scores.json ] && continue
  python3 schema/run_valentine_sae.py --data_root /projects/biro/xiaocong/data/Valentine-datasets --output_dir $O \
      --model_path $LOCAL/model --sae_path $LOCAL/sae.npz --layer 20 --preset column --max_length 1024 \
      --seed $seed --device cuda --overwrite
  python3 schema/eval_valentine_sae.py --run_dir $O --output_dir $O/eval --metrics $METRICS
  echo "done valentine seed $seed $(date)"
done
echo "ALL DONE $(date)"
