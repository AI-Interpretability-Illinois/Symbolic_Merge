# Paper: Rare Features, Shared Symbols (NAACL draft)

`main.tex` is the ACL-template draft (review mode: anonymous, line numbers).

Build (XeTeX via Tectonic; renders the Chinese/Arabic examples when a CJK/Arabic font is installed, otherwise falls back to romanisation):

    /u/xiaocong/anaconda3/envs/latex/bin/tectonic -X compile main.tex

or with a standard TeX Live: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

Overleaf: upload the zip as a new project (root file `main.tex`). Set the compiler to **XeLaTeX** to get the
Chinese/Arabic glyphs (Noto fonts are installed there); pdfLaTeX also compiles, with those terms romanised.

Regenerate the numbers and figures from the result files snapshotted in `data/`:

    python3 make_tables.py            # tables/*.tex (every number in the tables is copied by code)
    cd figures && python3 fig_pipeline.py && python3 fig_main.py && python3 fig_ablation.py && python3 fig_interp.py
    python3 significance.py           # data/significance.csv + tables/tab_significance.tex (paired bootstrap)
    python3 collect_delta_results.py  # Llama main table, KG/MultiFarm sweep, seed study (from main_table/)
    cd figures && python3 fig_sweep_kg.py
    python3 render_pages.py           # preview/ PNGs of the compiled PDF

`data/` holds copies of `main_table/summary/all_results.csv`, `ablation_minif2f.csv`, the interpretability
summary, the Valentine column-type split and the XLCoST merge test (from /projects/biro/xiaocong/main_table).

`references.bib`: every entry was AI-generated and then checked against the source named in its
comment; the `% --- AI-GENERATED REFERENCE (VERIFY) ---` delimiters mark them for a final human check.
