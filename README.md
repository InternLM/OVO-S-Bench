# OVO-S-Bench Project Page (`webpage` branch)

This orphan branch hosts the static project page for **OVO-S-Bench: A Hierarchical
Benchmark for Streaming Spatial Intelligence in Multimodal LLMs**.

## Local preview

```bash
python3 -m http.server -d . 8000
# open http://localhost:8000
```

## Deploy

Push this branch to GitHub and configure GitHub Pages:

```
Settings → Pages → Branch: webpage → Folder: / (root)
```

The page will be served at `https://internlm.github.io/OVO-S-Bench/`.

## Layout

```
index.html              # single-page site
static/
  css/                  # Bulma + custom (carried from OVO-Bench template)
  js/                   # carousel + slider scripts
  images/               # 300 dpi PNGs rasterized from the paper's PDFs
    teaser.png          # paper Fig. 1 — 1_Teaser.pdf
    taxonomy_statistics.png  # paper Fig. 3 — 3_Taxonomy_Statistics_Overview.pdf
    cot_failures.png    # paper Fig. 4 — generated/cot_failures.pdf
    fig_compression.png # paper Fig. 5 — generated/fig_compression.pdf
    scaling_curves.png  # appendix — generated/scaling_curves.pdf
    per_source_heatmap.png  # appendix — per_source_accuracy_heatmap_top12_reported.pdf
    logo.png            # favicon
  examples/             # 300 dpi PNG carousel slides
    taxonomy_examples-1.png        # paper Fig. 2 — 3_Taxonomy_Examples.pdf
    appendix_examples-01..10.png   # appendix_examples.pdf (10 pages)
```

All figures are rasterized at **300 dpi** with `pdftoppm` from the canonical
PDFs in `6a1629f975a090a05c2cb571/figs/`. Earlier draft PNGs (`teaser_ovo_s_bench-1`,
`taxonomy_levels-1`, `benchmark_statistics`, `construction_pipeline`, etc.) are
**not** used — the paper never references them.

## Section plan and status

| Section            | Status        | Source / Notes                                         |
| ------------------ | ------------- | ------------------------------------------------------ |
| Hero / title       | done          | arXiv / PDF / dataset buttons stubbed (`#`, "soon")    |
| What's New         | done          | Single 2026-06-01 entry                                |
| Teaser             | done          | paper Fig. 1                                           |
| Abstract           | done          | Verbatim from `sec/0_Abstract.tex`                     |
| Four-level taxonomy| done          | paper Fig. 3 (combined taxonomy + statistics)          |
| Construction       | done          | Text-only (paper has no construction figure)           |
| Key findings       | done          | 5 highlights from Sec. 5                               |
| Leaderboard        | partial       | Tab. 2 transcribed as static HTML; sortable widget +   |
|                    |               | submission portal pending dataset release              |
| Analysis           | done          | CoT failures (Fig. 4) + compression diagnostics (Fig. 5)|
| Examples carousel  | done          | Taxonomy examples (Fig. 2) + 10 appendix examples      |
| BibTeX             | placeholder   | Replace with arXiv entry once posted                   |

## TODO before public launch

- [ ] Replace `arXiv` / `Paper` / `Dataset` links with real URLs.
- [ ] Swap the BibTeX placeholder for the arXiv entry.
- [ ] Verify author Google-Scholar links (Pengyiang Liu link missing).
- [ ] Optional: add sortable / filterable leaderboard backed by JSON.
- [ ] Optional: add scaling-curve and per-source-heatmap figures to the
      Analysis section once the appendix is referenced from the page.
