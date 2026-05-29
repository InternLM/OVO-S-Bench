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
  images/
    teaser.png          # Fig. 1 (taxonomy levels overview)
    taxonomy_levels.png # Fig. 3 (four-level taxonomy diagram)
    statistics.png      # benchmark statistics
    construction.png    # construction pipeline
    logo.png            # favicon
  examples/
    taxonomy_examples-1.png
    appendix_examples-01..10.png  # carousel slides
```

## Section plan and status

| Section            | Status        | Notes                                                  |
| ------------------ | ------------- | ------------------------------------------------------ |
| Hero / title       | done          | arXiv / PDF / dataset buttons stubbed (`#`, "soon")    |
| What's New         | done          | Single 2026-05-29 entry; update on each release        |
| Teaser             | done          | `figs/generated/teaser_ovo_s_bench-1.png`              |
| Abstract           | done          | Verbatim from paper Sec. 0                             |
| Four-level taxonomy| done          | L1–L4 cards with task families                         |
| Statistics         | done          | Render of benchmark statistics figure                  |
| Construction       | done          | Pipeline figure + caption                              |
| Key findings       | done          | 4 highlights from Sec. 5                               |
| Leaderboard        | **partial**   | Static HTML table copied from paper Tab. 2; needs a    |
|                    |               | sortable widget and a submission portal once dataset   |
|                    |               | is live                                                |
| Examples carousel  | done          | 11 slides (1 taxonomy + 10 appendix examples)          |
| BibTeX             | placeholder   | Replace with arXiv eprint once posted                  |

## TODO before public launch

- [ ] Replace `arXiv` / `Paper` / `Dataset` links with real URLs.
- [ ] Add `.nojekyll` if any underscore-prefixed paths sneak in (none today).
- [ ] Swap the BibTeX placeholder for the arXiv entry.
- [ ] Consider adding: per-level breakdown plots, error-mode (CoT failure) figure,
      and a sortable/filterable leaderboard backed by a JSON file.
- [ ] Add a banner / carousel of representative video clips (the paper does not
      yet ship per-question MP4 thumbnails — generate after dataset release).
- [ ] Verify author Google-Scholar links (some are placeholders).
