# Benchmark data

The benchmark data — annotations + videos — is hosted on HuggingFace, not in
this repo.

```bash
huggingface-cli download InternLM/OVO-S-Bench \
    --repo-type dataset --local-dir ./data
```

Layout you should see:

```
data/
├── ovo_s_bench.parquet       # 1695 questions, ~35 MB
└── videos/
    ├── honda/         *.mp4         # L4 trajectory matching source clips
    ├── arkitscenes/   *.mp4
    ├── annotated_videos/            # L4.1 / L4.2 annotated clips
    │   ├── 01_CODa_full/
    │   ├── 04_RoomTour3D/
    │   └── ...
    ├── edited_videos/  *.mp4        # L3.2 edited clips
    ├── Ego4D/         *.mp4
    ├── RoomTour3D/    *.mp4
    ├── CODa_full/     *.mp4
    ├── OmniWorld/     *.mp4
    ├── Sekai/         *.mp4
    ├── VSI-Bench/     *.mp4
    └── YouTube/       *.mp4
```

The release parquet's `video_path` column already includes the source-dir
prefix relative to `videos/`, so no remapping is needed.

## Schema

| column                | type                   | notes                                                              |
| --------------------- | ---------------------- | ------------------------------------------------------------------ |
| `id`                  | string                 | L1-L3: `{subcat}_{idx}` (e.g. `1.1.1_0`); L4: `{main}_{NNN}`        |
| `source_dataset`      | string                 | Ego4D / RoomTour3D / CODa_full / OmniWorld / VSI-Bench / Sekai / YouTube / Honda / ARKitScenes |
| `video_id`            | string                 | Stable across releases                                             |
| `video_path`          | string                 | Relative to `videos/`                                              |
| `level`               | int (1-4)              |                                                                    |
| `task_main_category`  | string (e.g. `1.1`)    |                                                                    |
| `task_subcategory`    | string                 | L1-L3: dotted (`1.1.1`); L4: main_cat (`4.1`)                       |
| `task_type_name`      | string                 | Human-readable task name                                            |
| `question`            | string                 | English-translated                                                  |
| `options`             | list<[label, content]> | 2-7 options. L4.3 contents are `data:image/png;base64,...` URIs    |
| `query_times`         | list<float>            | Seconds from video start                                            |
| `evidence_times`      | list<[float, float]>   | Each evidence interval `[start, end]` in seconds                    |
| `answers`             | list<string>           | Correct option label(s)                                             |
