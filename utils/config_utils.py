"""Resolve the nested MODELS config into a flat {model_name: config} dict."""


def resolve_all_models(config):
    """Walk the nested MODELS structure and produce a flat dict.

    Handles three entry shapes inside each category:
      1. Standalone model  – dict with no 'variants' key  → name is the entry key
      2. Series with variants – dict with 'defaults'/'variants' → name is '{series}-{variant}'
         Special case: variant named 'base' is also aliased as just '{series}'

    Each resolved config gets two extra fields:
      - category: the top-level category key (e.g. 'api-model')
      - series:   the series key or model name for standalones
    """
    models_section = config.get("MODELS", {})
    resolved = {}

    for category, series_dict in models_section.items():
        if not isinstance(series_dict, dict):
            continue

        for series_name, series_config in series_dict.items():
            if not isinstance(series_config, dict):
                continue

            if "variants" in series_config:
                # Series with defaults + variants
                defaults = dict(series_config.get("defaults", {}))
                for variant_name, variant_overrides in series_config["variants"].items():
                    merged = {**defaults, **(variant_overrides or {})}
                    merged["category"] = category
                    merged["series"] = series_name

                    model_name = f"{series_name}-{variant_name}"
                    resolved[model_name] = merged

                    # 'base' variant is also accessible as just the series name
                    if variant_name == "base":
                        resolved[series_name] = merged
            else:
                # Standalone model (no variants key)
                standalone = {k: v for k, v in series_config.items() if k != "defaults"}
                standalone["category"] = category
                standalone["series"] = series_name
                resolved[series_name] = standalone

    return resolved
