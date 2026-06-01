"""
Cambrian-S offline inference for OVO-S evaluation.
Supports both standard Cambrian-S and Cambrian-S-LFP variants.

Requires:
    - Cambrian-S source code at _src/cambrian-s/
    - transformers==4.37.0, decord, torch, flash-attn
"""

import math
import sys
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from PIL import Image

from ..base import BaseModel, resolve_runtime_path

# Add Cambrian-S source to path
_CAMBRIAN_SRC = str(Path(__file__).resolve().parent.parent.parent / "_src" / "cambrian-s")
if _CAMBRIAN_SRC not in sys.path:
    sys.path.insert(0, _CAMBRIAN_SRC)

# Add lmms-eval dir for qwen2_monkey_patch
_LMMS_EVAL_DIR = str(Path(__file__).resolve().parent.parent.parent / "_src" / "cambrian-s" / "lmms-eval")
if _LMMS_EVAL_DIR not in sys.path:
    sys.path.insert(0, _LMMS_EVAL_DIR)


class CambrianModel(BaseModel):
    """Offline inference for Cambrian-S (Qwen2.5 + SigLIP-2)."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.local_path = resolve_runtime_path(config.get("local_path", config.get("model_id")))
        self.conv_template = config.get("conv_template", "qwen_2")
        self.video_max_frames = config.get("video_max_frames", 128)
        self.video_fps_cfg = config.get("video_fps", 1)
        self.miv_token_len = config.get("miv_token_len", 64)
        self.si_token_len = config.get("si_token_len", 729)
        self.image_aspect_ratio = config.get("image_aspect_ratio", "anyres")
        self.anyres_max_subimages = config.get("anyres_max_subimages", 9)
        self.num_beams = config.get("num_beams", 1)
        self.top_p = config.get("top_p", None)

        self._model = None
        self._tokenizer = None
        self._image_processor = None

    def _init_model(self):
        """Lazy initialization of Cambrian-S model."""
        if self._model is not None:
            return

        from cambrian.model.builder import load_pretrained_model
        from cambrian.mm_utils import get_model_name_from_path
        from cambrian.model.language_model.cambrian_qwen2 import CambrianQwenConfig

        print(f"Loading Cambrian-S from: {self.local_path}")
        model_name = get_model_name_from_path(self.local_path)
        model_config = CambrianQwenConfig.from_pretrained(self.local_path)
        for attr in ("mm_vision_tower_aux_list", "vision_tower_aux_list"):
            paths = getattr(model_config, attr, None)
            if paths:
                resolved = [resolve_runtime_path(p) for p in paths]
                if resolved != list(paths):
                    print(f"[PATH_RESOLVE] {attr}: {paths} -> {resolved}")
                    setattr(model_config, attr, resolved)
        self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(
            self.local_path, None, model_name, device_map="cuda:0", config=model_config
        )

        # Configure video/image processing parameters
        self._model.config.video_max_frames = self.video_max_frames
        self._model.config.video_fps = self.video_fps_cfg
        self._model.config.video_force_sample = False
        self._model.config.add_time_instruction = False
        self._model.config.miv_token_len = self.miv_token_len
        self._model.config.si_token_len = self.si_token_len
        self._model.config.image_aspect_ratio = self.image_aspect_ratio
        self._model.config.anyres_max_subimages = self.anyres_max_subimages
        print("Cambrian-S loaded successfully.")

    def _process_frames_as_video(self, frames: List[Image.Image]):
        """Process PIL frames through Cambrian vision pipeline as video frames."""
        from cambrian.mm_utils import expand2square

        processor_aux_list = self._image_processor
        video_aux_list = []

        for processor_aux in processor_aux_list:
            mean_bg = tuple(int(x * 255) for x in processor_aux.image_mean)
            per_frame = []
            for f in frames:
                squared = expand2square(f, mean_bg)
                # base_encoder.ProcessorWrapper wraps a torchvision Compose that
                # handles a single PIL image; preprocess returns {"pixel_values": [tensor]}.
                pv = processor_aux.preprocess(squared, return_tensors="pt")["pixel_values"][0]
                per_frame.append(pv)
            pixel_values = torch.stack(per_frame, dim=0)  # [N, C, H, W]
            video_aux_list.append(pixel_values)

        # Stack: each aux processor produces [N, C, H, W]
        # Wrap in batch dim: [1, N, C, H, W]
        visual_tensors = [v.unsqueeze(0) for v in video_aux_list]
        # video_sizes: (W, H, T)
        w, h = frames[0].size
        visual_sizes = [(w, h, len(frames))]

        return visual_tensors, visual_sizes

    def inference(self, frames: List[Image.Image], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        """Run inference with Cambrian-S."""
        # Image-option path (task 4.3.x): treat each option image as an extra
        # trailing frame; Cambrian's visual tower processes them like any
        # other frame and the prompt already labels them.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_frames
            )
        self._init_model()

        from cambrian.constants import IMAGE_TOKEN_INDEX
        from cambrian.conversation import conv_templates
        from cambrian.mm_utils import tokenizer_image_token

        # Process frames through vision pipeline
        visual_tensors, visual_sizes = self._process_frames_as_video(frames)

        # Build conversation prompt
        question = "<image>\n" + prompt
        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)

        do_sample = self.temperature > 0

        with torch.inference_mode():
            input_ids = input_ids.cuda()
            visual_tensors = [v.half().cuda() for v in visual_tensors]

            output_ids = self._model.generate(
                inputs=input_ids,
                images=visual_tensors,
                image_sizes=visual_sizes,
                use_cache=True,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_tokens,
            )

        output = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output


def _downsample_cache_states(cache_states, downsample_ratio, visual_features):
    """Downsample KV cache states via avg pooling for compression."""
    shape = cache_states.shape
    cache_states = (
        cache_states.flatten(0, 1)
        .unflatten(1, (visual_features.size(1), visual_features.size(2) + 1))
        .permute(0, 3, 1, 2)
    )
    cache_states = torch.nn.functional.avg_pool2d(
        cache_states, kernel_size=downsample_ratio, stride=downsample_ratio
    )
    cache_states = cache_states.flatten(2, 3).unflatten(0, shape[:2]).permute(0, 1, 3, 2)
    return cache_states


def _consolidate_global_cache(global_cache, layer_idx, method, threshold, budget):
    """Evict entries from global KV cache when memory budget is exceeded."""
    gc = global_cache[layer_idx]
    if sum(gc["lengths"]) <= budget:
        return

    if method == "drop_merge":
        # First pass: merge adjacent surprising pairs
        idx = 1
        while idx < len(gc["surprising_scores"]) - 1:
            if (gc["surprising_scores"][idx] >= threshold
                    and gc["surprising_scores"][idx + 1] >= threshold):
                gc["key_states"][idx] = (gc["key_states"][idx] + gc["key_states"][idx + 1]) / 2.0
                gc["value_states"][idx] = (gc["value_states"][idx] + gc["value_states"][idx + 1]) / 2.0
                gc["surprising_scores"][idx] = (gc["surprising_scores"][idx] + gc["surprising_scores"][idx + 1]) / 2.0
                for k in ("key_states", "value_states", "modalities", "lengths", "surprising_scores"):
                    gc[k].pop(idx + 1)
                torch.cuda.empty_cache()
                idx -= 1
            idx += 1
        # Second pass: drop least surprising until under budget
        while sum(gc["lengths"]) > budget:
            idx = int(np.array(gc["surprising_scores"][1:]).argmin())
            for k in ("key_states", "value_states", "modalities", "lengths", "surprising_scores"):
                gc[k].pop(idx + 1)
            torch.cuda.empty_cache()
            # Try merge neighbours
            if (1 <= idx < len(gc["surprising_scores"]) - 1
                    and gc["surprising_scores"][idx] >= threshold
                    and gc["surprising_scores"][idx + 1] >= threshold):
                gc["key_states"][idx] = (gc["key_states"][idx] + gc["key_states"][idx + 1]) / 2.0
                gc["value_states"][idx] = (gc["value_states"][idx] + gc["value_states"][idx + 1]) / 2.0
                gc["surprising_scores"][idx] = (gc["surprising_scores"][idx] + gc["surprising_scores"][idx + 1]) / 2.0
                for k in ("key_states", "value_states", "modalities", "lengths", "surprising_scores"):
                    gc[k].pop(idx + 1)
                torch.cuda.empty_cache()

    elif method == "drop":
        while sum(gc["lengths"]) > budget:
            idx = int(np.array(gc["surprising_scores"][1:]).argmin())
            for k in ("key_states", "value_states", "modalities", "lengths", "surprising_scores"):
                gc[k].pop(idx + 1)
            torch.cuda.empty_cache()


class CambrianLFPModel(CambrianModel):
    """Cambrian-S-LFP: Latent Frame Prediction with surprisingness-driven memory.

    Extends CambrianModel with:
    - Frame-by-frame KV cache processing via nfp_head
    - Surprisingness scoring (predicted vs actual next-frame features)
    - Sensory window with compression and consolidation
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.sensory_window_size = config.get("sensory_window_size", 32)
        self.surprise_threshold = config.get("surprise_threshold", 0.0)
        self.compression_downsample_ratio = config.get("compression_downsample_ratio", 2)
        self.consolidation_method = config.get("consolidation_method", "drop")
        self.consolidation_mem_budget = config.get("consolidation_mem_budget", 8192)
        self.retrieval_topk = config.get("retrieval_topk", 1)

    def _init_model(self):
        """Lazy init + apply monkey patches for frame-level KV cache control."""
        if self._model is not None:
            return
        super()._init_model()

        # Verify nfp_head exists
        if not hasattr(self._model.model, "nfp_head"):
            print("Warning: model has no nfp_head — LFP features disabled, "
                  "falling back to standard Cambrian-S inference.")
            self._has_nfp = False
        else:
            self._has_nfp = True

        # Apply monkey patches for manual KV cache management
        from qwen2_monkey_patch import Qwen2SdpaAttention, cambrian_qwen2_forward
        from cambrian.model.language_model.cambrian_qwen2 import CambrianQwenModel

        for layer in self._model.model.layers:
            layer.self_attn.__class__ = Qwen2SdpaAttention
        CambrianQwenModel.forward = cambrian_qwen2_forward
        print("LFP monkey patches applied.")

    def _encode_visual_features(self, visual_tensors, device):
        """Encode raw pixel tensors into projected visual features.

        Returns:
            visual_features: [N, H, W, C] projected features
            vit_visual_features: [N, H, W, C] raw ViT features (for nfp comparison)
        """
        # visual_tensors is a list of [1, N, C, H, W] per aux processor
        # We only use the first aux processor's output
        pixels = visual_tensors[0].flatten(0, 1)  # [N, C, H, W]
        block_size = 128
        miv_token_len = self._model.get_model().config.miv_token_len
        miv_side_len = int(math.sqrt(miv_token_len))

        all_features = []
        all_vit_features = []

        for bid in range(math.ceil(pixels.size(0) / block_size)):
            chunk = pixels[bid * block_size:(bid + 1) * block_size].half().to(device)
            chunk_feat = self._model.encode_images([chunk])[0]
            vit_chunk = chunk_feat.clone()
            chunk_feat = self._model.get_model().mm_projector(chunk_feat)

            side_len = int(math.sqrt(chunk_feat.size(1)))
            chunk_feat = chunk_feat.unflatten(1, (side_len, side_len)).permute(0, 3, 1, 2)
            vit_chunk = vit_chunk.unflatten(1, (side_len, side_len)).permute(0, 3, 1, 2)

            if side_len != miv_side_len:
                chunk_feat = torch.nn.functional.interpolate(
                    chunk_feat, size=(miv_side_len, miv_side_len),
                    mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
                vit_chunk = torch.nn.functional.interpolate(
                    vit_chunk, size=(miv_side_len, miv_side_len),
                    mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
            else:
                chunk_feat = chunk_feat.permute(0, 2, 3, 1)
                vit_chunk = vit_chunk.permute(0, 2, 3, 1)

            all_features.append(chunk_feat)
            all_vit_features.append(vit_chunk)

        return torch.cat(all_features, dim=0), torch.cat(all_vit_features, dim=0)

    def _add_newline_tokens(self, frame_feature):
        """Append image_newline token to each row of a frame feature.

        Input: [1, H, W, C]  ->  Output: [1, H*(W+1), C]
        """
        newline = self._model.model.image_newline[None, None, None, :]
        newline = newline.expand(*frame_feature.size()[:2], 1, -1)
        feat = torch.cat([frame_feature, newline], dim=2)  # [1, H, W+1, C]
        feat = feat.flatten(1, 2).flatten(0, 1)  # [(H*(W+1)), C]
        return feat

    def inference(self, frames: List[Image.Image], prompt: str,
                  option_images: Optional[List[Any]] = None) -> str:
        """LFP inference with frame-level surprisingness-driven memory."""
        # Image-option path (task 4.3.x): append option PIL images as trailing
        # frames before running the LFP pipeline. Fallback path forwards
        # option_images down to the base class for consistent handling.
        if option_images:
            from option_utils import append_option_images_to_frames
            frames = append_option_images_to_frames(
                frames, option_images, max_n=self.max_frames
            )
        self._init_model()

        # Fall back to standard inference if no nfp_head
        if not self._has_nfp:
            return super().inference(frames, prompt)

        from cambrian.constants import IMAGE_TOKEN_INDEX
        from cambrian.conversation import conv_templates
        from cambrian.mm_utils import tokenizer_image_token
        from cambrian.model.cambrian_arch import unpad_image

        device = next(self._model.parameters()).device
        num_layers = self._model.config.num_hidden_layers

        # Process frames through vision pipeline
        visual_tensors, visual_sizes = self._process_frames_as_video(frames)

        # Build conversation prompt
        question = "<image>\n" + prompt
        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            visual_tensors = [v.half().to(device) for v in visual_tensors]

            # Encode visual features
            visual_features, vit_visual_features = self._encode_visual_features(
                visual_tensors, device
            )

            # Unpad to original aspect ratio
            visual_features = unpad_image(visual_features, visual_sizes[0][:2])
            vit_visual_features = unpad_image(vit_visual_features, visual_sizes[0][:2])

            # Split input_ids at IMAGE_TOKEN position
            img_pos = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0][0]
            pre_img_tokens = input_ids[:, :img_pos]
            post_img_tokens = input_ids[:, img_pos + 1:]

            # --- Phase 1: Process pre-image text tokens ---
            pre_img_embeds = self._model.get_input_embeddings()(pre_img_tokens)

            def _new_layer_cache():
                return {"key_states": [], "value_states": [], "modalities": [],
                        "lengths": [], "surprising_scores": []}

            global_kv_cache = [_new_layer_cache() for _ in range(num_layers)]
            runtime_kv_cache = [_new_layer_cache() for _ in range(num_layers)]

            out = self._model(
                input_ids=None, inputs_embeds=pre_img_embeds,
                attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=True,
                output_attentions=False, output_hidden_states=True,
                return_dict=True,
            )

            for li, (ks, vs) in enumerate(out.past_key_values):
                for cache in (global_kv_cache, runtime_kv_cache):
                    cache[li]["key_states"].append(ks)
                    cache[li]["value_states"].append(vs)
                    cache[li]["modalities"].append("T")
                    cache[li]["lengths"].append(ks.size(2))
                    cache[li]["surprising_scores"].append(1.0)

            # --- Phase 2: Process frames one by one ---
            frame_feature_prediction = None

            for frame_idx in range(visual_features.size(0)):
                # Build past_key_values from runtime cache
                past_kv = []
                for li in range(num_layers):
                    past_kv.append((
                        torch.cat(runtime_kv_cache[li]["key_states"], dim=2),
                        torch.cat(runtime_kv_cache[li]["value_states"], dim=2),
                    ))

                frame_feat = visual_features[frame_idx:frame_idx + 1]

                # Compute surprisingness
                if frame_idx == 0 or frame_feature_prediction is None:
                    surprisingness = 1.0
                else:
                    pred = frame_feature_prediction.unflatten(
                        1, (vit_visual_features.size(1), vit_visual_features.size(2) + 1)
                    )[:, :, :-1]
                    target = vit_visual_features[frame_idx:frame_idx + 1].to(pred.device)
                    surprisingness = (
                        1 - torch.cosine_similarity(
                            pred.flatten(1, 2), target.flatten(1, 2), dim=-1
                        ).mean(1).item()
                    )

                # Add newline tokens and forward
                input_embeds = self._add_newline_tokens(frame_feat).unsqueeze(0)

                out = self._model(
                    input_ids=None, inputs_embeds=input_embeds,
                    attention_mask=None, position_ids=None,
                    past_key_values=past_kv, use_cache=True,
                    output_attentions=False, output_hidden_states=True,
                    return_dict=True,
                )

                # NFP prediction for next frame
                frame_feature_prediction = self._model.model.nfp_head(out.hidden_states)

                # Update runtime KV cache
                for li in range(num_layers):
                    input_len = past_kv[li][0].size(2)
                    new_ks = out.past_key_values[li][0][..., input_len:, :].clone()
                    new_vs = out.past_key_values[li][1][..., input_len:, :].clone()

                    runtime_kv_cache[li]["key_states"].append(new_ks)
                    runtime_kv_cache[li]["value_states"].append(new_vs)
                    runtime_kv_cache[li]["modalities"].append("I")
                    runtime_kv_cache[li]["lengths"].append(new_ks.size(2))
                    runtime_kv_cache[li]["surprising_scores"].append(surprisingness)

                    # Sensory window eviction
                    sw = self.sensory_window_size
                    if sw > 0 and len(runtime_kv_cache[li]["key_states"]) > sw + 1:
                        evicted_ks = runtime_kv_cache[li]["key_states"].pop(1)
                        evicted_vs = runtime_kv_cache[li]["value_states"].pop(1)
                        evicted_score = runtime_kv_cache[li]["surprising_scores"].pop(1)
                        evicted_mod = runtime_kv_cache[li]["modalities"].pop(1)
                        runtime_kv_cache[li]["lengths"].pop(1)

                        # Compress low-surprise frames
                        if (self.compression_downsample_ratio > 1
                                and evicted_score < self.surprise_threshold):
                            evicted_ks = _downsample_cache_states(
                                evicted_ks, self.compression_downsample_ratio, visual_features
                            )
                            evicted_vs = _downsample_cache_states(
                                evicted_vs, self.compression_downsample_ratio, visual_features
                            )

                        global_kv_cache[li]["key_states"].append(evicted_ks)
                        global_kv_cache[li]["value_states"].append(evicted_vs)
                        global_kv_cache[li]["modalities"].append(evicted_mod)
                        global_kv_cache[li]["lengths"].append(evicted_ks.size(2))
                        global_kv_cache[li]["surprising_scores"].append(evicted_score)

                        # Consolidate if over budget
                        if self.consolidation_method:
                            _consolidate_global_cache(
                                global_kv_cache, li, self.consolidation_method,
                                self.surprise_threshold, self.consolidation_mem_budget,
                            )

            # --- Phase 2.5: Flush remaining runtime cache into global ---
            for li in range(num_layers):
                for ci in range(1, len(runtime_kv_cache[li]["key_states"])):
                    ks = runtime_kv_cache[li]["key_states"][ci]
                    vs = runtime_kv_cache[li]["value_states"][ci]
                    score = runtime_kv_cache[li]["surprising_scores"][ci]

                    if (self.compression_downsample_ratio > 1
                            and score < self.surprise_threshold):
                        ks = _downsample_cache_states(
                            ks, self.compression_downsample_ratio, visual_features
                        )
                        vs = _downsample_cache_states(
                            vs, self.compression_downsample_ratio, visual_features
                        )

                    global_kv_cache[li]["key_states"].append(ks)
                    global_kv_cache[li]["value_states"].append(vs)
                    global_kv_cache[li]["modalities"].append(
                        runtime_kv_cache[li]["modalities"][ci]
                    )
                    global_kv_cache[li]["lengths"].append(ks.size(2))
                    global_kv_cache[li]["surprising_scores"].append(score)

                    if self.consolidation_method:
                        _consolidate_global_cache(
                            global_kv_cache, li, self.consolidation_method,
                            self.surprise_threshold, self.consolidation_mem_budget,
                        )

            # Build final past_key_values
            past_key_values = []
            for li in range(num_layers):
                past_key_values.append((
                    torch.cat(global_kv_cache[li]["key_states"], dim=2),
                    torch.cat(global_kv_cache[li]["value_states"], dim=2),
                ))

            # Set up retrieval if configured
            if self.retrieval_topk > 1:
                for li, layer in enumerate(self._model.model.layers):
                    layer.self_attn.use_retrieval = True
                    layer.self_attn.retrieval_topk = self.retrieval_topk
                    layer.self_attn.cache_modalities = global_kv_cache[li]["modalities"]
                    layer.self_attn.cache_lengths = global_kv_cache[li]["lengths"]

            # --- Phase 3: Process post-image text tokens ---
            post_img_embeds = self._model.get_input_embeddings()(post_img_tokens)

            out = self._model(
                input_ids=None, inputs_embeds=post_img_embeds,
                attention_mask=None, position_ids=None,
                past_key_values=past_key_values, use_cache=True,
                output_attentions=False, output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values

            # --- Phase 4: Token-by-token generation with repetition penalty ---
            logits = out.logits[:, -1, :]
            pred = logits.argmax(dim=-1)
            output_ids = torch.cat([
                torch.zeros_like(pred)[:, None].long().fill_(self._tokenizer.pad_token_id),
                pred[:, None],
            ], dim=1)

            for _ in range(self.max_tokens - 1):
                if pred.item() == self._tokenizer.eos_token_id:
                    break

                out = self._model(
                    input_ids=output_ids[:, -1:],
                    inputs_embeds=None,
                    attention_mask=None, position_ids=None,
                    past_key_values=past_key_values, use_cache=True,
                    output_attentions=False, output_hidden_states=True,
                    return_dict=True,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]

                # Repetition penalty (1.1x)
                score = torch.gather(logits, 1, output_ids)
                score = torch.where(score < 0, score * 1.1, score / 1.1)
                logits.scatter_(1, output_ids, score)

                pred = logits.argmax(dim=-1)
                output_ids = torch.cat([output_ids, pred[:, None]], dim=-1)

        output = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output
