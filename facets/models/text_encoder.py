"""
Text encoder for FACETS

"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Checkpoint inspection
# ===========================================================================

def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt: sd = ckpt['state_dict']
        elif 'model' in ckpt: sd = ckpt['model']
        else: sd = ckpt
    else:
        sd = ckpt
    return {k.replace('module.', ''): v for k, v in sd.items()}


_TEXT_SIGNATURE_KEYS = (
    'token_embedding.weight',
    'positional_embedding',
    'ln_final.weight',
    'text_projection',
)

_PREFIXES_TO_TRY = (
    '',
    'text_encoder.',
    'clip.',
    'clip_model.',
    'text_model.',
    'text.',
    'text_encoder.text.',
    'clip.text.',
)


def _find_text_prefix(sd: Dict[str, torch.Tensor]
                      ) -> Optional[str]:
    """
    Identify the prefix under which text-encoder weights sit.

    """
    for prefix in _PREFIXES_TO_TRY:
        found = sum((prefix + k) in sd for k in _TEXT_SIGNATURE_KEYS)
        if found >= 3:
            return prefix
    return None


def _detect_clip_variant(sd: Dict[str, torch.Tensor], prefix: str
                         ) -> Optional[Tuple[str, Dict[str, int]]]:
    """
    Detect OpenCLIP variant name from the text-encoder weight shapes.

    """
    try:
        tok = sd[prefix + 'token_embedding.weight']                # (V, D)
        pos = sd[prefix + 'positional_embedding']                  # (L, D)
        tp  = sd[prefix + 'text_projection']                       # (D, E)
    except KeyError:
        return None

    vocab_size = tok.shape[0]
    transformer_width = tok.shape[1]
    context_length = pos.shape[0]
    embed_dim = tp.shape[1]

    # Count transformer blocks: look for the highest block index.
    max_blk = -1
    for key in sd.keys():
        if key.startswith(prefix + 'transformer.resblocks.'):
            try:
                blk = int(key.split(prefix + 'transformer.resblocks.', 1)[1]
                          .split('.')[0])
                max_blk = max(max_blk, blk)
            except (ValueError, IndexError):
                pass
    if max_blk < 0:
        return None
    layers = max_blk + 1

    # Standard CLIP / OpenCLIP head counts.
    # (transformer_width -> transformer_heads)
    #   512 -> 8, 768 -> 12, 1024 -> 16, 1280 -> 20
    head_map = {512: 8, 768: 12, 1024: 16, 1280: 20}
    heads = head_map.get(transformer_width, max(1, transformer_width // 64))

    arch = {
        'embed_dim': embed_dim,
        'context_length': context_length,
        'vocab_size': vocab_size,
        'transformer_width': transformer_width,
        'transformer_heads': heads,
        'transformer_layers': layers,
    }

    if embed_dim == 512 and transformer_width == 512 and layers == 12:
        variant = 'ViT-B-32'
    elif embed_dim == 768 and transformer_width == 768 and layers == 12:
        variant = 'ViT-L-14'
    elif embed_dim == 1024 and transformer_width == 1024 and layers == 24:
        variant = 'ViT-H-14'
    elif embed_dim == 1280 and transformer_width == 1280 and layers == 32:
        variant = 'ViT-bigG-14'
    else:
        variant = None       # unknown; caller will handle

    return variant, arch


# ===========================================================================
# OpenCLIP builder
# ===========================================================================

def _try_open_clip(model_name: str, pretrained: Optional[str] = None):
    """
    Return (model, tokenizer) or (None, None) if open_clip unavailable.

    """
    try:
        import open_clip
    except ImportError:
        return None, None
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        tok = open_clip.get_tokenizer(model_name)
        return model, tok
    except Exception as e:
        print(f"[text_encoder] open_clip.create_model_and_transforms failed: {e}")
        return None, None


def _load_ulip_text_weights_into_model(model: nn.Module,
                                       sd: Dict[str, torch.Tensor],
                                       prefix: str) -> int:
    """
    Copy text-encoder weights from ULIP-2 state-dict into ``model``.

    """
    model_sd = model.state_dict()
    n_loaded = 0
    for k in model_sd:
        if k.startswith('visual.'):            # don't try to overwrite image side
            continue
        src_key = prefix + k
        if src_key in sd and model_sd[k].shape == sd[src_key].shape:
            model_sd[k] = sd[src_key]
            n_loaded += 1
    if n_loaded > 0:
        model.load_state_dict(model_sd, strict=False)
    return n_loaded


# ===========================================================================
# Wrapper
# ===========================================================================

class CLIPTextWrapper(nn.Module):
    """
    Wraps a text encoder for L2-normalized encoding of captions.

    """

    def __init__(self, clip_model, tokenizer, context_length: int = 77,
                 freeze: bool = True):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.context_length = context_length
        with torch.no_grad():
            test = self._tokenize(['test'])
            emb = self._encode(test)
            self.embed_dim = emb.shape[-1]
        if freeze:
            for p in self.clip_model.parameters():
                p.requires_grad_(False)
            self.clip_model.eval()

    def _tokenize(self, captions: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(captions)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(next(self.clip_model.parameters()).device)
        if hasattr(self.clip_model, 'encode_text'):
            feat = self.clip_model.encode_text(tokens)
        else:
            feat = self.clip_model(tokens)
        return feat

    @torch.no_grad()
    def encode(self, captions: List[str], normalize: bool = True
               ) -> torch.Tensor:
        tokens = self._tokenize(captions)
        feats = self._encode(tokens).float()
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats


# ===========================================================================
# Public factory
# ===========================================================================

def build_text_encoder_from_checkpoint(
        ulip_ckpt_path: Optional[str] = None,
        open_clip_variant: str = 'ViT-bigG-14',
        open_clip_pretrained: Optional[str] = 'laion2b_s39b_b160k',
        device: str = 'cuda',
        verbose: bool = True,
) -> CLIPTextWrapper:
    """
    Build the text encoder.

    """
    # ------------------ Try ULIP-2 checkpoint first ------------------
    ulip_model, ulip_tok = None, None
    used_ulip_weights = False
    if ulip_ckpt_path:
        try:
            sd = _load_state_dict(ulip_ckpt_path)
            prefix = _find_text_prefix(sd)
            if prefix is not None:
                det = _detect_clip_variant(sd, prefix)
                if det is not None:
                    variant, arch = det
                    if variant is not None:
                        if verbose:
                            print(f"[text_encoder] found CLIP text encoder in "
                                  f"ULIP checkpoint under prefix '{prefix}' "
                                  f"({variant}, embed_dim={arch['embed_dim']})")

                        # Instantiate the architecture (no pretrained download).
                        m, tok = _try_open_clip(variant, pretrained=None)
                        if m is not None:
                            n = _load_ulip_text_weights_into_model(m, sd, prefix)
                            if verbose:
                                print(f"[text_encoder] copied {n} tensors from "
                                      f"ULIP-2 checkpoint into the text encoder")
                            if n > 10:
                                ulip_model, ulip_tok = m, tok
                                used_ulip_weights = True
                        else:
                            if verbose:
                                print("[text_encoder] open_clip not available; "
                                      "cannot instantiate "
                                      f"{variant}")
                    else:
                        if verbose:
                            print(f"[text_encoder] found text-encoder-shaped "
                                  f"weights under '{prefix}' but could not "
                                  f"identify a standard CLIP variant "
                                  f"(arch = {arch}). Falling back to OpenCLIP.")
                else:
                    if verbose:
                        print("[text_encoder] incomplete text-encoder weights "
                              "in ULIP checkpoint; falling back to OpenCLIP.")
            else:
                if verbose:
                    print("[text_encoder] no CLIP text-encoder weights detected "
                          "in the ULIP checkpoint; falling back to OpenCLIP.")
        except Exception as e:
            if verbose:
                print(f"[text_encoder] inspection of ULIP checkpoint failed "
                      f"({e}); falling back to OpenCLIP.")

    # ------------------ Fall back to OpenCLIP pretrained ------------------
    if ulip_model is None:
        m, tok = _try_open_clip(open_clip_variant, open_clip_pretrained)
        if m is None:
            # Very last resort: HF CLIP-L
            try:
                from transformers import CLIPTokenizer, CLIPTextModel

                class _HFWrap(nn.Module):
                    def __init__(self, mdl): super().__init__(); self.m = mdl
                    def encode_text(self, toks):
                        out = self.m(toks)
                        return out.pooler_output

                class _HFTok:
                    def __init__(self, t): self.t = t
                    def __call__(self, caps):
                        return self.t(caps, padding='max_length',
                                      truncation=True, max_length=77,
                                      return_tensors='pt')['input_ids']

                tok_hf = CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-large-patch14")
                mdl_hf = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-large-patch14")
                m = _HFWrap(mdl_hf)
                tok = _HFTok(tok_hf)
                if verbose:
                    print("[text_encoder] using HF CLIP-L/14 as last-resort "
                          "fallback.")
            except Exception as e:
                raise RuntimeError(
                    "Could not construct a CLIP text encoder. Install "
                    "open_clip_torch or transformers. Error: " + str(e))
        else:
            if verbose:
                print(f"[text_encoder] using OpenCLIP "
                      f"{open_clip_variant}/{open_clip_pretrained}")
        ulip_model, ulip_tok = m, tok

    ulip_model = ulip_model.to(device)
    wrapper = CLIPTextWrapper(ulip_model, ulip_tok).to(device)

    # Annotate the wrapper so the caller can log provenance.
    wrapper.loaded_from_ulip_checkpoint = used_ulip_weights
    return wrapper
