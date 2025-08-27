from __future__ import annotations

from typing import Optional, Tuple, Any

from jax import Array
from flax import nnx
from MaxText.layers.models import Transformer




class TunixMaxTextAdaptor(nnx.Module):
  """Adapter exposing Tunix Trainer call signature over a Transformer model."""

  def __init__(
      self,
      base_model: Transformer,
  ):
    super().__init__()
    self.base = base_model

  # ------------------------------------------------------------------ #
  # Tunix call signature
  # ------------------------------------------------------------------ #
  def __call__(
      self,
      input_tokens: Array,  # [B, L]
      positions: Array,  # [B, L]
      cache: Optional[Any],  # Tunix currently passes None from Trainers
      attention_mask: Optional[Array],  # [B, L, L] or None
      output_hidden_states: bool = False,  # ignored
  ) -> Tuple[Array, None]:
    """Forward compatible with Tunix Trainers default loss.
    Returns logits, None.
    """
    logits = self.base(
        decoder_input_tokens=input_tokens,
        decoder_positions=positions,
        # TODO: @mazumdera - add support for packing
        decoder_segment_ids=None,
    )
    return logits, None


def create_maxtext_to_vllm_mappings():
  """Create mappings for transferring MaxText scanned state to vLLM unscanned state."""
  return {
      # Token embeddings - shard vocab dimension for TP
      "base.token_embedder.embedding": (
          "model.embed.embedding",
          ("model", None),
      ),
      # Final layer norm - no sharding needed
      "base.decoder.decoder_norm.scale": (
          "model.norm.scale",
          (None,),
      ),
      # LM head (logits projection) - shard vocab dimension for TP
      "base.decoder.logits_dense.kernel": (
          "model.lm_head",
          (None, "model"),
      ),
      # Layer-specific mappings (scanned -> unscanned)
      # MLP components - shard hidden dimensions for TP
      "base.decoder.layers.mlp.wi_0.kernel": (
          "model.layers.*.mlp.gate_proj.kernel",
          (None, "layer", "model"),
      ),  # gate_proj: (4096, 14336) - shard output
      "base.decoder.layers.mlp.wi_1.kernel": (
          "model.layers.*.mlp.up_proj.kernel",
          (None, "layer", "model"),
      ),  # up_proj: (4096, 14336) - shard output
      "base.decoder.layers.mlp.wo.kernel": (
          "model.layers.*.mlp.down_proj.kernel",
          ("model", "layer", None),
      ),  # down_proj: (14336, 4096) - shard input
      # Layer norms - no sharding needed
      "base.decoder.layers.pre_self_attention_layer_norm.scale": (
          "model.layers.*.input_layernorm.scale",
          (None, "layer"),
      ),
      "base.decoder.layers.post_self_attention_layer_norm.scale": (
          "model.layers.*.post_attention_layernorm.scale",
          (None, "layer"),
      ),
      # Attention components - shard head dimensions for TP
      "base.decoder.layers.self_attention.query.kernel": (
          "model.layers.*.self_attn.q_proj.kernel",
          (None, "layer", "model", None),
      ),  # q_proj: shard num_heads # NOT MATCH
      "base.decoder.layers.self_attention.key.kernel": (
          "model.layers.*.self_attn.k_proj.kernel",
          (None, "layer", "model", None),
      ),  # k_proj: shard num_kv_heads
      "base.decoder.layers.self_attention.value.kernel": (
          "model.layers.*.self_attn.v_proj.kernel",
          (None, "layer", "model", None),  # match
      ),  # v_proj: shard num_kv_heads
      "base.decoder.layers.self_attention.out.kernel": (
          "model.layers.*.self_attn.o_proj.kernel",
          ("model", "layer", None, None),
      ),  # o_proj: shard input heads #match
  }
