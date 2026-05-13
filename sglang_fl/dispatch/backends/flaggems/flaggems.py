# FlagGems backend class.

from __future__ import annotations

from .. import Backend


class FlagGemsBackend(Backend):
    """FlagGems (FlagOS) default backend — Triton-based implementations."""

    _available = None

    @property
    def name(self) -> str:
        return "flaggems"

    def is_available(self) -> bool:
        if FlagGemsBackend._available is None:
            try:
                import flag_gems  # noqa: F401

                FlagGemsBackend._available = True
            except ImportError:
                FlagGemsBackend._available = False
        return FlagGemsBackend._available

    def silu_and_mul(self, obj, x):
        from .impl.activation import silu_and_mul_flaggems

        return silu_and_mul_flaggems(obj, x)

    def rms_norm(self, obj, x, residual=None):
        from .impl.normalization import rms_norm_flaggems

        return rms_norm_flaggems(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query,
        key,
        cos,
        sin,
        position_ids,
        rotary_interleaved=False,
        inplace=True,
    ):
        from .impl.rotary import rotary_embedding_flaggems

        return rotary_embedding_flaggems(
            obj, query, key, cos, sin, position_ids, rotary_interleaved, inplace
        )

    def topk(
        self,
        obj,
        hidden_states,
        router_logits,
        *,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
    ):
        from .impl.topk import topk_flaggems

        return topk_flaggems(
            obj,
            hidden_states,
            router_logits,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    def fused_moe(self, obj, layer, dispatch_output):
        from .impl.fused_moe import fused_moe_flaggems

        return fused_moe_flaggems(obj, layer, dispatch_output)

    def chunk_gated_delta_rule(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state=None,
        initial_state_indices=None,
        cu_seqlens=None,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    ):
        from .impl.fla import chunk_gated_delta_rule_flaggems

        return chunk_gated_delta_rule_flaggems(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            initial_state_indices,
            cu_seqlens,
            head_first,
            use_qk_l2norm_in_kernel,
        )

    def fused_recurrent_gated_delta_rule(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
    ):
        from .impl.fla import fused_recurrent_gated_delta_rule_flaggems

        return fused_recurrent_gated_delta_rule_flaggems(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            ssm_state_indices,
            num_accepted_tokens,
            use_qk_l2norm_in_kernel,
        )

    def fused_recurrent_gated_delta_rule_packed_decode(
        self,
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel=False,
    ):
        from .impl.fla import fused_recurrent_gated_delta_rule_packed_decode_flaggems

        return fused_recurrent_gated_delta_rule_packed_decode_flaggems(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            scale,
            initial_state,
            out,
            ssm_state_indices,
            use_qk_l2norm_in_kernel,
        )
