# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Fused Q/K RoPE kernel for vision models (no KV cache).

This kernel applies Rotary Position Embedding (RoPE) to both query and key
tensors in a single fused operation, optimized for BF16 on GPU.
"""

from complex import ComplexSIMD
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from algorithm import parallelize
from sys.info import simd_width_of
from register import register_internal

from utils.index import IndexList


@always_inline
fn _rope[
    dtype: DType,
    freq_dtype: DType,
    width: Int,
](val: SIMD[dtype, width], cos: SIMD[freq_dtype, width], sin: SIMD[freq_dtype, width]) -> SIMD[dtype, width]:
    """Apply RoPE rotation using complex multiplication.
    
    val is complex (interleaved real/imag), cos/sin are the rotation coefficients.
    Complex multiplication for rotation: 
        out_re = x_re * cos - x_im * sin
        out_im = x_re * sin + x_im * cos
    """
    var x_complex = val.cast[freq_dtype]().deinterleave()
    var x_re = x_complex[0]
    var x_im = x_complex[1]
    
    # cos/sin are repeated [c0, c0, c1, c1], so deinterleaving gives [c0, c1] in both parts
    # We need the values corresponding to x_re (even indices) and x_im (odd indices)
    # Since they are repeated, the even and odd parts are identical.
    var cos_parts = cos.deinterleave()
    var sin_parts = sin.deinterleave()
    
    var cos_half = cos_parts[0]
    var sin_half = sin_parts[0] 
    
    # Apply rotation
    var out_re = x_re * cos_half - x_im * sin_half
    var out_im = x_re * sin_half + x_im * cos_half
    
    return rebind[SIMD[dtype, width]](out_re.interleave(out_im).cast[dtype]())


@register_internal("mo.fused_qk_rope_vision")
fn fused_qk_rope_vision[
    target: StaticString,
](
    q_out: OutputTensor,
    k_out: OutputTensor,
    query: InputTensor[dtype=q_out.dtype, rank=q_out.rank],
    key: InputTensor[dtype=k_out.dtype, rank=k_out.rank],
    freqs_cos: InputTensor,
    freqs_sin: InputTensor,
    ctx: DeviceContextPtr,
) raises:
    """Fused Q/K RoPE for vision models.
    
    Applies RoPE to query and key tensors simultaneously.
    Input shapes:
        - query: [B, S, num_heads, head_dim]
        - key: [B, S, num_heads, head_dim]
        - freqs_cos: [S, head_dim]  (cos values, interleaved pairs)
        - freqs_sin: [S, head_dim]  (sin values, interleaved pairs)
    Output shapes:
        - q_out: same as query
        - k_out: same as key
    """

    # Extract dimensions
    var batch_size = query.dim_size(0)
    var seq_len = query.dim_size(1)
    var num_q_heads = query.dim_size(2)
    var head_dim = query.dim_size(3)
    var num_k_heads = key.dim_size(2)
    
    comptime simd_width = simd_width_of[DType.bfloat16]()

    # Verify vector length divides head_dim
    # constrained[head_dim % simd_width == 0, "head_dim must be multiple of simd_width"]()
    debug_assert(head_dim % simd_width == 0, "head_dim must be multiple of simd_width")

    # Define the worker function for foreach
    # We parallelize over Batch (b) and Sequence (s) dimensions
    # Inner loops over heads (h) and vector dim (v) will be sequential or unrolled
    
    @parameter
    fn _worker(idx: Int):
        var b = idx // seq_len
        var s = idx % seq_len
        
        for v in range(0, head_dim, simd_width):
            # Load freq (shared across heads and batch)
            var cos_val = freqs_cos.load[width=simd_width](IndexList[2](s, v))
            var sin_val = freqs_sin.load[width=simd_width](IndexList[2](s, v))
            
            var cos_bf16 = cos_val.cast[DType.bfloat16]()
            var sin_bf16 = sin_val.cast[DType.bfloat16]()
            
            # Process Query Heads
            for h in range(num_q_heads):
                var val = query.load[width=simd_width](IndexList[4](b, s, h, v))
                var res = _rope(val, cos_bf16, sin_bf16)
                q_out.store[width=simd_width](IndexList[4](b, s, h, v), res)
                
            # Process Key Heads
            for h in range(num_k_heads):
                var val = key.load[width=simd_width](IndexList[4](b, s, h, v))
                var res = _rope(val, cos_bf16, sin_bf16)
                k_out.store[width=simd_width](IndexList[4](b, s, h, v), res)
                
    # Launch parallel execution
    # Total items = B * S
    parallelize[_worker](batch_size * seq_len)
