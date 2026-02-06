from testing import assert_true
from math import cos, sin
from random import random_float64
from complex import ComplexSIMD

from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from nn.fused_qk_rope_vision import fused_qk_rope_vision
from utils import IndexList
from memory import memcpy

# Hardcode dtype to bfloat16 as the kernel is specialized for it
comptime dtype = DType.bfloat16

def run_test[freq_dtype: DType](ctx: DeviceContext):
    # Dimensions
    comptime batch = 1
    comptime seq_len = 4
    comptime num_q_heads = 2
    comptime num_k_heads = 2
    comptime head_dim = 16 

    # Layouts
    comptime input_layout = Layout.row_major(batch, seq_len, num_q_heads, head_dim)
    comptime k_input_layout = Layout.row_major(batch, seq_len, num_k_heads, head_dim)
    comptime freq_layout = Layout.row_major(seq_len, head_dim)

    # Shapes
    var input_shape = IndexList[4](batch, seq_len, num_q_heads, head_dim)
    var k_input_shape = IndexList[4](batch, seq_len, num_k_heads, head_dim)
    var freq_shape = IndexList[2](seq_len, head_dim)

    # Runtime Layouts
    var input_rl = RuntimeLayout[input_layout].row_major(input_shape)
    var k_input_rl = RuntimeLayout[k_input_layout].row_major(k_input_shape)
    var freq_rl = RuntimeLayout[freq_layout].row_major(freq_shape)

    # Device Allocations
    var q_dev = ctx.enqueue_create_buffer[dtype](input_shape.flattened_length())
    var k_dev = ctx.enqueue_create_buffer[dtype](k_input_shape.flattened_length())
    var cos_dev = ctx.enqueue_create_buffer[freq_dtype](freq_shape.flattened_length())
    var sin_dev = ctx.enqueue_create_buffer[freq_dtype](freq_shape.flattened_length())
    var q_out_dev = ctx.enqueue_create_buffer[dtype](input_shape.flattened_length())
    var k_out_dev = ctx.enqueue_create_buffer[dtype](k_input_shape.flattened_length())

    # Initialize Data
    with q_dev.map_to_host() as ptr:
        var t = LayoutTensor[dtype, input_layout](ptr, input_rl)
        for b in range(batch):
            for s in range(seq_len):
                for h in range(num_q_heads):
                    for d in range(head_dim):
                        var val = Float32(s * 100 + h * 10 + d)
                        var idx = IndexList[4](b, s, h, d)
                        t.store(idx, val.cast[dtype]())

    with k_dev.map_to_host() as ptr:
        var t = LayoutTensor[dtype, k_input_layout](ptr, k_input_rl)
        for b in range(batch):
            for s in range(seq_len):
                for h in range(num_k_heads):
                    for d in range(head_dim):
                        var val = Float32(s * 100 + h * 10 + d) + 0.5
                        var idx = IndexList[4](b, s, h, d)
                        t.store(idx, val.cast[dtype]())

    with cos_dev.map_to_host() as c_ptr, sin_dev.map_to_host() as s_ptr:
        var ct = LayoutTensor[freq_dtype, freq_layout](c_ptr, freq_rl)
        var st = LayoutTensor[freq_dtype, freq_layout](s_ptr, freq_rl)
        for s in range(seq_len):
            for d in range(head_dim // 2):
                var angle = Float32(s) * 0.1 + Float32(d) * 0.5
                var c = cos(angle)
                var s_val = sin(angle)
                
                var idx0 = IndexList[2](s, d*2)
                var idx1 = IndexList[2](s, d*2+1)
                
                ct.store(idx0, c.cast[freq_dtype]())
                ct.store(idx1, c.cast[freq_dtype]())
                
                st.store(idx0, s_val.cast[freq_dtype]())
                st.store(idx1, s_val.cast[freq_dtype]())

    # Create Tensors for Kernel
    var q = LayoutTensor[dtype, input_layout](q_dev, input_rl)
    var k = LayoutTensor[dtype, k_input_layout](k_dev, k_input_rl)
    var cos_t = LayoutTensor[freq_dtype, freq_layout](cos_dev, freq_rl)
    var sin_t = LayoutTensor[freq_dtype, freq_layout](sin_dev, freq_rl)
    var q_out = LayoutTensor[dtype, input_layout](q_out_dev, input_rl)
    var k_out = LayoutTensor[dtype, k_input_layout](k_out_dev, k_input_rl)

    # Run Kernel
    fused_qk_rope_vision[freq_dtype, repeat_interleave=True, target="gpu"](
        q, k, cos_t, sin_t, q_out, k_out, ctx
    )
    ctx.synchronize()

    # Verify Output
    with q_out_dev.map_to_host() as ptr:
        var t = LayoutTensor[dtype, input_layout](ptr, input_rl)
        # Check position s=1, h=0, d=0,1 (angle 0.1)
        # q[0]=100, q[1]=101
        # cos=cos(0.1)=0.995, sin=sin(0.1)=0.0998
        # Rotated 0: 100*c - 101*s = 99.5 - 10.08 = 89.42
        # Rotated 1: 100*s + 101*c = 9.98 + 100.495 = 110.475
        
        var idx0 = IndexList[4](0, 1, 0, 0)
        var idx1 = IndexList[4](0, 1, 0, 1)
        
        # Explicit width=1 to avoid inference error
        var q0 = t.load[width=1](idx0).cast[DType.float32]()
        var q1 = t.load[width=1](idx1).cast[DType.float32]()
        
        # Loose checks for BF16 precision
        assert_true(q0 > 89.0 and q0 < 90.0)
        assert_true(q1 > 110.0 and q1 < 111.0)
        
    print("Test finished successfully")

def main():
    with DeviceContext() as ctx:
        run_test[DType.float32](ctx)
