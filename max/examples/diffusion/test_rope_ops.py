
from max import functional as F
from max.dtype import DType
from max.tensor import Tensor
import numpy as np

def test_complex_mul():
    # Create input x: [1, 4] -> [1, 2, 2] interleaved
    # 1+2i, 3+4i
    x_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    x = Tensor(x_np)
    
    # Create freqs: cis = cos + i*sin
    # rot = 90 deg -> cos=0, sin=1 -> 0 + 1i
    cis_np = np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float32).reshape(1, 2, 2)
    cis = Tensor(cis_np)
    
    # Expected out: x * i
    # (1+2i)i = -2+i -> -2, 1
    # (3+4i)i = -4+3i -> -4, 3
    # Result: [-2, 1, -4, 3]
    
    print("Testing F.as_interleaved_complex...")
    try:
        x_c = F.as_interleaved_complex(x)
        print(f"x_c shape: {x_c.shape}, dtype: {x_c.dtype}")
    except Exception as e:
        print(f"as_interleaved_complex failed: {e}")
        return

    print("Testing F.complex_mul with complex + complex...")
    try:
        # We need cis as complex too
        # cis is [1, 2, 2]. We need to interpret last dim as complex?
        # F.as_interleaved_complex expects last dim to be flattened [..., D]?
        # Or does it handle [..., 2]?
        # Let's try converting cis to complex
        cis_flat = F.reshape(cis, [1, 4])
        cis_c = F.as_interleaved_complex(cis_flat)
        
        res = F.complex_mul(x_c, cis_c)
        print(f"res shape: {res.shape}, dtype: {res.dtype}")
        
        # Convert back to real?
        # How to view complex as float? F.view? F.cast?
        # F.as_real_interleaved? (guessing name)
        # Or just cast?
    except Exception as e:
        print(f"complex_mul failed: {e}")

    print("Testing F.complex_mul with complex + float[..., 2]...")
    try:
        res2 = F.complex_mul(x_c, cis)
        print(f"res2 shape: {res2.shape}, dtype: {res2.dtype}")
    except Exception as e:
        print(f"complex_mul with float[...,2] failed: {e}")

if __name__ == "__main__":
    test_complex_mul()
