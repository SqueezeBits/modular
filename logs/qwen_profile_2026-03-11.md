# Qwen E2E Profile Log

Date: 2026-03-11 UTC

## Command

```bash
env MODULAR_ENABLE_PROFILING=detailed nsys profile -o /root/modular/qwen_full_e2e_profile_detailed --force-overwrite true --sample=none --trace=cuda,nvtx,osrt bash -lc './bazelw run //max/examples/diffusion:simple_offline_generation -- --model Qwen/Qwen-Image-2512 --prompt "A young woman with long brown curly pigtails, pastel striped shirt, standing outdoors, photorealistic" --negative-prompt "low quality" --width 768 --height 1024 --num-inference-steps 30 --true-cfg-scale 4.0 --seed 0 --output woman_e2e.png && ./bazelw run //max/examples/diffusion:simple_offline_generation -- --model Qwen/Qwen-Image-2512 --prompt "A man wearing a plain white t-shirt and jeans, standing outdoors, photorealistic" --negative-prompt "low quality" --width 768 --height 1024 --num-inference-steps 30 --true-cfg-scale 4.0 --seed 1 --output man_e2e.png && ./bazelw run //max/examples/diffusion:simple_offline_generation -- --model Qwen/Qwen-Image-Edit-2511 --prompt "Place these two people side by side in a natural outdoor portrait photo." --negative-prompt "duplicate person, extra limbs, distorted body" --negative-prompt " " --input-image woman_e2e.png --input-image man_e2e.png --width 1536 --height 1024 --num-inference-steps 40 --guidance-scale 1.0 --true-cfg-scale 4.0 --seed 0 --output output_combined_e2e.png'
```

## Outputs

- `woman_e2e.png`
- `man_e2e.png`
- `output_combined_e2e.png`
- `qwen_full_e2e_profile.nsys-rep`
- `qwen_full_e2e_profile_detailed.nsys-rep`
- `qwen_full_e2e_profile_detailed.sqlite`

## Profile Summary

- Total GPU kernel time: `87113.19 ms`
- CUDA API `cuLaunchKernelEx`: `582549` calls, `66210.81 ms`
- CUDA API `cuMemcpyHtoDAsync_v2`: `7866` calls, `22297.49 ms`
- CUDA API `cuStreamSynchronize`: `8085` calls, `9372.84 ms`

Memcpy summary:

- HtoD (`copyKind=1`): `7866` calls, `21809.71 ms`, `158843.89 MiB`
- DtoH (`copyKind=2`): `9` calls, `3.33 ms`, `48.11 MiB`

Top GPU kernels by total time:

| Kernel | Calls | Total ms | Avg us |
| --- | ---: | ---: | ---: |
| `nn_mha_sm90__mha_sm90_DType6A6A6A6AcB6A6AsA_e609e354331829ff` | 12000 | 21707.48 | 1808.96 |
| `linalg_matmul_gpu_sm90_matmu6A6A6A6A6A6A6A6A_6effb095107ff004` | 12000 | 14027.50 | 1168.96 |
| `linalg_matmul_gpu_sm90_matmu6A6A6A6A6A6A6A6A_ab3d09c1c88076f3` | 19200 | 9717.83 | 506.14 |
| `nn_concat__fused_concat_inner_6A6A6A6A_2c56d37f507dad26` | 24000 | 7489.97 | 312.08 |
| `linalg_matmul_gpu_sm90_matmu6A6A6A6A6A6A6A6A_ddf5a62230c7bb48` | 4800 | 7324.38 | 1525.91 |
| `linalg_matmul_gpu_sm90_matmu6A6A6A6A6A6A6A6A_c5018a11ce849709` | 28800 | 3217.21 | 111.71 |
| `std_algorithm_backend_gpu_el6A6A6A6A6A6A6A6A_0aa393558e370523` | 24000 | 2685.18 | 111.88 |
| `std_algorithm_backend_gpu_el6A6A6A6A6A6A6A6A_f850c1afcb142161` | 24000 | 2680.57 | 111.69 |
| `linalg_matmul_gpu_sm90_matmu6A6A6A6A6A6A6A6A_6ba045ef370f502a` | 7200 | 2226.81 | 309.28 |
| `nn_normalization_rms_norm_gpu_6A6A6A6A_d916bc089e667269` | 48000 | 1464.59 | 30.51 |

## Observations

- The dominant overhead is still GPU launch count, not DtoH copies.
- DtoH traffic is low; HtoD traffic remains high.
- The main remaining opportunities are:
  - cache more small host-created buffers before execution
  - reduce helper kernel count for concat/tile/reshape paths
  - reduce repeated HtoD uploads for shape-derived and conditioning inputs
- `MODULAR_ENABLE_PROFILING=detailed` did not produce NVTX rows in `nsys stats`; analysis above is based on CUDA API and kernel tables.

## Recent Optimizations Included

- Moved multimodal text/vision merge to device-side scatter.
- Removed vision encoder output CPU round-trip.
- Moved prompt hidden-state trimming to compiled device helpers.
- Moved common repeat/tile cases (`num_images_per_prompt == 2`, `batch_size == 2`) to device helpers.
- Changed `scheduler_step` to use `shape_to_tensor(...)` instead of host `int(...)`.
- Added small scalar buffer caching for CFG scales.
- Added caches for multimodal vision auxiliary inputs and condition image IDs.
