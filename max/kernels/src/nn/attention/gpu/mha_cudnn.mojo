# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.ffi import (
    _DLHandle,
    _Global,
    _find_dylib,
    _get_global_or_null,
    OwnedDLHandle,
    c_char,
    external_call,
)
from std.gpu.host import DeviceBuffer, DeviceContext
from std.gpu.host._nvidia_cuda import CUDA
from std.memory import alloc
from std.os import abort
from std.os import getenv
from std.pathlib import Path
from std.runtime.tracing import Trace, TraceLevel, trace_arg
from std.sys import has_nvidia_gpu_accelerator
from std.utils import StaticTuple

from _cudnn.infer import (
    cudnnContext,
    cudnnCreate,
    cudnnDataType_t,
    cudnnDestroy,
    cudnnSetStream,
    cudnnStatus_t,
)
from layout import Layout, LayoutTensor


comptime _AnyOpaquePointer = OpaquePointer[AnyOrigin[mut=True]]
comptime _ExternalOpaquePointer = OpaquePointer[MutExternalOrigin]
comptime _CStringPtr = UnsafePointer[c_char, MutExternalOrigin]
comptime _Shape4 = Tuple[Int, Int, Int, Int]
comptime _DeviceBufferU8 = DeviceBuffer[DType.uint8]


def _cudnn_sdpa_frontend_dylib_paths() -> List[Path]:
    var paths = List[Path]()

    if lib_dir := getenv("MODULAR_CUDNN_SDPA_LIB_DIR"):
        var dir_path = Path(lib_dir)
        paths.append(dir_path / "libcudnn_sdpa_frontend_shim.so")

    if workspace_dir := getenv("BUILD_WORKSPACE_DIRECTORY"):
        paths.append(
            Path(workspace_dir)
            / "bazel-bin"
            / "max"
            / "kernels"
            / "src"
            / "nn"
            / "libcudnn_sdpa_frontend_shim.so"
        )

    if runfiles_dir := getenv("RUNFILES_DIR"):
        var runfiles_root = Path(runfiles_dir)
        paths.append(
            runfiles_root
            / "_main"
            / "max"
            / "kernels"
            / "src"
            / "nn"
            / "libcudnn_sdpa_frontend_shim.so"
        )

        if workspace_name := getenv("TEST_WORKSPACE"):
            paths.append(
                runfiles_root
                / workspace_name
                / "max"
                / "kernels"
                / "src"
                / "nn"
                / "libcudnn_sdpa_frontend_shim.so"
            )

    paths.append(
        Path("bazel-bin")
        / "max"
        / "kernels"
        / "src"
        / "nn"
        / "libcudnn_sdpa_frontend_shim.so"
    )
    paths.append(Path("libcudnn_sdpa_frontend_shim.so"))

    return paths^


comptime CUDA_CUDNN_SDPA_FRONTEND_LIBRARY = _Global[
    "CUDA_CUDNN_SDPA_FRONTEND_LIBRARY", _init_cudnn_sdpa_frontend_dylib
]


def _init_cudnn_sdpa_frontend_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        _cudnn_sdpa_frontend_dylib_paths()
    )


def _get_cudnn_sdpa_frontend_dylib() raises -> _DLHandle:
    var dylib_ptr = CUDA_CUDNN_SDPA_FRONTEND_LIBRARY.get_or_create_ptr()
    if not dylib_ptr[]:
        raise Error(
            "Unable to find libcudnn_sdpa_frontend_shim.so. Set MODULAR_CUDNN_SDPA_LIB_DIR or build MAX from the workspace root so bazel-bin/max/kernels/src/nn/libcudnn_sdpa_frontend_shim.so is present."
        )
    return dylib_ptr[].borrow()


@always_inline
def _check_cudnn_error(
    status: cudnnStatus_t, what: StaticString = "cuDNN"
) raises:
    if status != cudnnStatus_t.CUDNN_STATUS_SUCCESS:
        raise Error(
            what,
            " failed with raw cuDNN status ",
            Int(status._value),
        )


@always_inline
def _check_c_api_error(
    status: _CStringPtr, what: StaticString
) raises:
    if Int(status) != 0:
        raise Error(what, ": ", String(unsafe_from_utf8_ptr=status))


@always_inline
def _logical_bhsd_shape[
    dtype: DType,
    layout: Layout,
](tensor: LayoutTensor[
    dtype,
    layout,
    address_space=AddressSpace.GENERIC,
    ...,
]) -> _Shape4:
    return (
        tensor.dim[0](),
        tensor.dim[2](),
        tensor.dim[1](),
        tensor.dim[3](),
    )


@always_inline
def _logical_bhsd_stride[
    dtype: DType,
    layout: Layout,
](tensor: LayoutTensor[
    dtype,
    layout,
    address_space=AddressSpace.GENERIC,
    ...,
]) -> _Shape4:
    return (
        tensor.stride(0),
        tensor.stride(2),
        tensor.stride(1),
        tensor.stride(3),
    )


@always_inline
def _alignment_from_ptr[
    dtype: DType,
    origin: Origin,
](ptr: UnsafePointer[Scalar[dtype], origin]) -> Int:
    var addr = Int(ptr)
    if addr % 16 == 0:
        return 16
    if addr % 8 == 0:
        return 8
    if addr % 4 == 0:
        return 4
    if addr % 2 == 0:
        return 2
    return 1


@always_inline
def _to_i64_shape(shape: _Shape4) -> StaticTuple[Int64, 4]:
    return StaticTuple[Int64, 4](
        Int64(shape[0]),
        Int64(shape[1]),
        Int64(shape[2]),
        Int64(shape[3]),
    )


struct CuDNNSDPAMeta(ImplicitlyCopyable, RegisterPassable):
    var ptr_handle: UnsafePointer[cudnnContext, AnyOrigin[mut=True]]
    var plan: _ExternalOpaquePointer
    var workspace_size: Int
    var is_set: Bool
    var scale: Float32
    var q_shape: _Shape4
    var q_stride: _Shape4
    var k_shape: _Shape4
    var k_stride: _Shape4
    var v_shape: _Shape4
    var v_stride: _Shape4
    var o_shape: _Shape4
    var o_stride: _Shape4
    var q_alignment: Int
    var k_alignment: Int
    var v_alignment: Int
    var o_alignment: Int

    def __init__(out self) raises:
        self.ptr_handle = UnsafePointer[cudnnContext, AnyOrigin[mut=True]]()
        _check_cudnn_error(
            cudnnCreate(UnsafePointer(to=self.ptr_handle)),
            "cudnnCreate",
        )
        self.plan = _ExternalOpaquePointer()
        self.workspace_size = 0
        self.is_set = False
        self.scale = 0.0
        self.q_shape = (0, 0, 0, 0)
        self.q_stride = (0, 0, 0, 0)
        self.k_shape = (0, 0, 0, 0)
        self.k_stride = (0, 0, 0, 0)
        self.v_shape = (0, 0, 0, 0)
        self.v_stride = (0, 0, 0, 0)
        self.o_shape = (0, 0, 0, 0)
        self.o_stride = (0, 0, 0, 0)
        self.q_alignment = 0
        self.k_alignment = 0
        self.v_alignment = 0
        self.o_alignment = 0

        var dylib = _get_cudnn_sdpa_frontend_dylib()
        _check_c_api_error(
            dylib.call["max_cudnn_sdpa_plan_create", _CStringPtr](
                UnsafePointer(to=self.plan)
            ),
            "max_cudnn_sdpa_plan_create",
        )

    def reset(mut self) raises:
        if Int(self.plan) != 0:
            _get_cudnn_sdpa_frontend_dylib().call[
                "max_cudnn_sdpa_plan_destroy", NoneType
            ](
                self.plan.bitcast[NoneType]()
            )
            self.plan = _ExternalOpaquePointer()
        self.workspace_size = 0
        self.is_set = False

    def __del__(deinit self):
        try:
            self.reset()
            if Int(self.ptr_handle) != 0:
                _check_cudnn_error(cudnnDestroy(self.ptr_handle), "cudnnDestroy")
        except e:
            abort(String(e))


def _get_cudnn_sdpa_meta(
    ctx: DeviceContext,
) raises -> UnsafePointer[CuDNNSDPAMeta, AnyOrigin[mut=True]]:
    var cache_key = "CUDA_CUDNN_SDPA_META_CACHE_" + String(ctx.id())

    if ptr_meta := _get_global_or_null(cache_key):
        var ptr = ptr_meta.unsafe_value().bitcast[CuDNNSDPAMeta]()
        _check_cudnn_error(
            cudnnSetStream(ptr[].ptr_handle, CUDA(ctx.stream())),
            "cudnnSetStream",
        )
        return ptr

    var new_ptr_meta = alloc[CuDNNSDPAMeta](1)
    new_ptr_meta.init_pointee_move(CuDNNSDPAMeta())

    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_key),
        new_ptr_meta.bitcast[NoneType](),
    )

    _check_cudnn_error(
        cudnnSetStream(new_ptr_meta[].ptr_handle, CUDA(ctx.stream())),
        "cudnnSetStream",
    )
    return new_ptr_meta


def flash_attention_cudnn_supported(
    q: LayoutTensor[address_space=AddressSpace.GENERIC, ...],
    k: LayoutTensor[address_space=AddressSpace.GENERIC, ...],
    v: LayoutTensor[address_space=AddressSpace.GENERIC, ...],
    output: LayoutTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
) -> Bool:
    if not has_nvidia_gpu_accelerator():
        return False
    comptime if (
        q.dtype != DType.bfloat16
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or output.dtype != q.dtype
        or q.rank != 4
        or k.rank != 4
        or v.rank != 4
        or output.rank != 4
    ):
        return False

    if Int(q.ptr) == 0 or Int(k.ptr) == 0 or Int(v.ptr) == 0 or Int(output.ptr) == 0:
        return False

    if q.dim[0]() != k.dim[0]() or q.dim[0]() != v.dim[0]() or q.dim[0]() != output.dim[0]():
        return False
    if k.dim[1]() != v.dim[1]():
        return False
    if q.dim[1]() != output.dim[1]():
        return False
    if q.dim[2]() <= 0 or k.dim[2]() <= 0 or v.dim[2]() <= 0:
        return False
    if q.dim[3]() != k.dim[3]():
        return False
    if v.dim[3]() != output.dim[3]():
        return False

    # The Klein DiT path is dense BSHD with a contiguous head dimension.
    if q.stride(3) != 1 or k.stride(3) != 1 or v.stride(3) != 1 or output.stride(3) != 1:
        return False

    return True


def _require_flash_attention_cudnn_supported(
    q: LayoutTensor[address_space=AddressSpace.GENERIC, ...],
    k: LayoutTensor[address_space=AddressSpace.GENERIC, ...],
    v: LayoutTensor[address_space=AddressSpace.GENERIC, ...],
    output: LayoutTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
) raises:
    if not flash_attention_cudnn_supported(q, k, v, output):
        raise Error(
            "CuDNN attention backend only supports dense rank-4 NVIDIA bf16 no-mask attention with contiguous head dimension"
        )


def _build_or_reuse_cudnn_plan(
    meta: UnsafePointer[CuDNNSDPAMeta, AnyOrigin[mut=True]],
    q: LayoutTensor[address_space=AddressSpace.GENERIC, ...],
    k: LayoutTensor[address_space=AddressSpace.GENERIC, ...],
    v: LayoutTensor[address_space=AddressSpace.GENERIC, ...],
    output: LayoutTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    scale: Float32,
) raises:
    var q_shape = _logical_bhsd_shape(q)
    var q_stride = _logical_bhsd_stride(q)
    var k_shape = _logical_bhsd_shape(k)
    var k_stride = _logical_bhsd_stride(k)
    var v_shape = _logical_bhsd_shape(v)
    var v_stride = _logical_bhsd_stride(v)
    var o_shape = _logical_bhsd_shape(output)
    var o_stride = _logical_bhsd_stride(output)

    var q_alignment = _alignment_from_ptr(q.ptr)
    var k_alignment = _alignment_from_ptr(k.ptr)
    var v_alignment = _alignment_from_ptr(v.ptr)
    var o_alignment = _alignment_from_ptr(output.ptr)

    if (
        meta[].is_set
        and meta[].scale == scale
        and meta[].q_shape == q_shape
        and meta[].q_stride == q_stride
        and meta[].k_shape == k_shape
        and meta[].k_stride == k_stride
        and meta[].v_shape == v_shape
        and meta[].v_stride == v_stride
        and meta[].o_shape == o_shape
        and meta[].o_stride == o_stride
        and meta[].q_alignment == q_alignment
        and meta[].k_alignment == k_alignment
        and meta[].v_alignment == v_alignment
        and meta[].o_alignment == o_alignment
    ):
        return

    var workspace_size = Int64(0)
    var dylib = _get_cudnn_sdpa_frontend_dylib()

    _check_c_api_error(
        dylib.call["max_cudnn_sdpa_plan_build", _CStringPtr](
            meta[].plan.bitcast[NoneType](),
            meta[].ptr_handle.bitcast[NoneType](),
            scale,
            Int64(q_shape[0]),
            Int64(q_shape[1]),
            Int64(q_shape[2]),
            Int64(q_shape[3]),
            Int64(q_stride[0]),
            Int64(q_stride[1]),
            Int64(q_stride[2]),
            Int64(q_stride[3]),
            Int64(q_alignment),
            Int64(k_shape[0]),
            Int64(k_shape[1]),
            Int64(k_shape[2]),
            Int64(k_shape[3]),
            Int64(k_stride[0]),
            Int64(k_stride[1]),
            Int64(k_stride[2]),
            Int64(k_stride[3]),
            Int64(k_alignment),
            Int64(v_shape[0]),
            Int64(v_shape[1]),
            Int64(v_shape[2]),
            Int64(v_shape[3]),
            Int64(v_stride[0]),
            Int64(v_stride[1]),
            Int64(v_stride[2]),
            Int64(v_stride[3]),
            Int64(v_alignment),
            Int64(o_shape[0]),
            Int64(o_shape[1]),
            Int64(o_shape[2]),
            Int64(o_shape[3]),
            Int64(o_stride[0]),
            Int64(o_stride[1]),
            Int64(o_stride[2]),
            Int64(o_stride[3]),
            Int64(o_alignment),
            UnsafePointer(to=workspace_size),
        ),
        "max_cudnn_sdpa_plan_build",
    )

    meta[].workspace_size = Int(workspace_size)
    meta[].is_set = True
    meta[].scale = scale
    meta[].q_shape = q_shape
    meta[].q_stride = q_stride
    meta[].k_shape = k_shape
    meta[].k_stride = k_stride
    meta[].v_shape = v_shape
    meta[].v_stride = v_stride
    meta[].o_shape = o_shape
    meta[].o_stride = o_stride
    meta[].q_alignment = q_alignment
    meta[].k_alignment = k_alignment
    meta[].v_alignment = v_alignment
    meta[].o_alignment = o_alignment


def flash_attention_cudnn(
    q: LayoutTensor[address_space=AddressSpace.GENERIC, ...],
    k: LayoutTensor[address_space=AddressSpace.GENERIC, ...],
    v: LayoutTensor[address_space=AddressSpace.GENERIC, ...],
    output: LayoutTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    scale: Float32,
    ctx: DeviceContext,
) raises:
    _require_flash_attention_cudnn_supported(q, k, v, output)

    @always_inline
    @parameter
    def description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q", q.runtime_layout.shape.value),
                    trace_arg("k", k.runtime_layout.shape.value),
                    trace_arg("v", v.runtime_layout.shape.value),
                    trace_arg("output", output.runtime_layout.shape.value),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flash_attention_cudnn",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        var meta = _get_cudnn_sdpa_meta(ctx)
        _build_or_reuse_cudnn_plan(meta, q, k, v, output, scale)

        var workspace_buffer = _DeviceBufferU8(ctx, {}, 0, owning=False)
        var workspace_ptr = _ExternalOpaquePointer()
        if meta[].workspace_size > 0:
            workspace_buffer = ctx.enqueue_create_buffer[DType.uint8](
                meta[].workspace_size
            )
            workspace_ptr = workspace_buffer.unsafe_ptr().unsafe_origin_cast[
                MutExternalOrigin
            ]().bitcast[NoneType]()

        var dylib = _get_cudnn_sdpa_frontend_dylib()
        _check_c_api_error(
            dylib.call["max_cudnn_sdpa_plan_execute", _CStringPtr](
                meta[].plan.bitcast[NoneType](),
                meta[].ptr_handle.bitcast[NoneType](),
                q.ptr.bitcast[NoneType](),
                k.ptr.bitcast[NoneType](),
                v.ptr.bitcast[NoneType](),
                output.ptr.bitcast[NoneType](),
                workspace_ptr.bitcast[NoneType](),
            ),
            "max_cudnn_sdpa_plan_execute",
        )
