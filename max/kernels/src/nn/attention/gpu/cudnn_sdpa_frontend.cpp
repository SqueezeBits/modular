#include <dlfcn.h>

#include <cstdio>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#include "cudnn_frontend.h"

#if defined(__GNUC__)
#define MAX_CUDNN_SDPA_EXPORT __attribute__((visibility("default")))
#else
#define MAX_CUDNN_SDPA_EXPORT
#endif

namespace {

thread_local std::array<char, 4096> g_last_error = {};

const char* set_error(std::string message) {
  std::snprintf(g_last_error.data(), g_last_error.size(), "%s",
                message.c_str());
  return g_last_error.data();
}

void check_status(cudnn_frontend::error_t status, const char* what) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(what) + ": " + status.get_message());
  }
}

void ensure_cudnn_loaded() {
#if defined(NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)
  if (cudnn_frontend::cudnn_dlhandle != nullptr) {
    return;
  }

  constexpr std::array<const char*, 3> kCandidateLibs = {
      "libcudnn.so.9",
      "libcudnn.so",
      "libcudnn.so.8",
  };

  for (const char* lib : kCandidateLibs) {
    if (void* handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL)) {
      cudnn_frontend::cudnn_dlhandle = handle;
      return;
    }
  }

  const char* err = dlerror();
  throw std::runtime_error(
      std::string("Unable to dlopen cuDNN shared library: ") +
      (err != nullptr ? err : "unknown error"));
#endif
}

using TensorAttrs = cudnn_frontend::graph::Tensor_attributes;
using Graph = cudnn_frontend::graph::Graph;

struct TensorDescriptor {
  std::array<int64_t, 4> dim;
  std::array<int64_t, 4> stride;
  int64_t alignment;
};

TensorDescriptor make_tensor_descriptor(int64_t d0, int64_t d1, int64_t d2,
                                        int64_t d3, int64_t s0, int64_t s1,
                                        int64_t s2, int64_t s3,
                                        int64_t alignment) {
  return {{
              d0,
              d1,
              d2,
              d3,
          },
          {
              s0,
              s1,
              s2,
              s3,
          },
          alignment};
}

std::shared_ptr<TensorAttrs> add_input_tensor(Graph& graph, const char* name,
                                              const TensorDescriptor& desc) {
  return graph.tensor(TensorAttrs()
                          .set_name(name)
                          .set_dim(std::vector<int64_t>(
                              desc.dim.begin(), desc.dim.end()))
                          .set_stride(std::vector<int64_t>(
                              desc.stride.begin(), desc.stride.end()))
                          .set_data_type(cudnn_frontend::DataType_t::BFLOAT16)
                          .set_alignment(desc.alignment));
}

void configure_output_tensor(const std::shared_ptr<TensorAttrs>& tensor,
                             const TensorDescriptor& desc) {
  tensor->set_output(true)
      .set_dim(std::vector<int64_t>(desc.dim.begin(), desc.dim.end()))
      .set_stride(
          std::vector<int64_t>(desc.stride.begin(), desc.stride.end()))
      .set_data_type(cudnn_frontend::DataType_t::BFLOAT16)
      .set_alignment(desc.alignment);
}

struct CuDnnSdpaPlan {
  std::unique_ptr<Graph> graph;
  std::shared_ptr<TensorAttrs> q;
  std::shared_ptr<TensorAttrs> k;
  std::shared_ptr<TensorAttrs> v;
  std::shared_ptr<TensorAttrs> o;
  int64_t workspace_size = 0;
  bool built = false;

  void build(cudnnHandle_t handle, float scale, const TensorDescriptor& q_desc,
             const TensorDescriptor& k_desc, const TensorDescriptor& v_desc,
             const TensorDescriptor& o_desc) {
    ensure_cudnn_loaded();

    graph = std::make_unique<Graph>();
    graph->set_name("max_flash_attention_cudnn")
        .set_io_data_type(cudnn_frontend::DataType_t::BFLOAT16)
        .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
        .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    q = add_input_tensor(*graph, "Q", q_desc);
    k = add_input_tensor(*graph, "K", k_desc);
    v = add_input_tensor(*graph, "V", v_desc);

    auto outputs = graph->sdpa(
        q, k, v,
        cudnn_frontend::graph::SDPA_attributes()
            .set_name("sdpa")
            .set_is_inference(true)
            .set_attn_scale(scale)
            .set_implementation(
                cudnn_frontend::AttentionImplementation_t::COMPOSITE));
    o = outputs[0];
    configure_output_tensor(o, o_desc);

    std::vector<cudnn_frontend::HeurMode_t> heuristics = {
        cudnn_frontend::HeurMode_t::A,
        cudnn_frontend::HeurMode_t::FALLBACK,
    };
    check_status(graph->build(handle, heuristics), "cudnn_frontend::Graph::build");
    workspace_size = graph->get_workspace_size();
    built = true;
  }

  void execute(cudnnHandle_t handle, const void* q_ptr, const void* k_ptr,
               const void* v_ptr, void* o_ptr, void* workspace_ptr) const {
    if (!built || !graph || !q || !k || !v || !o) {
      throw std::runtime_error("cuDNN SDPA plan has not been built");
    }

    std::unordered_map<std::shared_ptr<TensorAttrs>, void*> tensor_ptrs = {
        {q, const_cast<void*>(q_ptr)},
        {k, const_cast<void*>(k_ptr)},
        {v, const_cast<void*>(v_ptr)},
        {o, o_ptr},
    };
    check_status(graph->execute(handle, tensor_ptrs, workspace_ptr),
                 "cudnn_frontend::Graph::execute");
  }
};

template <typename Func>
const char* wrap_errors(Func&& fn) {
  try {
    fn();
    return nullptr;
  } catch (const cudnn_frontend::cudnnException& e) {
    return set_error(std::string(e.what()));
  } catch (const std::exception& e) {
    return set_error(std::string(e.what()));
  } catch (...) {
    return set_error("unknown cuDNN SDPA error");
  }
}

}  // namespace

#if defined(NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)
namespace cudnn_frontend {
void* cudnn_dlhandle = nullptr;
}  // namespace cudnn_frontend
#endif

extern "C" MAX_CUDNN_SDPA_EXPORT const char* max_cudnn_sdpa_plan_create(
    void** out_plan) {
  return wrap_errors([&] {
    if (out_plan == nullptr) {
      throw std::runtime_error("out_plan must not be null");
    }
    *out_plan = new CuDnnSdpaPlan();
  });
}

extern "C" MAX_CUDNN_SDPA_EXPORT void max_cudnn_sdpa_plan_destroy(void* plan) {
  delete static_cast<CuDnnSdpaPlan*>(plan);
}

extern "C" MAX_CUDNN_SDPA_EXPORT const char* max_cudnn_sdpa_plan_build(
    void* plan, void* cudnn_handle, float scale, int64_t q_dim0, int64_t q_dim1,
    int64_t q_dim2, int64_t q_dim3, int64_t q_stride0, int64_t q_stride1,
    int64_t q_stride2, int64_t q_stride3, int64_t q_alignment, int64_t k_dim0,
    int64_t k_dim1, int64_t k_dim2, int64_t k_dim3, int64_t k_stride0,
    int64_t k_stride1, int64_t k_stride2, int64_t k_stride3,
    int64_t k_alignment, int64_t v_dim0, int64_t v_dim1, int64_t v_dim2,
    int64_t v_dim3, int64_t v_stride0, int64_t v_stride1, int64_t v_stride2,
    int64_t v_stride3, int64_t v_alignment, int64_t o_dim0, int64_t o_dim1,
    int64_t o_dim2, int64_t o_dim3, int64_t o_stride0, int64_t o_stride1,
    int64_t o_stride2, int64_t o_stride3, int64_t o_alignment,
    int64_t* workspace_size) {
  return wrap_errors([&] {
    if (plan == nullptr) {
      throw std::runtime_error("plan must not be null");
    }
    if (cudnn_handle == nullptr) {
      throw std::runtime_error("cudnn_handle must not be null");
    }
    if (workspace_size == nullptr) {
      throw std::runtime_error("workspace_size must not be null");
    }

    auto* typed_plan = static_cast<CuDnnSdpaPlan*>(plan);
    typed_plan->build(
        static_cast<cudnnHandle_t>(cudnn_handle), scale,
        make_tensor_descriptor(q_dim0, q_dim1, q_dim2, q_dim3, q_stride0,
                               q_stride1, q_stride2, q_stride3, q_alignment),
        make_tensor_descriptor(k_dim0, k_dim1, k_dim2, k_dim3, k_stride0,
                               k_stride1, k_stride2, k_stride3, k_alignment),
        make_tensor_descriptor(v_dim0, v_dim1, v_dim2, v_dim3, v_stride0,
                               v_stride1, v_stride2, v_stride3, v_alignment),
        make_tensor_descriptor(o_dim0, o_dim1, o_dim2, o_dim3, o_stride0,
                               o_stride1, o_stride2, o_stride3, o_alignment));
    *workspace_size = typed_plan->workspace_size;
  });
}

extern "C" MAX_CUDNN_SDPA_EXPORT const char* max_cudnn_sdpa_plan_execute(
    void* plan, void* cudnn_handle, const void* q_ptr, const void* k_ptr,
    const void* v_ptr, void* o_ptr, void* workspace_ptr) {
  return wrap_errors([&] {
    if (plan == nullptr) {
      throw std::runtime_error("plan must not be null");
    }
    if (cudnn_handle == nullptr) {
      throw std::runtime_error("cudnn_handle must not be null");
    }
    if (q_ptr == nullptr || k_ptr == nullptr || v_ptr == nullptr ||
        o_ptr == nullptr) {
      throw std::runtime_error("Q/K/V/O pointers must not be null");
    }

    auto* typed_plan = static_cast<CuDnnSdpaPlan*>(plan);
    typed_plan->execute(static_cast<cudnnHandle_t>(cudnn_handle), q_ptr, k_ptr,
                        v_ptr, o_ptr, workspace_ptr);
  });
}
