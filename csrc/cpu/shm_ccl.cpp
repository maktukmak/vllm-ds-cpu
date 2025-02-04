#include "cpu/cpu_types.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
#define MAX_SHM_RANK_NUM 8
#define MAX_THREAD_NUM 12
#define PER_THREAD_SHM_BUFFER_BYTES (2 * 1024 * 1024)
#define MIN_THREAD_PROCESS_SIZE (8 * 1024)

template <typename scalar_t>
struct KernelVecType {
  using scalar_vec_t = void;
};

template <>
struct KernelVecType<float> {
  using scalar_vec_t = vec_op::FP32Vec16;
};

template <>
struct KernelVecType<c10::BFloat16> {
  using scalar_vec_t = vec_op::BF16Vec16;
};

template <>
struct KernelVecType<c10::Half> {
  using scalar_vec_t = vec_op::FP16Vec16;
};

enum class ThreadSHMStat : char { THREAD_READY = 0, SHM_DATA_READY, DONE };

struct ThreadSHMContext {
  volatile ThreadSHMStat thread_stats[MAX_SHM_RANK_NUM];
  int thread_id;
  int thread_num;
  int rank;
  int group_size;
  size_t _spinning_count;
  int swizzled_ranks[MAX_SHM_RANK_NUM];
  void* thread_shm_ptrs[MAX_SHM_RANK_NUM];
  ThreadSHMContext* shm_contexts[MAX_SHM_RANK_NUM];

  ThreadSHMContext(const int thread_id, const int thread_num, const int rank,
                   const int group_size, void* thread_shm_ptr)
      : thread_id(thread_id),
        thread_num(thread_num),
        rank(rank),
        group_size(group_size),
        _spinning_count(0) {
    static_assert(sizeof(ThreadSHMContext) % 64 == 0);
    TORCH_CHECK(group_size <= MAX_SHM_RANK_NUM);
    TORCH_CHECK((size_t)this % 64 == 0);
    TORCH_CHECK((size_t)thread_shm_ptr % 64 == 0);
    for (int i = 0; i < MAX_SHM_RANK_NUM; ++i) {
      shm_contexts[i] = nullptr;
      thread_shm_ptrs[i] = nullptr;
      swizzled_ranks[i] = (i + rank) % group_size;
      thread_stats[i] = ThreadSHMStat::DONE;
    }
    set_context(rank, this, thread_shm_ptr);
  }

  void set_context(int rank, ThreadSHMContext* ptr, void* thread_shm_ptr) {
    TORCH_CHECK(rank < MAX_SHM_RANK_NUM);
    TORCH_CHECK(ptr);
    TORCH_CHECK(thread_shm_ptr);
    TORCH_CHECK_EQ(ptr->thread_num, thread_num);
    TORCH_CHECK_EQ(ptr->thread_id, thread_id);
    shm_contexts[rank] = ptr;
    thread_shm_ptrs[rank] = thread_shm_ptr;
  }

  template <typename T>
  T* get_thread_shm_ptr(int rank) {
    return reinterpret_cast<T*>(thread_shm_ptrs[rank]);
  }

  int get_swizzled_rank(int idx) { return swizzled_ranks[idx]; }

  void wait_for_all(ThreadSHMStat prev_stat) {
    for (int idx = 1; idx < group_size; ++idx) {
      int rank = get_swizzled_rank(idx);
      while (thread_stats[rank] == prev_stat) {
        ++_spinning_count;
        _mm_pause();
      }
    }
  }

  void set_thread_stat(ThreadSHMStat stat) {
    for (int idx = 1; idx < group_size; ++idx) {
      int rank = get_swizzled_rank(idx);
      shm_contexts[rank]->thread_stats[this->rank] = stat;
    }
  }

  // DONE -> THREAD_READY -> SHM_DATA_READY -> DONE -> ...
  void barrier(ThreadSHMStat next_stat) {
    if (next_stat == ThreadSHMStat::THREAD_READY) {
      set_thread_stat(ThreadSHMStat::THREAD_READY);
      wait_for_all(ThreadSHMStat::DONE);
    } else if (next_stat == ThreadSHMStat::SHM_DATA_READY) {
      set_thread_stat(ThreadSHMStat::SHM_DATA_READY);
      wait_for_all(ThreadSHMStat::THREAD_READY);
    } else if (next_stat == ThreadSHMStat::DONE) {
      set_thread_stat(ThreadSHMStat::DONE);
      wait_for_all(ThreadSHMStat::SHM_DATA_READY);
    } else {
      TORCH_CHECK(false, "Invalid next_stat to barrier.");
    }
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "SHMContext:";
    ss << "\nrank: " << rank;
    ss << "\ngroup_size: " << group_size;
    ss << "\nthread_num: " << thread_num;
    ss << "\nthread_id: " << thread_id;

    ss << "\nshm_ctx_stat_loop_seq: [";
    for (int i = 0; i < group_size; ++i) {
      ss << swizzled_ranks[i] << ", ";
    }
    ss << "]";

    ss << "\nshm_contexts: [";
    for (int i = 0; i < group_size; ++i) {
      if (shm_contexts[i]) {
        ss << shm_contexts[i]->rank << ", ";
      }
    }
    ss << "]";

    return ss.str();
  }
};

class SHMManager {
 public:
  explicit SHMManager(const std::string& name, const int rank,
                      const int group_size)
      : _rank(rank),
        _group_size(group_size),
        _thread_num(std::min(torch::get_num_threads(), MAX_THREAD_NUM)),
        _shm_names({""}),
        _shared_mem_ptrs({nullptr}),
        _shm_ctx(nullptr) {
    _shm_names[rank] = get_shm_name(name, rank);
    _shared_mem_ptrs[rank] = init_shm(rank);
    _shm_ctx = reinterpret_cast<ThreadSHMContext*>(_shared_mem_ptrs[rank]);

    for (int i = 0; i < _thread_num; ++i) {
      ThreadSHMContext* ctx = new (_shm_ctx + i)
          ThreadSHMContext(i, _thread_num, _rank, _group_size,
                           compute_thread_shm_ptr(_shm_ctx, i));
    }
  }

  void join(const std::string& name) {
    for (int rank_idx = 0; rank_idx < _group_size; ++rank_idx) {
      if (rank_idx != _rank) {
        TORCH_CHECK(_shm_names[rank_idx].empty());
        TORCH_CHECK(_shared_mem_ptrs[rank_idx] == nullptr);
        _shm_names[rank_idx] = get_shm_name(name, rank_idx);
        _shared_mem_ptrs[rank_idx] = init_shm(rank_idx);
        ThreadSHMContext* target_ctx =
            reinterpret_cast<ThreadSHMContext*>(_shared_mem_ptrs[rank_idx]);
        for (int thread_idx = 0; thread_idx < _thread_num; ++thread_idx) {
          _shm_ctx[thread_idx].set_context(
              rank_idx, target_ctx + thread_idx,
              compute_thread_shm_ptr(target_ctx, thread_idx));
        }
      }
    }
  }

  ~SHMManager() { destroy_shm(); }

  ThreadSHMContext* get_shm_ctx() const { return _shm_ctx; }

  static std::string get_shm_name(const std::string& name, int rank) {
    return name + "_" + std::to_string(rank);
  }

  static void create_singleton_instance(const std::string& name,
                                        const int group_size, const int rank) {
    std::call_once(SingletonCreationFlag, [&]() {
      TORCH_CHECK(
          SingletonInstance == nullptr,
          "Duplicate initialization of SHMManager singleton is not allowed.");
      SingletonInstance = std::make_unique<SHMManager>(name, rank, group_size);
    });
  }

  static SHMManager* get_singleton_instance() {
    return SingletonInstance.get();
  }

 protected:
  static std::unique_ptr<SHMManager> SingletonInstance;
  static std::once_flag SingletonCreationFlag;

 private:
  static size_t round_to_alignment(size_t num) {
    return ((num + 63) / 64) * 64;
  }

  int8_t* compute_thread_shm_ptr(ThreadSHMContext* ctx, int thread_id) {
    int8_t* thread_shm_ptr =
        reinterpret_cast<int8_t*>(ctx) +
        round_to_alignment(_thread_num * sizeof(ThreadSHMContext));
    return thread_shm_ptr +
           thread_id * round_to_alignment(PER_THREAD_SHM_BUFFER_BYTES);
  }

  size_t compute_shm_size() {
    const size_t rounded_rank_buffer_size =
        round_to_alignment(PER_THREAD_SHM_BUFFER_BYTES) * _thread_num;
    const size_t rounded_thread_shm_ctx_size =
        round_to_alignment(_thread_num * sizeof(ThreadSHMContext));
    const size_t shm_size =
        rounded_thread_shm_ctx_size + rounded_rank_buffer_size;
    return shm_size;
  }

  void* init_shm(int target_rank) {
    const std::string& shm_name = _shm_names[target_rank];
    const int local_rank = _rank;
    const size_t shm_size = compute_shm_size();

    int fd = -1;
    if (local_rank == target_rank) {
      fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR,
                    S_IRUSR | S_IWUSR);

      if (fd == -1)
        TORCH_CHECK(false, "create shm in SHMManager failed. errno: " +
                               std::to_string(errno));

      if (ftruncate(fd, shm_size) == -1)
        TORCH_CHECK(false, "ftruncate in SHMManager failed. errno: " +
                               std::to_string(errno));
    } else {
      fd = shm_open(shm_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);

      if (fd == -1)
        TORCH_CHECK(false, "open shm in SHMManager failed. errno: " +
                               std::to_string(errno));
    }

    void* shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_POPULATE, fd, 0);

    if (shm_ptr == MAP_FAILED) {
      TORCH_CHECK(false,
                  "mmap in SHMManager failed. errno: " + std::to_string(errno));
    }

    if (close(fd) != 0) {
      TORCH_CHECK(
          false, "close in SHMManager failed. errno: " + std::to_string(errno));
    }

    TORCH_CHECK((size_t)shm_ptr % 64 == 0);

    return shm_ptr;
  }

  void destroy_shm() {
    std::stringstream ss;
    ss << "local rank " << _rank << ": [";
    for (int thread_id = 0; thread_id < _thread_num; ++thread_id) {
      ss << _shm_ctx[thread_id]._spinning_count << ", ";
    }
    ss << "]\n";
    std::printf("%s\n", ss.str().c_str());

    for (int i = 0; i < MAX_SHM_RANK_NUM; ++i) {
      if (_shared_mem_ptrs[i] != nullptr) {
        munmap(_shared_mem_ptrs[i], compute_shm_size());
      }

      if (!_shm_names[i].empty()) {
        shm_unlink(_shm_names[i].c_str());
      }
    }
  }

  int _rank;
  int _group_size;
  int _thread_num;
  std::array<std::string, MAX_SHM_RANK_NUM> _shm_names;
  std::array<void*, MAX_SHM_RANK_NUM> _shared_mem_ptrs;
  ThreadSHMContext* _shm_ctx;
};

namespace shm_cc_ops {
template <typename scalar_t, typename F>
void shm_cc_loop(ThreadSHMContext* ctx, int64_t elem_num, F&& inner_func) {
  int thread_num = ctx->thread_num;
  int64_t total_bytes = elem_num * sizeof(scalar_t);
  int64_t total_units_num =
      (total_bytes + MIN_THREAD_PROCESS_SIZE - 1) / MIN_THREAD_PROCESS_SIZE;
  int64_t per_thread_units_num =
      (total_units_num + thread_num - 1) / thread_num;
  int64_t per_unit_elem_num = MIN_THREAD_PROCESS_SIZE / sizeof(scalar_t);
  int64_t max_per_thread_iteration_elem_num =
      PER_THREAD_SHM_BUFFER_BYTES / sizeof(scalar_t);
  int64_t per_thread_elem_num = per_unit_elem_num * per_thread_units_num;

#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < thread_num; ++i) {
    int64_t offset = i * per_thread_elem_num;
    int64_t end = std::min(elem_num, offset + per_thread_elem_num);
    int64_t curr_elem_num =
        std::min(max_per_thread_iteration_elem_num, end - offset);
    ThreadSHMContext* thread_ctx = ctx + i;

    while (curr_elem_num > 0) {
      inner_func(thread_ctx, offset, curr_elem_num);

      offset += max_per_thread_iteration_elem_num;
      curr_elem_num = std::min(max_per_thread_iteration_elem_num, end - offset);
    }
  }
}
};  // namespace shm_cc_ops

namespace shm_cc_ops {

void memcpy_from_shm(void* dst, void* src, const int64_t bytes) {
  const int64_t aligned_bytes = ((bytes >> 6) << 6);  // 64 bytes aligned
  int64_t i = 0;
#pragma GCC unroll 4
  for (; i < aligned_bytes; i += 64) {
    vec_op::INT8Vec64 data(
        true, (int8_t*)src + i);  // stream loading shm to avoid caching
    data.save((int8_t*)dst + i);
  }
  if (aligned_bytes < bytes) {
    vec_op::INT8Vec64 data(true, (int8_t*)src + aligned_bytes);
    data.save((int8_t*)dst + aligned_bytes, bytes - aligned_bytes);
  }
}

void memcpy_to_shm(void* dst, void* src, const int64_t bytes) {
#pragma GCC unroll 4
  for (int64_t i = 0; i < bytes; i += 64) {
    vec_op::INT8Vec64 data((int8_t*)src + i);
    data.nt_save((int8_t*)dst + i);
  }
}

void memcpy(void* dst, void* src, const int64_t bytes) {
  const int64_t aligned_bytes = ((bytes >> 6) << 6);  // 64 bytes aligned
  int64_t i = 0;
#pragma GCC unroll 4
  for (; i < aligned_bytes; i += 64) {
    vec_op::INT8Vec64 data((int8_t*)src + i);
    data.save((int8_t*)dst + i);
  }
  if (aligned_bytes < bytes) {
    vec_op::INT8Vec64 data((int8_t*)src + aligned_bytes);
    data.save((int8_t*)dst + aligned_bytes, bytes - aligned_bytes);
  }
}

template <typename scalar_t, int RANKS>
void all_reduce_sum_impl(ThreadSHMContext* ctx, scalar_t* data,
                         size_t elem_num) {
  CPU_KERNEL_GUARD_IN(all_reduce_sum_impl)
  using vec_t = typename KernelVecType<scalar_t>::scalar_vec_t;
  constexpr int64_t vec_elem_num = vec_t::get_elem_num();
  const int worldsize = ctx->group_size;

  shm_cc_ops::shm_cc_loop<scalar_t>(
      ctx, elem_num,
      [&](ThreadSHMContext* thread_ctx, int64_t data_offset,
          int64_t data_elem_num) {
        int rank = thread_ctx->rank;
        scalar_t* thread_shm_ptr =
            thread_ctx->get_thread_shm_ptr<scalar_t>(rank);
        scalar_t* thread_data_ptr = data + data_offset;
        int64_t thread_data_elem_num = data_elem_num * sizeof(scalar_t);

        scalar_t* remote_data_ptrs[RANKS - 1];
        vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
          remote_data_ptrs[idx] = thread_ctx->get_thread_shm_ptr<scalar_t>(
              thread_ctx->get_swizzled_rank(idx + 1));
        });

        thread_ctx->barrier(ThreadSHMStat::THREAD_READY);

        shm_cc_ops::memcpy_to_shm(thread_shm_ptr, thread_data_ptr,
                                  thread_data_elem_num);

        thread_ctx->barrier(ThreadSHMStat::SHM_DATA_READY);

        int64_t aligned_data_elem_num =
            (data_elem_num / vec_elem_num) * vec_elem_num;
        int64_t i = 0;
#pragma GCC unroll 4
        for (; i < aligned_data_elem_num; i += vec_elem_num) {
          vec_t local_data(thread_data_ptr + i);  // load from cache
          vec_op::FP32Vec16 local_data_fp32(local_data);
          vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
            vec_t remote_data(
                true, remote_data_ptrs[idx] + i);  // stream load from shm
            vec_op::FP32Vec16 remote_data_fp32(remote_data);
            local_data_fp32 = local_data_fp32 + remote_data_fp32;  // sum reduce
          });
          vec_t reduced_data(local_data_fp32);
          reduced_data.save(thread_data_ptr + i);
        }

        if (i < data_elem_num) {
          vec_t local_data(thread_data_ptr + i);  // load from cache
          vec_op::FP32Vec16 local_data_fp32(local_data);
          vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
            vec_t remote_data(
                true, remote_data_ptrs[idx] + i);  // stream load from shm
            vec_op::FP32Vec16 remote_data_fp32(remote_data);
            local_data_fp32 = local_data_fp32 + remote_data_fp32;  // sum reduce
          });
          vec_t reduced_data(local_data_fp32);
          reduced_data.save(thread_data_ptr + i,
                            data_elem_num - aligned_data_elem_num);
        }

        thread_ctx->barrier(ThreadSHMStat::DONE);
      });

  return;
}
};  // namespace shm_cc_ops

std::unique_ptr<SHMManager> SHMManager::SingletonInstance = nullptr;
std::once_flag SHMManager::SingletonCreationFlag = std::once_flag();

template <typename scalar_t>
void shm_allreduce_sum(ThreadSHMContext* ctx, scalar_t* data, size_t elem_num) {
  switch (ctx->group_size) {
    case 2:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 2>(ctx, data, elem_num);
      break;
    case 4:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 4>(ctx, data, elem_num);
      break;
    case 8:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 8>(ctx, data, elem_num);
      break;
    default:
      TORCH_CHECK(false,
                  "Invalid world size: " + std::to_string(ctx->group_size));
  }
}

template <typename scalar_t>
void shm_gather_impl(ThreadSHMContext* ctx, scalar_t* data, size_t elem_num,
                     scalar_t** outputs, const int dst) {
  CPU_KERNEL_GUARD_IN(shm_gather_impl)
  const int worldsize = ctx->group_size;

  shm_cc_ops::shm_cc_loop<scalar_t>(
      ctx, elem_num,
      [&](ThreadSHMContext* thread_ctx, int64_t data_offset,
          int64_t data_elem_num) {
        int rank = thread_ctx->rank;
        scalar_t* thread_shm_ptr =
            thread_ctx->get_thread_shm_ptr<scalar_t>(rank);

        thread_ctx->barrier(ThreadSHMStat::THREAD_READY);

        shm_cc_ops::memcpy_to_shm(thread_shm_ptr, data + data_offset,
                                  data_elem_num * sizeof(scalar_t));

        thread_ctx->barrier(ThreadSHMStat::SHM_DATA_READY);

        if (rank == dst) {
          shm_cc_ops::memcpy(outputs[rank] + data_offset, data + data_offset,
                             data_elem_num * sizeof(scalar_t));
          for (int i = 1; i < worldsize; ++i) {
            int src_rank = thread_ctx->get_swizzled_rank(i);
            scalar_t* src_ptr =
                thread_ctx->get_thread_shm_ptr<scalar_t>(src_rank);  // shm
            scalar_t* dst_ptr = outputs[src_rank] + data_offset;
            shm_cc_ops::memcpy_from_shm(dst_ptr, src_ptr,
                                        data_elem_num * sizeof(scalar_t));
          }
        }

        thread_ctx->barrier(ThreadSHMStat::DONE);
      });

  return;
}
}  // namespace

void shm_gather(torch::Tensor& data,
                const std::optional<std::vector<torch::Tensor>>& outputs,
                int64_t dst) {
  TORCH_CHECK(data.is_contiguous())
  VLLM_DISPATCH_FLOATING_TYPES(data.scalar_type(), "shm_gather_impl", [&] {
    CPU_KERNEL_GUARD_IN(shm_gather_impl)

    if (outputs.has_value()) {
      TORCH_CHECK_LE(outputs->size(), MAX_SHM_RANK_NUM);
      scalar_t* output_ptrs[MAX_SHM_RANK_NUM] = {nullptr};
      for (int i = 0; i < outputs->size(); ++i) {
        output_ptrs[i] = outputs->at(i).data_ptr<scalar_t>();
      }
      shm_gather_impl(SHMManager::get_singleton_instance()->get_shm_ctx(),
                      data.data_ptr<scalar_t>(), data.numel(), output_ptrs,
                      dst);
    } else {
      shm_gather_impl(SHMManager::get_singleton_instance()->get_shm_ctx(),
                      data.data_ptr<scalar_t>(), data.numel(), (scalar_t**)(0),
                      dst);
    }

    CPU_KERNEL_GUARD_OUT(shm_gather_impl)
  });
}

void shm_allreduce(torch::Tensor& data) {
  TORCH_CHECK(data.is_contiguous())
  VLLM_DISPATCH_FLOATING_TYPES(data.scalar_type(), "shm_allreduce_sum", [&] {
    CPU_KERNEL_GUARD_IN(shm_allreduce_sum)
    shm_allreduce_sum(SHMManager::get_singleton_instance()->get_shm_ctx(),
                      data.data_ptr<scalar_t>(), data.numel());
    CPU_KERNEL_GUARD_OUT(shm_allreduce_sum)
  });
}

void init_shm_manager(const std::string& name, const int64_t group_size,
                      const int64_t rank) {
  SHMManager::create_singleton_instance(name, group_size, rank);
}

std::string join_shm_manager(const std::string& name) {
  TORCH_CHECK(SHMManager::get_singleton_instance());
  SHMManager::get_singleton_instance()->join(name);
  return SHMManager::get_singleton_instance()->get_shm_ctx()->to_string();
}
