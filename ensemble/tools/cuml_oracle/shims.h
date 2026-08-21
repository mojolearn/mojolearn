// Host-only shims so cuML's OWN decision-tree headers compile and run here.
//
// Nothing in this file implements any part of their algorithm. It supplies
// the CUDA-shaped names their headers reference -- HDI/DI, raft::log,
// raft::laneId/WarpSize/shfl, atomicAdd/CAS/Exch, __syncthreads,
// __threadfence, threadIdx/blockDim -- so that `bins.cuh`, `split.cuh`,
// `dataset.h` and `objectives.cuh` can be included VERBATIM and called.
//
// The point is that the numbers this oracle prints are computed by THEIR
// code, not by a transcription of it. Their gain functions, their bin
// arithmetic, their `Split::update` total order and their `lower_bound` are
// all `HDI`/`DI` -- host-device inline -- so a plain c++ compiler runs them
// unchanged. Only the `__global__` kernels need a GPU, and none of the
// arithmetic lives there.
//
// blockDim.x is 1 and threadIdx.x is 0, so `Gain`'s grid-stride loop
// (`objectives.cuh:167`, `:368`) walks EVERY bin in one call, serially. That
// is the same set of candidates a full block visits, reduced in index order
// instead of by warp -- and since the candidates are what is being compared,
// not the reduction schedule, that is the right shape for an oracle.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <array>
#include <algorithm>

#define HDI inline
#define DI inline

namespace raft {
template <typename T> HDI T log(T x) { return std::log(x); }
constexpr int WarpSize = 32;
HDI int laneId() { return 0; }
template <typename T> HDI T shfl(T v, int) { return v; }
template <typename T> HDI T alignTo(T x, T y) { return ((x + y - 1) / y) * y; }
}  // namespace raft

namespace cuda { namespace std {
template <typename T, ::std::size_t N> using array = ::std::array<T, N>;
} }

struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 threadIdx;
static Dim3 blockDim = {1, 1, 1};
static Dim3 blockIdx;
static Dim3 gridDim = {1, 1, 1};

template <typename T> HDI T atomicAdd(T* a, T v) { T o = *a; *a += v; return o; }
HDI int atomicCAS(int* a, int c, int v) { int o = *a; if (o == c) *a = v; return o; }
HDI int atomicExch(int* a, int v) { int o = *a; *a = v; return o; }
HDI void __syncthreads() {}
HDI void __threadfence() {}

enum CRITERION {
  GINI, ENTROPY, MSE, MAE, POISSON, GAMMA, INVERSE_GAUSSIAN, CRITERION_END
};
