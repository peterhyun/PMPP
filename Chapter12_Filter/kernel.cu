// PMPP (5th ed.) Chapter 12 — Filter, Exercises 1–4
// ---------------------------------------------------------------
// Visual Studio CUDA project / CUDA Toolkit 12.x / C++17
//
// This is an EXERCISE SKELETON, not a completed implementation.
// Fill in the CUDA TODOs yourself. The CPU reference, CUDA allocations,
// launches, input generation, and PASS/FAIL checks are already provided.
// Tests are EXPECTED TO FAIL until the TODOs are completed.
//
// Run:  PMPP_Ch12_Exercises_1_to_4_Skeleton.exe [1|2|3|4|all]
//       Default selection: 1
// Build (RTX 3070 / sm_86):
//       nvcc -O2 -std=c++17 -arch=sm_86 \
//         PMPP_Ch12_Exercises_1_to_4_Skeleton.cu -o ch12_filter.exe
//
// Filter condition chosen for the tests: KEEP EVEN INTEGERS.
// The filter must be STABLE: retained values preserve their input order.
// Example: [9, 4, 7, 2, 6, 3, 2, 10, 1] -> [4, 2, 6, 2, 10]
//
// Source exercise statements:
//   1. Three conceptual stages: condition, exclusive scan, output scatter.
//      Uses CUB DeviceScan from the host for the scanning stage.
//   2. Single-kernel stable filter using a one-pass scan.
//   3. Privatize Exercise 2 to improve output coalescing.
//   4. Apply thread coarsening to Exercise 3.
//
// Ex 1 launches a condition kernel, calls cub::DeviceScan::ExclusiveSum,
// then launches an output-scatter kernel. CUB may internally launch
// multiple kernels: this is the intended practical three-stage version.
// Ex 2/3/4 host launches ONE kernel per filter call. In these exercises,
// blocks obtain LOGICAL tile indices dynamically from a device counter,
// as in the previous Chapter 11 one-pass scan exercise. Physical blockIdx.x
// need not reflect execution order.
//
// To use in your existing Visual Studio PMPP.sln:
//   Create/choose a Chapter 12 CUDA project and add this file as its .cu.
//   Remove/exclude any other source file that defines main().

#include <cuda_runtime.h>
#include <cub/cub.cuh>     // DeviceScan (Ex 1), BlockScan (Ex 2-4).
#include <cuda/atomic>    // Available for device-scope acquire/release state.

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err_ = (call);                                       \
        if (err_ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__      \
                      << ": " << cudaGetErrorString(err_) << '\n';            \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                      \
    } while (false)

constexpr int E1_THREADS = 256;
constexpr int E234_THREADS = 256;
constexpr int E4_COARSENING_FACTOR = 4;
constexpr int E4_TILE = E234_THREADS * E4_COARSENING_FACTOR;

__host__ __device__ inline bool passesCondition(int value) {
    return (value % 2) == 0;  // Keep even numbers; includes negative evens.
}

// ===============================================================
// EXERCISE 1: Stable filter as THREE conceptual stages
// ===============================================================

// Kernel #1: Evaluate the keep condition, one result per input value.
// Expected: keep[i] is exactly 0 or 1.
__global__ void ex1_evaluateCondition(const int* input, int* keep, int N) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    // TODO 1.1: Guard the input index against N.
    if (i >= N)
    {
        return;
    }

    // TODO 1.2: Write keep[i] = 1 if passesCondition(input[i]), else 0.
    keep[i] = passesCondition(input[i]) ? 1 : 0;
}

// Stage #2 is HOST-SIDE cub::DeviceScan::ExclusiveSum (see runOneTest).
// No scan implementation is required for this exercise.
//
// // Kernel #3: Scatter kept values to their scanned positions, and set
// outputSize = total count of retained values.
__global__ void ex1_writeOutput(const int* input, const int* keep,
    const int* offsets, int* output,
    int* outputSize, int N) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    // TODO 1.4: If i < N and keep[i], write input[i] to output[offsets[i]].
    if (i >= N)
    {
        return;
    }

    // TODO 1.5: Have exactly one thread write the final outputSize.
    //           Use the exclusive scan result and keep[]; no extra kernel.
    if (keep[i])
    {
        output[offsets[i]] = input[i];
    }

    if (i == N - 1)
    {
        *outputSize = offsets[i] + keep[i];
    }
}

// ===============================================================
// SHARED BOILERPLATE FOR EXERCISES 2, 3, AND 4
// ===============================================================
// The host initializes all states to EMPTY before EACH one-pass launch.
// A block first publishes its local keep count, then eventually its
// inclusive prefix. Other blocks may look backward through these states.
// Reuse (or paste) your Chapter 11 single-pass scan / decoupled
// look-back implementation. You do NOT need to reimplement block scan;
// cub::BlockScan handles that. The inter-block prefix calculation is
// still needed if the filter must truly be ONE kernel per invocation.
//
// IMPORTANT: flags must be published/read with the proper device-scope
// ordering (release/acquire, or an equivalent correctly ordered protocol).
// blockIdx.x alone is not a safe global execution order.
enum BlockState : unsigned int {
    EMPTY = 0u,
    AGGREGATE_READY = 1u,
    PREFIX_READY = 2u
};

// Return the exclusive number of kept values in all LOGICAL earlier tiles.
__device__ int unfinishedBlockExclusivePrefix(
    int blockKeepCount, unsigned int logicalTileId,
    int* blockAggregates, int* blockInclusivePrefixes,
    unsigned int* blockStates) {
    // This is the exact same implementation from previous ch

    float prevBlockInclusivePrefix;

    // We can public blockAggregate info immediately
    blockAggregates[logicalTileId] = blockKeepCount;
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> myBlockFlag_ref(blockStates[logicalTileId]);
    myBlockFlag_ref.store(BlockState::AGGREGATE_READY, cuda::memory_order_release);

    if (logicalTileId > 0)
    {
        unsigned int prevBlockFlag = BlockState::EMPTY;
        {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> prevBlockFlag_ref(blockStates[logicalTileId - 1]);
            do
            {
                prevBlockFlag = prevBlockFlag_ref.load(cuda::memory_order_acquire);
            } while (prevBlockFlag == BlockState::EMPTY);
        }
        if (prevBlockFlag == BlockState::PREFIX_READY)
        {
            prevBlockInclusivePrefix = blockInclusivePrefixes[logicalTileId - 1];
        }
        else if (prevBlockFlag == BlockState::AGGREGATE_READY)
        {
            prevBlockInclusivePrefix = blockAggregates[logicalTileId - 1];
            // Keep going back multiple blocks until it finds a block that has prefix_ready, so you can add them
            for (int bidIter = static_cast<int>(logicalTileId) - 2; bidIter >= 0; bidIter--)
            {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> prevBlockFlag_ref(blockStates[bidIter]);
                do
                {
                    prevBlockFlag = prevBlockFlag_ref.load(cuda::memory_order_acquire);
                } while (prevBlockFlag == BlockState::EMPTY);
                if (prevBlockFlag == BlockState::PREFIX_READY)
                {
                    prevBlockInclusivePrefix += blockInclusivePrefixes[bidIter];
                    break;
                }
                else if (prevBlockFlag == BlockState::AGGREGATE_READY)
                {
                    prevBlockInclusivePrefix += blockAggregates[bidIter];
                }
            }
        }
    }
    else
    {
        prevBlockInclusivePrefix = 0.0f;
    }

    blockInclusivePrefixes[logicalTileId] = prevBlockInclusivePrefix + blockKeepCount;
    myBlockFlag_ref.store(BlockState::PREFIX_READY, cuda::memory_order_release);

    return prevBlockInclusivePrefix;
}

// ===============================================================
// EXERCISE 2: ONE kernel, stable filter fused into one-pass scan
// ===============================================================
__global__ void ex2_onePassStableFilter(
    const int* input, int* output, int* outputSize, int N,
    unsigned int* tileCounter, int* blockAggregates,
    int* blockInclusivePrefixes, unsigned int* blockStates) {

    using BlockScan = cub::BlockScan<int, E234_THREADS>;
    __shared__ BlockScan::TempStorage scanStorage;
    __shared__ unsigned int logicalTileShared;
    __shared__ int blockOffsetShared;

    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
        logicalTileShared = atomicAdd(tileCounter, 1u);
        blockOffsetShared = 0;
    }
    __syncthreads();

    const unsigned int tile = logicalTileShared;
    const int i = static_cast<int>(tile) * E234_THREADS + tid;

    // TODO 2.1: Read input[i] if valid and evaluate passesCondition().
    if (i >= N)
    {
        return;
    }
    int keep = passesCondition(input[i]) ? 1 : 0;

    // TODO 2.2: Use cub::BlockScan::ExclusiveSum to calculate the
    //           block-local exclusive positions and tile keep count.
    //           You do NOT need to implement a local scan from scratch.
    int localExclusivePrefix = 0;
    int blockAggregate = 0;
    BlockScan(scanStorage).ExclusiveSum(keep, localExclusivePrefix, blockAggregate);

    if (tid == 0) {
        blockOffsetShared = unfinishedBlockExclusivePrefix(
            blockAggregate, tile, blockAggregates,
            blockInclusivePrefixes, blockStates);
    }
    __syncthreads();

    // TODO 2.3: Scatter each retained value to the global output array,
    //           using BOTH its local offset and the preceding-tiles offset.
    if (keep)
    {
        output[blockOffsetShared + localExclusivePrefix] = input[i];
    }
     
    // TODO 2.4: Write outputSize exactly once for the final logical tile.
    if (i == N-1)
    {
        *outputSize = blockOffsetShared + localExclusivePrefix + keep;
    }
}

// ===============================================================
// EXERCISE 3: Add PRIVATIZATION to Exercise 2
// ===============================================================
// Each tile first compacts retained values into shared memory, preserving
// their order, then consecutive threads flush the compacted segment to
// global output. The goal is better coalescing of the OUTPUT writes.
__global__ void ex3_privatizedStableFilter(
    const int* input, int* output, int* outputSize, int N,
    unsigned int* tileCounter, int* blockAggregates,
    int* blockInclusivePrefixes, unsigned int* blockStates) {

    using BlockScan = cub::BlockScan<int, E234_THREADS>;
    __shared__ BlockScan::TempStorage scanStorage;
    __shared__ unsigned int logicalTileShared;
    __shared__ int blockOffsetShared;

    // shared mem for privatization
    __shared__ int s_blockOutput[E234_THREADS];

    const int tid = static_cast<int>(threadIdx.x);
    s_blockOutput[tid] = 0;
    if (tid == 0) {
        logicalTileShared = atomicAdd(tileCounter, 1u);
        blockOffsetShared = 0;
    }
    __syncthreads();

    const unsigned int tile = logicalTileShared;
    const int i = static_cast<int>(tile) * E234_THREADS + tid;

    // TODO 2.1: Read input[i] if valid and evaluate passesCondition().
    int value = i < N ? input[i] : 0;
    int keep = (i < N && passesCondition(value)) ? 1 : 0;

    // TODO 2.2: Use cub::BlockScan::ExclusiveSum to calculate the
    //           block-local exclusive positions and tile keep count.
    //           You do NOT need to implement a local scan from scratch.
    int localExclusivePrefix = 0;
    int blockAggregate = 0;
    BlockScan(scanStorage).ExclusiveSum(keep, localExclusivePrefix, blockAggregate);
    if (tid == 0) {
        blockOffsetShared = unfinishedBlockExclusivePrefix(
            blockAggregate, tile, blockAggregates,
            blockInclusivePrefixes, blockStates);
    }

    if (keep)
    {
        s_blockOutput[localExclusivePrefix] = value;
    }
    __syncthreads();

    // TODO 2.3: Scatter each retained value to the global output array,
    //           using BOTH its local offset and the preceding-tiles offset.
    if (tid < blockAggregate)
    {
        output[blockOffsetShared + tid] = s_blockOutput[tid];
    }

    // TODO 2.4: Write outputSize exactly once for the final logical tile.
    if (i == N - 1)
    {
        *outputSize = blockOffsetShared + localExclusivePrefix + keep;
    }
}

// ===============================================================
// EXERCISE 4: Add THREAD COARSENING to Exercise 3
// ===============================================================
// Each thread now owns E4_COARSENING_FACTOR input values; one logical tile
// covers E4_TILE values. Stable order must still be preserved even when a
// thread's multiple input values produce different keep counts.
__global__ void ex4_coarsenedPrivatizedStableFilter(
    const int* input, int* output, int* outputSize, int N,
    unsigned int* tileCounter, int* blockAggregates,
    int* blockInclusivePrefixes, unsigned int* blockStates) {

    using BlockScan = cub::BlockScan<int, E234_THREADS>;
    __shared__ BlockScan::TempStorage scanStorage;
    __shared__ int privateOutput[E4_TILE];
    __shared__ unsigned int logicalTileShared;
    __shared__ int blockOffsetShared;

    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
        logicalTileShared = atomicAdd(tileCounter, 1u);
        /*
        blockOffsetShared = 0;
        */
    }
    __syncthreads();

    const unsigned int tile = logicalTileShared;
    const int tileStart = static_cast<int>(tile) * E4_TILE;
    int values[E4_COARSENING_FACTOR] = {};
    int keeps[E4_COARSENING_FACTOR] = {};

    // TODO 4.1: Load up to E4_COARSENING_FACTOR values per thread.
    //           You choose the per-thread index mapping; maintain
    //           each input element's ORIGINAL relative order.

    // TODO: This is not a coalesced load, so need to profile if coalesced load + syncthreads() would be faster than this
    int threadTotalKeepsCount = 0;
    for (int i = 0; i < E4_COARSENING_FACTOR; i++)
    {
        int globalIdx = tileStart + tid * E4_COARSENING_FACTOR + i;
        values[i] = (globalIdx < N) ? input[globalIdx] : 0;
        keeps[i] = (globalIdx < N) ? passesCondition(values[i]) : 0;
        if (keeps[i])
        {
            threadTotalKeepsCount++;
        }
    }

    // TODO 4.2: Evaluate keep flags and derive all LOCAL output offsets.
    //           Include the per-thread kept count in the block-wide scan.
    int threadTotalKeepsCountExclusivePrefix = 0;
    int blockAggregate = 0;
    BlockScan(scanStorage).ExclusiveSum(threadTotalKeepsCount, threadTotalKeepsCountExclusivePrefix, blockAggregate);

    // TODO 4.3: Write retained values into privateOutput[] in stable order,
    //           and set blockKeepCountShared.
    int j = 0;
    for (int i = 0; i < E4_COARSENING_FACTOR; i++)
    {
        if (keeps[i])
        {
            privateOutput[threadTotalKeepsCountExclusivePrefix + j] = values[i];
            j++;
        }
    }

    if (tid == 0) {
        blockOffsetShared = unfinishedBlockExclusivePrefix(
            blockAggregate, tile, blockAggregates,
            blockInclusivePrefixes, blockStates);
    }
    __syncthreads();

    // TODO 4.4: Flush consecutive privateOutput[] elements to global output.
    //           Each thread may need to flush multiple packed elements.
    for (int i = 0; i < E4_COARSENING_FACTOR; i++)
    {
        int privateOutputOffset = tid + i * E234_THREADS;
        if (privateOutputOffset < blockAggregate)
        {
            output[blockOffsetShared + privateOutputOffset] = privateOutput[privateOutputOffset];
        }
    }

    // TODO 4.5: Write outputSize from the final logical tile.
    if (tid == 0 && (tile == ((N - 1) / E4_TILE)))
    {
        *outputSize = blockOffsetShared + blockAggregate;
    }
}

// ===============================================================
// HOST-SIDE REFERENCE + TEST HARNESS: already implemented
// ===============================================================
enum class Pattern {
    Mixed,
    AllKeep,
    AllDrop,
    Alternating,
    OnlyFirst,
    OnlyLast,
    MostlyKeep
};

const char* patternName(Pattern p) {
    switch (p) {
    case Pattern::Mixed:       return "mixed";
    case Pattern::AllKeep:     return "all-keep";
    case Pattern::AllDrop:     return "all-drop";
    case Pattern::Alternating: return "alternating";
    case Pattern::OnlyFirst:   return "first-only";
    case Pattern::OnlyLast:    return "last-only";
    case Pattern::MostlyKeep:  return "mostly-keep";
    }
    return "unknown";
}

std::vector<int> makeInput(int N, Pattern p) {
    std::vector<int> result(static_cast<size_t>(N));
    std::mt19937 rng(12000u + static_cast<unsigned int>(N));
    std::uniform_int_distribution<int> dist(-30, 30);
    for (int i = 0; i < N; ++i) {
        switch (p) {
        case Pattern::Mixed:
            result[i] = dist(rng); break;
        case Pattern::AllKeep:
            result[i] = 2 * (i - N / 2); break;
        case Pattern::AllDrop:
            result[i] = 2 * i + 1; break;
        case Pattern::Alternating:
            result[i] = (i % 2 == 0) ? 2 * i : 2 * i + 1; break;
        case Pattern::OnlyFirst:
            result[i] = (i == 0) ? 6 : 2 * i + 1; break;
        case Pattern::OnlyLast:
            result[i] = (i == N - 1) ? -8 : 2 * i + 1; break;
        case Pattern::MostlyKeep:
            result[i] = (i % 7 == 0) ? 2 * i + 1 : 2 * i; break;
        }
    }
    return result;
}

std::vector<int> cpuStableFilter(const std::vector<int>& input) {
    std::vector<int> kept;
    kept.reserve(input.size());
    for (int v : input) {
        if (passesCondition(v)) kept.push_back(v);
    }
    return kept;
}

struct TestCase {
    int N;
    Pattern pattern;
};

const char* exName(int ex) {
    switch (ex) {
    case 1: return "Ex 1: three-stage filter (CUB DeviceScan)";
    case 2: return "Ex 2: 1-pass fused kernel";
    case 3: return "Ex 3: privatized filter";
    case 4: return "Ex 4: privatized + coarsened filter";
    }
    return "unknown";
}

bool runOneTest(int ex, TestCase test) {
    const std::vector<int> input = makeInput(test.N, test.pattern);
    const std::vector<int> expected = cpuStableFilter(input);

    // No kernel needed for the empty-array case.
    if (test.N == 0) {
        std::cout << "  PASS  N=   0  pattern=" << patternName(test.pattern)
            << "  kept=0\n";
        return true;
    }

    const size_t bytes = static_cast<size_t>(test.N) * sizeof(int);
    int* dInput = nullptr;
    int* dOutput = nullptr;
    int* dOutputSize = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dInput), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dOutput), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dOutputSize), sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dInput, input.data(), bytes, cudaMemcpyHostToDevice));
    // Recognizable invalid values so an unfinished kernel doesn't pass
    // simply because freshly allocated memory happens to contain zeros.
    CUDA_CHECK(cudaMemset(dOutput, 0xA5, bytes));
    CUDA_CHECK(cudaMemset(dOutputSize, 0xFF, sizeof(int))); // -1

    if (ex == 1) {
        int* dKeep = nullptr;
        int* dOffsets = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dKeep), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dOffsets), bytes));
        CUDA_CHECK(cudaMemset(dKeep, 0xFF, bytes));
        CUDA_CHECK(cudaMemset(dOffsets, 0xFF, bytes));

        const int blocks = (test.N + E1_THREADS - 1) / E1_THREADS;

        // Stage 1: GPU condition evaluation. TODO remains inside kernel.
        ex1_evaluateCondition << <blocks, E1_THREADS >> > (dInput, dKeep, test.N);
        CUDA_CHECK(cudaGetLastError());

        // Stage 2: CUB device-wide EXCLUSIVE scan. Fully provided.
        // CUB owns any internal kernel launches needed to implement it.
        void* dScanTemp = nullptr;
        size_t scanTempBytes = 0;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            dScanTemp, scanTempBytes, dKeep, dOffsets, test.N));
        CUDA_CHECK(cudaMalloc(&dScanTemp, scanTempBytes));
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            dScanTemp, scanTempBytes, dKeep, dOffsets, test.N));

        // Stage 3: GPU scatter / output count. TODO remains inside kernel.
        ex1_writeOutput << <blocks, E1_THREADS >> > (
            dInput, dKeep, dOffsets, dOutput, dOutputSize, test.N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(dScanTemp));
        CUDA_CHECK(cudaFree(dKeep));
        CUDA_CHECK(cudaFree(dOffsets));
    }
    else {
        const int tileSize = (ex == 4) ? E4_TILE : E234_THREADS;
        const int blocks = (test.N + tileSize - 1) / tileSize;
        const size_t tileBytes = static_cast<size_t>(blocks) * sizeof(int);
        const size_t stateBytes = static_cast<size_t>(blocks) * sizeof(unsigned int);

        unsigned int* dTileCounter = nullptr;
        int* dAggregates = nullptr;
        int* dInclusivePrefixes = nullptr;
        unsigned int* dStates = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTileCounter),
            sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dAggregates), tileBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dInclusivePrefixes), tileBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dStates), stateBytes));
        CUDA_CHECK(cudaMemset(dTileCounter, 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(dAggregates, 0, tileBytes));
        CUDA_CHECK(cudaMemset(dInclusivePrefixes, 0, tileBytes));
        CUDA_CHECK(cudaMemset(dStates, 0, stateBytes));

        if (ex == 2) {
            ex2_onePassStableFilter << <blocks, E234_THREADS >> > (
                dInput, dOutput, dOutputSize, test.N, dTileCounter,
                dAggregates, dInclusivePrefixes, dStates);
        }
        else if (ex == 3) {
            ex3_privatizedStableFilter << <blocks, E234_THREADS >> > (
                dInput, dOutput, dOutputSize, test.N, dTileCounter,
                dAggregates, dInclusivePrefixes, dStates);
        }
        else {
            ex4_coarsenedPrivatizedStableFilter << <blocks, E234_THREADS >> > (
                dInput, dOutput, dOutputSize, test.N, dTileCounter,
                dAggregates, dInclusivePrefixes, dStates);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(dTileCounter));
        CUDA_CHECK(cudaFree(dAggregates));
        CUDA_CHECK(cudaFree(dInclusivePrefixes));
        CUDA_CHECK(cudaFree(dStates));
    }

    int actualSize = -1;
    std::vector<int> actual(static_cast<size_t>(test.N));
    CUDA_CHECK(cudaMemcpy(&actualSize, dOutputSize, sizeof(int),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual.data(), dOutput, bytes,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));
    CUDA_CHECK(cudaFree(dOutputSize));

    int errors = 0;
    if (actualSize != static_cast<int>(expected.size())) {
        std::cout << "    size mismatch: GPU=" << actualSize
            << " CPU=" << expected.size() << '\n';
        ++errors;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            if (errors < 5) {
                std::cout << "    output[" << i << "]: GPU=" << actual[i]
                    << " CPU=" << expected[i] << '\n';
            }
            ++errors;
        }
    }

    std::cout << "  " << (errors ? "FAIL" : "PASS")
        << "  N=" << std::setw(5) << test.N
        << "  pattern=" << std::setw(11) << std::left
        << patternName(test.pattern) << std::right
        << "  kept=" << expected.size();
    if (errors) std::cout << "  (" << errors << " mismatches)";
    std::cout << '\n';
    return errors == 0;
}

bool runExercise(int ex) {
    std::cout << "\n" << exName(ex) << '\n';

    // Covers empty inputs, warp/block boundaries, partial tiles, many
    // logical tiles, all-keep, all-drop, mostly-keep, duplicates, and
    // negative values. Exact stable order is checked, not just counts.
    const std::vector<TestCase> tests = {
        {0,    Pattern::Mixed},
        {1,    Pattern::AllKeep},
        {1,    Pattern::AllDrop},
        {8,    Pattern::Mixed},
        {31,   Pattern::Alternating},
        {32,   Pattern::Mixed},
        {33,   Pattern::AllKeep},
        {127,  Pattern::AllDrop},
        {128,  Pattern::Mixed},
        {129,  Pattern::OnlyLast},
        {255,  Pattern::Mixed},
        {256,  Pattern::MostlyKeep},
        {257,  Pattern::OnlyFirst},
        {511,  Pattern::Mixed},
        {512,  Pattern::Alternating},
        {513,  Pattern::Mixed},
        {1023, Pattern::OnlyLast},
        {1024, Pattern::AllKeep},
        {1025, Pattern::Mixed},
        {4097, Pattern::Mixed},
        {8193, Pattern::MostlyKeep}
    };

    bool allPassed = true;
    for (const TestCase& test : tests) {
        allPassed = runOneTest(ex, test) && allPassed;
    }
    std::cout << (allPassed ? "ALL TESTS PASS\n"
        : "SOME TESTS FAIL (normal before finishing TODOs)\n");
    return allPassed;
}

int main(int argc, char** argv) {
    const std::string selection = (argc > 1) ? argv[1] : "1";
    if (selection != "1" && selection != "2" && selection != "3" &&
        selection != "4" && selection != "all") {
        std::cerr << "Usage: ch12_filter.exe [1|2|3|4|all]\n";
        return EXIT_FAILURE;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable GPU found.\n";
        return EXIT_FAILURE;
    }

    bool passed = true;
    if (selection == "1" || selection == "all") {
        passed = runExercise(1) && passed;
    }
    if (selection == "2" || selection == "all") {
        passed = runExercise(2) && passed;
    }
    if (selection == "3" || selection == "all") {
        passed = runExercise(3) && passed;
    }
    if (selection == "4" || selection == "all") {
        passed = runExercise(4) && passed;
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
