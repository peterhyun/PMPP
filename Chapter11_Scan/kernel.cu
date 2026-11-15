// PMPP Chapter 11 — CUDA coding exercises 4, 5, and 7
// ------------------------------------------------------------
// Visual Studio CUDA project (CUDA Toolkit 12.3+, C++17)
//
// This is an EXERCISE SKELETON, not a completed solution.
// Complete the TODOs in the GPU kernels yourself. The host-side
// setup, launches, CPU references, and correctness checks are done.
//
// Run with command-line argument:  7   (default), 4, 5, or all
// Recommended order: Exercise 7 -> Exercise 4 -> Exercise 5
//
// Important distinction:
//   Ex 4, 7: each block scans its OWN segment (prefix resets per block).
//   Ex 5:    the scan spans the ENTIRE input (no reset between blocks).
//
// Expected until TODOs are completed: FAILED tests are normal.

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda/atomic>  // For the acquire/release flag protocol you'll write in Ex 5

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error_ = (call);                                           \
        if (error_ != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(error_)          \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n';       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                      \
    } while (false)

constexpr int E4_THREADS = 128;
constexpr int E4_VALUES_PER_THREAD = 4;
constexpr int E4_TILE = E4_THREADS * E4_VALUES_PER_THREAD;

constexpr int E5_THREADS = 256;
constexpr int E7_THREADS = 256;  // Power of two for the Brent-Kung exercise.

// ============================================================
// EXERCISE 4: Vectorized Kogge-Stone per-block inclusive scan
// Reference: textbook Figure 11.3
// ============================================================
// 4 floats per thread, so each 128-thread block owns 512 floats.
// You decide on an appropriate shared-memory layout and how to
// vectorize FULL tiles while correctly handling the last partial tile.
//
// I chose thread coarsened + vectorized load/store so this is good for less threads firing (really small hardware lol)
__global__ void ex4_vectorizedKoggeStone(const float* input, float* output,
    int N) {
    const int tid = static_cast<int>(threadIdx.x);
    const int tileStart = static_cast<int>(blockIdx.x) * E4_TILE;
    const int threadStart = tileStart + tid * E4_VALUES_PER_THREAD;
    float items[E4_VALUES_PER_THREAD] = { 0.0f };

    // TODO 4.1: Load this thread's four consecutive elements.
    //           Use a vector load on valid full tiles; handle tail elements.
    if (threadStart + E4_VALUES_PER_THREAD - 1 < N)
    {
        // When profiling, must check if this generates a vectorized load in ptx!!!
        memcpy(items, input + threadStart, sizeof(float4));
    }
    else
    {
        for (int i = 0; i < E4_VALUES_PER_THREAD; i++)
        {
            const int idx = threadStart + i;
            if (idx < N)
            {
                items[i] = input[idx];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < E4_VALUES_PER_THREAD - 1; i++)
    {
        items[i + 1] += items[i];
    }

    float threadSum = items[3];
    using BlockScan = cub::BlockScan<float, E4_THREADS>;
    __shared__ BlockScan::TempStorage tempStorage;
    //threadSum = blockScan(threadSum);

    float localInclusivePrefix = 0.0f;
    BlockScan(tempStorage).InclusiveSum(threadSum, localInclusivePrefix);

    __shared__ float threadSums[E4_THREADS];
    //threadSums[tid] = threadSum;
    threadSums[tid] = localInclusivePrefix;
    __syncthreads();

    // Add previous thread's partial sums
    float prevPartialSum = (threadIdx.x > 0) ? threadSums[threadIdx.x - 1] : 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        items[i] += prevPartialSum;
    }

    // Write output
    if (threadStart + E4_VALUES_PER_THREAD - 1 < N)
    {
        // Check if vectorized
        memcpy(output + threadStart, items, sizeof(float4));
    }
    else
    {
        for (int i = 0; i < E4_VALUES_PER_THREAD; i++)
        {
            const int idx = threadStart + i;
            if (idx < N)
            {
                output[idx] = items[i];
            }
        }
    }
}

// ============================================================
// EXERCISE 5: Decoupled look-back for device-wide inclusive scan
// References: textbook Figures 11.17 and 11.18
// ============================================================
// To isolate the actual task in Exercise 5, CUB handles the *local*
// per-block scan. You must implement ONLY the cross-block look-back.
// This differs from Figure 11.18's locally coarsened scan, but has
// the same inter-block data flow for the exercise.
//
// Use two ready states, plus the initial state:
//   EMPTY           : this block has not published useful data.
//   AGGREGATE_READY : its local aggregate is published.
//   PREFIX_READY    : its inclusive block prefix is published.
//
// flags[] is initialized to EMPTY by the host. Each new launch must
// reset the flags and the counter before the scan kernel starts.
enum BlockStatus : unsigned int {
    EMPTY = 0u,
    AGGREGATE_READY = 1u,
    PREFIX_READY = 2u
};

// Return the EXCLUSIVE prefix of all blocks before 'bid'.
// 'blockAggregate' is the sum of THIS block's elements.
__device__ float ex5_interBlockScanDecoupled(
    float blockAggregate,
    unsigned int bid,
    float* blockAggregates,
    float* blockInclusivePrefixes,
    unsigned int* flags) {
    // shared mem as all the threads in the block need to see this value, despite one leader thread of block is doing the update
    float prevBlockInclusivePrefix;

    // We can public blockAggregate info immediately
    blockAggregates[bid] = blockAggregate;
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> myBlockFlag_ref(flags[bid]);
    myBlockFlag_ref.store(BlockStatus::AGGREGATE_READY, cuda::memory_order_release);

    if (bid > 0)
    {
        unsigned int prevBlockFlag = BlockStatus::EMPTY;
        {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> prevBlockFlag_ref(flags[bid - 1]);
            do
            {
                prevBlockFlag = prevBlockFlag_ref.load(cuda::memory_order_acquire);
            } while (prevBlockFlag == BlockStatus::EMPTY);
        }
        if (prevBlockFlag == BlockStatus::PREFIX_READY)
        {
            prevBlockInclusivePrefix = blockInclusivePrefixes[bid - 1];
        }
        else if (prevBlockFlag == BlockStatus::AGGREGATE_READY)
        {
            prevBlockInclusivePrefix = blockAggregates[bid - 1];
            // Keep going back multiple blocks until it finds a block that has prefix_ready, so you can add them
            for (int bidIter = static_cast<int>(bid) - 2; bidIter >= 0; bidIter--)
            {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> prevBlockFlag_ref(flags[bidIter]);
                do
                {
                    prevBlockFlag = prevBlockFlag_ref.load(cuda::memory_order_acquire);
                }
                while (prevBlockFlag == BlockStatus::EMPTY);
                if (prevBlockFlag == BlockStatus::PREFIX_READY)
                {
                    prevBlockInclusivePrefix += blockInclusivePrefixes[bidIter];
                    break;
                }
                else if (prevBlockFlag == BlockStatus::AGGREGATE_READY)
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

    blockInclusivePrefixes[bid] = prevBlockInclusivePrefix + blockAggregate;
    myBlockFlag_ref.store(BlockStatus::PREFIX_READY, cuda::memory_order_release);

    return prevBlockInclusivePrefix;
}

__global__ void ex5_deviceWideScan(const float* input, float* output,
    int N,
    unsigned int* blockCounter,
    float* blockAggregates,
    float* blockInclusivePrefixes,
    unsigned int* flags) {
    using BlockScan = cub::BlockScan<float, E5_THREADS>;
    __shared__ BlockScan::TempStorage tempStorage;
    __shared__ unsigned int bid_s;
    __shared__ float exclusiveBlockPrefix_s;

    const unsigned int tid = threadIdx.x;

    // Assign logical block IDs in the order blocks arrive at the counter.
    // (Unlike blockIdx.x, this helps prevent waiting on unscheduled blocks.)
    if (tid == 0) {
        bid_s = atomicAdd(blockCounter, 1u);
    }
    __syncthreads();

    const unsigned int bid = bid_s;
    const int i = static_cast<int>(bid) * E5_THREADS + static_cast<int>(tid);
    const float value = (i < N) ? input[i] : 0.0f;

    float localInclusivePrefix = 0.0f;
    float blockAggregate = 0.0f;
    BlockScan(tempStorage).InclusiveSum(value, localInclusivePrefix,
        blockAggregate);

    // All local inclusive prefixes are ready. One leader calculates
    // the prefix contributed by preceding logical blocks.
    if (tid == 0) {
        exclusiveBlockPrefix_s = ex5_interBlockScanDecoupled(
            blockAggregate, bid, blockAggregates,
            blockInclusivePrefixes, flags);
    }
    __syncthreads();

    if (i < N) {
        output[i] = localInclusivePrefix + exclusiveBlockPrefix_s;
    }
}

// ============================================================
// EXERCISE 7: Brent-Kung per-block inclusive scan
// Reference: textbook Figure 11.19
// ============================================================
// One element per thread. Incomplete final blocks load zeros.
// Implement the tree stages without changing the provided CPU/host code.
__global__ void ex7_brentKung(const float* input, float* output, int N) {
    __shared__ float sharedData[E7_THREADS];

    const int tid = static_cast<int>(threadIdx.x);
    const int i = static_cast<int>(blockIdx.x) * E7_THREADS + tid;

    sharedData[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // TODO 7.1: Implement the Brent-Kung reduction (up-sweep) tree.
    //           Decide which thread participates at each stride.
    //           Add the necessary block-wide synchronization.
    int stride = 1;
    for (; stride < blockDim.x ; stride *= 2)
    {
        if ((tid + 1) % (stride * 2) == 0)
        {
            sharedData[tid] += sharedData[tid - stride];
        }

        __syncthreads(); // This has to run every iteration!!!
    }

    // stride == blockDim.x now, which is why we do / 4 later.
    // 
    // TODO 7.2: Implement the reverse (down-sweep) tree for an
    //           INCLUSIVE scan (not an exclusive scan).
    //           Add the necessary block-wide synchronization.

    // For the equation source = (k + 1) * 2s - 1 where s is stride, k = 0, 1, 2, ... ,
    // tid can be used as 1. the source idx, or 2. the k itself
    // Code below is used as source idx but commented the k itself approach too
    for (stride /= 4; stride >= 1 ; stride /= 2)
    {
        // Approach 1
        if (((tid + 1) % (2 * stride) == 0) && (tid + stride) < E7_THREADS)
        {
            sharedData[tid + stride] += sharedData[tid];
        }

        /* // Approach 2
        int source = (tid + 1) * 2 * stride - 1;
        if (source + stride < E7_THREADS)
        {
            sharedData[source + stride] += sharedData[source];
        }
        */
        __syncthreads();
    }

    if (i < N) {
        output[i] = sharedData[tid];
    }
}

// ============================================================
// CPU REFERENCES AND TEST HARNESS — DO NOT NEED TO EDIT
// ============================================================
enum class Exercise { Ex4, Ex5, Ex7 };

int tileSize(Exercise ex) {
    if (ex == Exercise::Ex4) return E4_TILE;
    if (ex == Exercise::Ex7) return E7_THREADS;
    return 0; // Ex 5 is a whole-array scan.
}

const char* exerciseName(Exercise ex) {
    switch (ex) {
    case Exercise::Ex4: return "Exercise 4 (vectorized Kogge-Stone)";
    case Exercise::Ex5: return "Exercise 5 (decoupled look-back)";
    case Exercise::Ex7: return "Exercise 7 (Brent-Kung)";
    }
    return "Unknown exercise";
}

// Host inclusive scan; resets after each tile for Ex 4/7.
// Uses integer-valued floats so these correctness tests are not
// obscured by floating-point re-association/round-off issues.
std::vector<float> cpuReference(const std::vector<float>& input,
    Exercise ex) {
    std::vector<float> expected(input.size(), 0.0f);
    const int tile = tileSize(ex);
    float running = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        if (tile != 0 && (i % static_cast<size_t>(tile) == 0)) {
            running = 0.0f;
        }
        running += input[i];
        expected[i] = running;
    }
    return expected;
}

std::vector<float> makeInput(int N) {
    if (N == 8) {
        return { 4.0f, 6.0f, 7.0f, 1.0f,
                2.0f, 8.0f, 5.0f, 2.0f };
    }
    // Fixed seed -> reproducible test cases. Includes negatives and zeros.
    std::mt19937 rng(11000u + static_cast<unsigned int>(N));
    std::uniform_int_distribution<int> dist(-5, 5);
    std::vector<float> x(static_cast<size_t>(N));
    for (float& v : x) v = static_cast<float>(dist(rng));
    return x;
}

bool compareAndPrint(const std::vector<float>& got,
    const std::vector<float>& expected,
    Exercise ex, int N) {
    int failures = 0;
    for (int i = 0; i < N; ++i) {
        const float actual = got[static_cast<size_t>(i)];
        const float wanted = expected[static_cast<size_t>(i)];
        if (!std::isfinite(actual) || std::fabs(actual - wanted) > 0.001f) {
            if (failures < 4) {
                std::cout << "    mismatch at i=" << i
                    << ": GPU=" << actual << " CPU=" << wanted << '\n';
            }
            ++failures;
        }
    }
    std::cout << "  " << (failures == 0 ? "PASS" : "FAIL")
        << "  N=" << std::setw(5) << N;
    if (failures) std::cout << "  (" << failures << " mismatches)";
    std::cout << '\n';
    return failures == 0;
}

bool runOne(Exercise ex, int N) {
    const std::vector<float> input = makeInput(N);
    const std::vector<float> expected = cpuReference(input, ex);
    std::vector<float> result(static_cast<size_t>(N));
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    float* dInput = nullptr;
    float* dOutput = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dInput), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dOutput), bytes));
    CUDA_CHECK(cudaMemcpy(dInput, input.data(), bytes, cudaMemcpyHostToDevice));
    // Recognizable unwritten output. No false PASS from stale GPU memory.
    CUDA_CHECK(cudaMemset(dOutput, 0xFF, bytes));

    if (ex == Exercise::Ex4) {
        const int blocks = (N + E4_TILE - 1) / E4_TILE;
        ex4_vectorizedKoggeStone << <blocks, E4_THREADS >> > (dInput, dOutput, N);
    }
    else if (ex == Exercise::Ex7) {
        const int blocks = (N + E7_THREADS - 1) / E7_THREADS;
        ex7_brentKung << <blocks, E7_THREADS >> > (dInput, dOutput, N);
    }
    else {
        const int blocks = (N + E5_THREADS - 1) / E5_THREADS;
        const size_t blockBytes = static_cast<size_t>(blocks) * sizeof(float);
        const size_t flagBytes = static_cast<size_t>(blocks) * sizeof(unsigned int);

        unsigned int* dBlockCounter = nullptr;
        float* dBlockAggregates = nullptr;
        float* dBlockInclusivePrefixes = nullptr;
        unsigned int* dFlags = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dBlockCounter),
            sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dBlockAggregates),
            blockBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dBlockInclusivePrefixes),
            blockBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dFlags), flagBytes));
        CUDA_CHECK(cudaMemset(dBlockCounter, 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(dBlockAggregates, 0, blockBytes));
        CUDA_CHECK(cudaMemset(dBlockInclusivePrefixes, 0, blockBytes));
        CUDA_CHECK(cudaMemset(dFlags, 0, flagBytes));

        ex5_deviceWideScan << <blocks, E5_THREADS >> > (
            dInput, dOutput, N, dBlockCounter,
            dBlockAggregates, dBlockInclusivePrefixes, dFlags);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(dFlags));
        CUDA_CHECK(cudaFree(dBlockInclusivePrefixes));
        CUDA_CHECK(cudaFree(dBlockAggregates));
        CUDA_CHECK(cudaFree(dBlockCounter));
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), dOutput, bytes,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));

    return compareAndPrint(result, expected, ex, N);
}

bool runExercise(Exercise ex) {
    std::cout << "\n" << exerciseName(ex) << '\n';

    // Checks the book's sample, warp boundaries, tile boundaries,
    // incomplete last tiles, and multiple blocks.
    const std::vector<int> sizes = {
        1, 8, 31, 32, 33, 127, 128, 129,
        255, 256, 257, 511, 512, 513, 1024, 1031, 2049
    };

    bool allPassed = true;
    for (int N : sizes) {
        const bool passed = runOne(ex, N);
        allPassed = passed && allPassed;
    }
    std::cout << (allPassed ? "ALL TESTS PASS\n"
        : "SOME TESTS FAIL (expected before completing TODOs)\n");
    return allPassed;
}

int main(int argc, char** argv) {
    const std::string selection = (argc >= 2) ? argv[1] : "7";
    if (selection != "4" && selection != "5" &&
        selection != "7" && selection != "all") {
        std::cerr << "Usage: PMPP_Ch11_Exercises_4_5_7_Skeleton.exe"
            " [4|5|7|all]\n";
        return EXIT_FAILURE;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable GPU found.\n";
        return EXIT_FAILURE;
    }

    bool ok = true;
    if (selection == "4" || selection == "all") {
        ok = runExercise(Exercise::Ex4) && ok;
    }
    if (selection == "5" || selection == "all") {
        ok = runExercise(Exercise::Ex5) && ok;
    }
    if (selection == "7" || selection == "all") {
        ok = runExercise(Exercise::Ex7) && ok;
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
