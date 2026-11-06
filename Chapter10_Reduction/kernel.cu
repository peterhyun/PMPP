// PMPP Chapter 10 — Exercises 3, 4, 5
// CUDA 12.x / C++17 — intentionally incomplete WORKING SKELETON.
// Fill in TODOs in the kernels. The provided host tests should FAIL until then.
//
// Build (RTX 3070 / sm_86):
//   nvcc -O2 -std=c++17 -arch=sm_86 PMPP_Ch10_Exercises_3_4_5.cu -o reduction_exercises.exe
// Run (PowerShell):
//   .\reduction_exercises.exe

#include <cuda_runtime.h>
#include <math_constants.h>  // CUDART_INF_F

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                               \
        if (err_ != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__        \
                      << ": " << cudaGetErrorString(err_) << '\n';              \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

constexpr int BLOCK_THREADS = 128;  // threads per block, NOT warps per block
constexpr int COARSE_FACTOR = 4;
constexpr int SEGMENT_ELEMENTS = 2 * COARSE_FACTOR * BLOCK_THREADS;

// -----------------------------------------------------------------------------
// Exercise 3: Mirror Figure 10.8's reduction so the active values stay at the
// RIGHT end of the input tile. Result ends in input[2 * blockDim.x - 1].
//
// Like Fig. 10.8, this kernel modifies its input IN PLACE (global memory).
// One block, 2*blockDim.x elements. The book's illustration uses 8 threads
// and 16 elements, but test the pattern with larger blocks too.
// -----------------------------------------------------------------------------
__global__ void exercise3_rightAlignedSum(float* input, float* output) {
    unsigned int tid = threadIdx.x;

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (tid < stride) {
            input[2 * blockDim.x - 1 - tid] += input[2 * blockDim.x - 1 - tid - stride];
        }
        // All threads must reach this barrier, even if they skipped the addition.
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = input[2 * blockDim.x - 1];
    }
}

// -----------------------------------------------------------------------------
// Exercise 4: Figure 10.20 with MAX instead of SUM.
//
// Input length must be an exact multiple of SEGMENT_ELEMENTS for this exercise.
// Each block produces ONE partial maximum in output[blockIdx.x].
// The host test combines block results to check the final maximum.
// -----------------------------------------------------------------------------
__global__ void exercise4_coarsenedMax(const float* input, float* output) {
    __shared__ float partial[BLOCK_THREADS];

    unsigned int tid = threadIdx.x;
    unsigned int segment = (2 * COARSE_FACTOR * blockDim.x) * blockIdx.x;
    unsigned int i = segment + tid;

    // Important: negative-only inputs must still work.
    float partialMax = input[i];

    for (unsigned int c = 1; c < 2 * COARSE_FACTOR; ++c) {
        partialMax = fmaxf(partialMax, input[i + c * blockDim.x]);
    }

    partial[tid] = partialMax;
 
    // Now we have a block sized shared memory to reduce from
    // We can choose different algorithms like TwoStageWarpLevelSumReduction (split it to warps and do reductions warp level) or 
    // Fig 10.10 (the simplest version of a block reduction, no warp operations)
    // Reduce the per-thread maxima to one maximum per block.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride)
        {
            partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        }
    }

    if (tid == 0) {
        output[blockIdx.x] = partial[0];
    }
}

// -----------------------------------------------------------------------------
// Exercise 5: Figure 10.20 with ARBITRARY input length N.
//
// Each block still covers SEGMENT_ELEMENTS logical input positions, but the
// last block can be partial. Never read input[index] when index >= N.
// Result: one partial SUM per block in output[blockIdx.x].
// -----------------------------------------------------------------------------
__global__ void exercise5_coarsenedSumAnyN(const float* input,
    float* output,
    unsigned int N) {
    __shared__ float partial[BLOCK_THREADS];

    unsigned int tid = threadIdx.x;
    unsigned int segment = (2 * COARSE_FACTOR * blockDim.x) * blockIdx.x;
    unsigned int i = segment + tid;

    // Zero is the identity for addition. Some threads may have no valid data.
    float partialSum = i < N ? input[i] : 0.0f;

    for (unsigned int c = 1; c < 2 * COARSE_FACTOR; ++c) {
        partialSum += (i + c * blockDim.x < N) ? input[i + c * blockDim.x] : 0.0f;
        // TODO 5a: Compute this thread's global input index for pass c.
        // unsigned int index = ???;
        // TODO 5b: Only load/accumulate input[index] when index < N.
        // Do NOT return early: every thread must reach __syncthreads().
    }

    partial[tid] = partialSum;

    // Finish with a block-wide reduction in shared memory.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
    }

    if (tid == 0) {
        output[blockIdx.x] = partial[0];
    }
}

// ========================= TEST HARNESS (provided) ==========================
// You should not need to modify anything below to complete the exercises.

enum class Exercise { Three, Four, Five };

std::vector<float> makeInput(std::size_t n, bool negativeOnly = false) {
    std::vector<float> input(n);
    for (std::size_t k = 0; k < n; ++k) {
        int v = static_cast<int>((k * 17 + 11) % 23) - 11;
        input[k] = negativeOnly ? -static_cast<float>(std::abs(v) + 1)
            : static_cast<float>(v);
    }
    return input;
}

std::vector<float> launchAndRead(Exercise exercise,
    const std::vector<float>& input,
    int threads,
    int outputCount) {
    float* deviceInput = nullptr;
    float* deviceOutput = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceInput),
        input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput),
        static_cast<std::size_t>(outputCount) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(deviceInput, input.data(),
        input.size() * sizeof(float), cudaMemcpyHostToDevice));

    // A recognizable sentinel makes incompletely written output easier to spot.
    std::vector<float> initialOutput(outputCount, 1234567.0f);
    CUDA_CHECK(cudaMemcpy(deviceOutput, initialOutput.data(),
        outputCount * sizeof(float), cudaMemcpyHostToDevice));

    switch (exercise) {
    case Exercise::Three:
        exercise3_rightAlignedSum << <1, threads >> > (
            deviceInput, deviceOutput);
        break;
    case Exercise::Four:
        exercise4_coarsenedMax << <outputCount, BLOCK_THREADS >> > (
            deviceInput, deviceOutput);
        break;
    case Exercise::Five:
        exercise5_coarsenedSumAnyN << <outputCount, BLOCK_THREADS >> > (
            deviceInput, deviceOutput, static_cast<unsigned int>(input.size()));
        break;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result(outputCount);
    CUDA_CHECK(cudaMemcpy(result.data(), deviceOutput,
        outputCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceOutput));
    CUDA_CHECK(cudaFree(deviceInput));
    return result;
}

bool approximatelyEqual(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return a == b;
    return std::fabs(a - b) <= 1e-3f * std::max(1.0f, std::fabs(b));
}

bool checkResults(const std::string& name,
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        std::cout << "[FAIL] " << name << " (wrong output count)\n";
        return false;
    }
    for (std::size_t k = 0; k < actual.size(); ++k) {
        if (!approximatelyEqual(actual[k], expected[k])) {
            std::cout << "[FAIL] " << name << ": partial[" << k << "] expected "
                << expected[k] << ", got " << actual[k] << '\n';
            return false;
        }
    }
    std::cout << "[PASS] " << name << '\n';
    return true;
}

std::vector<float> referencePartials(const std::vector<float>& input,
    int segmentElements,
    bool maximum) {
    std::size_t segment = static_cast<std::size_t>(segmentElements);
    std::size_t blocks = (input.size() + segment - 1) / segment;
    std::vector<float> result;
    result.reserve(blocks);

    for (std::size_t b = 0; b < blocks; ++b) {
        std::size_t start = b * segment;
        std::size_t end = std::min(start + segment, input.size());
        float value = maximum ? -std::numeric_limits<float>::infinity() : 0.0f;
        for (std::size_t k = start; k < end; ++k) {
            if (maximum) value = std::max(value, input[k]);
            else value += input[k];
        }
        result.push_back(value);
    }
    return result;
}

bool testExercise3() {
    std::cout << "\nExercise 3: right-aligned reduction\n";
    bool ok = true;
    for (int threads : {8, 32, 128, 256}) {
        auto input = makeInput(2 * static_cast<std::size_t>(threads));
        float sum = 0.0f;
        for (float v : input) sum += v;
        auto actual = launchAndRead(Exercise::Three, input, threads, 1);
        ok = checkResults("blockDim.x=" + std::to_string(threads),
            actual, { sum }) && ok;
    }
    return ok;
}

bool testExercise4() {
    std::cout << "\nExercise 4: coarsened MAX reduction\n";
    bool ok = true;
    for (bool negativeOnly : {false, true}) {
        for (int blocks : {1, 3}) {
            auto input = makeInput(static_cast<std::size_t>(blocks) * SEGMENT_ELEMENTS,
                negativeOnly);
            auto actual = launchAndRead(Exercise::Four, input, BLOCK_THREADS, blocks);
            auto expected = referencePartials(input, SEGMENT_ELEMENTS, true);
            std::string label = "blocks=" + std::to_string(blocks) +
                (negativeOnly ? " (all negative)" : " (mixed signs)");
            ok = checkResults(label, actual, expected) && ok;
        }
    }
    return ok;
}

bool testExercise5() {
    std::cout << "\nExercise 5: coarsened SUM for arbitrary N\n";
    bool ok = true;
    const std::vector<std::size_t> lengths = { 1, 7, 31, 127, 128, 129,
                                              255, 256, 257, 1023, 1024, 1025, 4099, 8195 };
    for (std::size_t n : lengths) {
        auto input = makeInput(n);
        int blocks = static_cast<int>((n + SEGMENT_ELEMENTS - 1) / SEGMENT_ELEMENTS);
        auto actual = launchAndRead(Exercise::Five, input, BLOCK_THREADS, blocks);
        auto expected = referencePartials(input, SEGMENT_ELEMENTS, false);
        ok = checkResults("N=" + std::to_string(n) +
            ", blocks=" + std::to_string(blocks), actual, expected) && ok;
    }
    return ok;
}

int main() {
    std::cout << "PMPP Chapter 10: Exercises 3, 4, 5\n"
        << "blockDim.x=" << BLOCK_THREADS << " threads = "
        << (BLOCK_THREADS / 32) << " warps per block\n"
        << "coarsened segment length=" << SEGMENT_ELEMENTS << '\n';

    bool ok3 = testExercise3();
    bool ok4 = testExercise4();
    bool ok5 = testExercise5();

    std::cout << "\n" << ((ok3 && ok4 && ok5) ? "ALL TESTS PASS" :
        "SOME TESTS FAILED (expected until you fill the TODOs)")
        << '\n';
    return (ok3 && ok4 && ok5) ? EXIT_SUCCESS : EXIT_FAILURE;
}
