// PMPP Chapter 7, Exercises 8-11: CUDA starter project
// Compatible with a standard Visual Studio CUDA Runtime project.
//
// All four GPU kernels are intentionally unfinished.
// Complete the TODOs yourself. The CPU reference produces the expected answers.
//
// Conventions:
//   3D arrays: index = (z * height + y) * width + x  (x is fastest)
//   3D filter: index = (fz * FILTER_WIDTH + fy) * FILTER_WIDTH + fx
//   2D arrays: index = y * width + x
//   Zero padding: any input coordinate outside the array contributes 0.
//   Filters use the orientation of PMPP Figures 7.7, 7.9, 7.12
//   (no flipping of filter indices).

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

//////////// Helper functions ///////////////////
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = (call);                                           \
        if (error != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,         \
                __LINE__, cudaGetErrorString(error));                         \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                     \
    } while (0)

__host__ __device__  int IDX2D(int row, int col, int width)
{
    return row * width + col;
}

__host__ __device__ int IDX3D(int z, int y, int x, int height, int width)
{
    return z * height * width + IDX2D(y, x, width);
}

/////////////////////////////////////////////////

constexpr int RADIUS = 1;
constexpr int FILTER_WIDTH = 2 * RADIUS + 1;
constexpr int FILTER_3D_SIZE = FILTER_WIDTH * FILTER_WIDTH * FILTER_WIDTH;
constexpr int FILTER_2D_SIZE = FILTER_WIDTH * FILTER_WIDTH;

// Exercise 10: input tile is 4x4x4; output tile is 2x2x2.
// Figure 7.12 has a 32x32 2D input tile; a 32x32x32 3D thread block
// would exceed the CUDA limit of 1024 threads/block, so this test uses 4.
constexpr int IN_3D_TILE = 4;
constexpr int OUT_3D_TILE = IN_3D_TILE - 2 * RADIUS;

// Exercise 11: 4x4 output threads load a 6x6 input tile.
// Figure 7.12 uses 32x32 input threads and a smaller output tile.
// Exercise 11 reverses the thread count: only 4x4 threads for 4x4 outputs.
constexpr int OUT_2D_TILE = 4;
constexpr int IN_2D_TILE = OUT_2D_TILE + 2 * RADIUS;

// Like the F[][] declaration in Figure 7.12, these filters live in
// GPU constant memory; exercise 9/10 extends F to three dimensions.
__constant__ float constantFilter3D[FILTER_WIDTH][FILTER_WIDTH][FILTER_WIDTH];
__constant__ float constantFilter2D[FILTER_WIDTH][FILTER_WIDTH];

// ---------------------------------------------------------------------------
// EXERCISE 8: Extend Figure 7.7 (basic 2D convolution) to 3D convolution.
// ---------------------------------------------------------------------------
__global__ void exercise8_basic3D(
    const float* input, const float* filter, float* output,
    int width, int height, int depth)
{
    // TODO 1: Calculate this thread's (outX, outY, outZ).
    // TODO 2: Return if the output location lies outside the volume.
    // TODO 3: Visit the 3x3x3 filter neighborhood, with zero padding.
    // TODO 4: Write the resulting value into output.

    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outZ = blockIdx.z * blockDim.z + threadIdx.z;

    if (outX >= width || outY >= height || outZ >= depth)
    {
        return;
    }

    float pValue = 0.0f;
    for (int fz = 0; fz < FILTER_WIDTH; fz++)
    {
        for (int fy = 0; fy < FILTER_WIDTH; fy++)
        {
            for (int fx = 0; fx < FILTER_WIDTH; fx++)
            {
                int inX = outX + fx - RADIUS;
                int inY = outY + fy - RADIUS;
                int inZ = outZ + fz - RADIUS;
                if (inX >= 0 && inX < width && inY >= 0 && inY < height && inZ >= 0 && inZ < depth)
                {
                    pValue += input[IDX3D(inZ, inY, inX, height, width)] * filter[IDX3D(fz, fy, fx, FILTER_WIDTH, FILTER_WIDTH)];
                }
            }
        }
    }
    output[IDX3D(outZ, outY, outX, height, width)] = pValue;
}

// ---------------------------------------------------------------------------
// EXERCISE 9: Extend Figure 7.9 (constant-memory filter) to 3D.
// Use constantFilter3D instead of a filter pointer argument.
// ---------------------------------------------------------------------------
__global__ void exercise9_constant3D(
    const float* input, float* output,
    int width, int height, int depth)
{
    // TODO 1: Calculate this thread's output coordinates in 3D.
    // TODO 2: Handle bounds / zero padding.
    // TODO 3: Accumulate the 3D convolution using
    //         constantFilter3D[fz][fy][fx] (not a filter pointer).
    // TODO 4: Store the output.

    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outZ = blockIdx.z * blockDim.z + threadIdx.z;

    if (outX >= width || outY >= height || outZ >= depth)
    {
        return;
    }

    float pValue = 0.0f;
    for (int fz = 0; fz < FILTER_WIDTH; fz++)
    {
        for (int fy = 0; fy < FILTER_WIDTH; fy++)
        {
            for (int fx = 0; fx < FILTER_WIDTH; fx++)
            {
                int inX = outX + fx - RADIUS;
                int inY = outY + fy - RADIUS;
                int inZ = outZ + fz - RADIUS;
                if (inX >= 0 && inX < width && inY >= 0 && inY < height && inZ >= 0 && inZ < depth)
                {
                    pValue += input[IDX3D(inZ, inY, inX, height, width)] * constantFilter3D[fz][fy][fx];
                }
            }
        }
    }
    output[IDX3D(outZ, outY, outX, height, width)] = pValue;
}

// ---------------------------------------------------------------------------
// EXERCISE 10: Extend Figure 7.12 (shared-memory input tile) to 3D.
// Launch uses 4x4x4 threads to stage a 4x4x4 input tile, then produces
// a 2x2x2 output tile. Constant filter from Exercise 9 is reused.
// ---------------------------------------------------------------------------
__global__ void exercise10_tiled3D(
    const float* input, float* output,
    int width, int height, int depth)
{
    __shared__ float tile[IN_3D_TILE][IN_3D_TILE][IN_3D_TILE];

    // TODO 1: Calculate the input coordinates corresponding to each
    //         thread's shared-memory location (include halo offset).
    // TODO 2: Cooperatively load the input tile. For out-of-bounds
    //         positions, store 0 into shared memory.

    int xGlobal = blockIdx.x * OUT_3D_TILE + threadIdx.x - RADIUS;
    int yGlobal = blockIdx.y * OUT_3D_TILE + threadIdx.y - RADIUS;
    int zGlobal = blockIdx.z * OUT_3D_TILE + threadIdx.z - RADIUS;

    if (xGlobal >= 0 && xGlobal < width && yGlobal >= 0 && yGlobal < height && zGlobal >= 0 && zGlobal < depth)
    {
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = input[IDX3D(zGlobal, yGlobal, xGlobal, height, width)];
    }
    else
    {
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Synchronize AFTER every participating thread has loaded its data.
    __syncthreads();

    // TODO 3: Identify the threads responsible for the 2x2x2 output tile.
    // TODO 4: Accumulate from tile[][][] using
    //         constantFilter3D[fz][fy][fx].
    // TODO 5: Write each valid output element.
    // IMPORTANT: Do not return before the __syncthreads() above.

    // If input thread is out of output array, return
    if (!(xGlobal >= 0 && xGlobal < width && yGlobal >= 0 && yGlobal < height && zGlobal >= 0 && zGlobal < depth))
    {
        return;
    }

    // Also then check if the thread in a block (which is input tile size) is actually within the output tile
    int xTile = threadIdx.x - RADIUS;
    int yTile = threadIdx.y - RADIUS;
    int zTile = threadIdx.z - RADIUS;
    if (!(xTile >= 0 && xTile < OUT_3D_TILE && yTile >= 0 && yTile < OUT_3D_TILE && zTile >= 0 && zTile < OUT_3D_TILE))
    {
        return;
    }

    float outputVal = 0.0f;
    for (int fz = 0; fz < FILTER_WIDTH; fz++)
    {
        for (int fy = 0; fy < FILTER_WIDTH; fy++)
        {
            for (int fx = 0; fx < FILTER_WIDTH; fx++)
            {
                outputVal += constantFilter3D[fz][fy][fx] * tile[zTile + fz][yTile + fy][xTile + fx];
            }
        }
    }
    output[IDX3D(zGlobal, yGlobal, xGlobal, height, width)] = outputVal;
}

// ---------------------------------------------------------------------------
// EXERCISE 11: Same 2D tiled convolution idea as Figure 7.12,
// but use only as many threads as there are OUTPUT tile elements.
// Launch uses 4x4 threads; shared input tile contains 6x6 elements.
// Therefore some threads MUST load more than one input element.
// ---------------------------------------------------------------------------
__global__ void exercise11_tiled2D_multipleLoads(
    const float* input, float* output, int width, int height)
{
    __shared__ float tile[IN_2D_TILE][IN_2D_TILE];

    // TODO 1: Determine the shared input tile's global origin.
    // TODO 2: Have the 4x4 threads cooperatively load ALL 6x6 values.
    //         Remember halo positions and zero padding.

    int col = blockIdx.x * OUT_2D_TILE + threadIdx.x;
    int row = blockIdx.y * OUT_2D_TILE + threadIdx.y;

    /*
    // Sooo, early return here is wrong, we still need this for the share memory loading
    if (col >= width || row >= height)
    {
        return;
    }
    */

    // blockDim.x and OUT_2D_TILE would be the same as the block now matches the output tile size

    // How manu elements to load?
    int tid = threadIdx.y * OUT_2D_TILE + threadIdx.x;
    for (int i = tid; i < IN_2D_TILE * IN_2D_TILE; i += OUT_2D_TILE * OUT_2D_TILE)
    {
        int tileRow = i / IN_2D_TILE;
        int tileCol = i % IN_2D_TILE;

        // Mapping the shared input tile's coordinates to the global input's coordinates
        int globalMemIdxX = blockIdx.x * OUT_2D_TILE + tileCol - RADIUS;
        int globalMemIdxY = blockIdx.y * OUT_2D_TILE + tileRow - RADIUS;
        if (globalMemIdxX >= 0 && globalMemIdxX < width && globalMemIdxY >= 0 && globalMemIdxY < height)
        {
            tile[tileRow][tileCol] = input[IDX2D(globalMemIdxY, globalMemIdxX, width)];
        }
        else
        {
            tile[tileRow][tileCol] = 0.0f;
        }
    }

    __syncthreads();

    // TODO 3: Map each thread to its own output pixel.
    // TODO 4: Compute convolution from tile using
    //         constantFilter2D[fy][fx].
    // TODO 5: Write valid output pixels.
    // IMPORTANT: Do not return before the __syncthreads() above.
    if (col < width && row < height)
    {
        float pValue = 0.0f;
        for (int fRow = 0; fRow < FILTER_WIDTH; fRow++)
        {
            for (int fCol = 0; fCol < FILTER_WIDTH; fCol++)
            {
                pValue += constantFilter2D[fRow][fCol] * tile[threadIdx.y + fRow][threadIdx.x + fCol];
            }
        }
        output[IDX2D(row, col, width)] = pValue;
    }
}

// ---------------------------------------------------------------------------
// CPU reference implementations (complete, for correctness checking).
// These are NOT solutions to the CUDA kernel exercises.
// ---------------------------------------------------------------------------
static void reference3D(const std::vector<float>& input,
    const std::vector<float>& filter, std::vector<float>& output,
    int width, int height, int depth)
{
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int fz = 0; fz < FILTER_WIDTH; ++fz) {
                    for (int fy = 0; fy < FILTER_WIDTH; ++fy) {
                        for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
                            const int iz = z + fz - RADIUS;
                            const int iy = y + fy - RADIUS;
                            const int ix = x + fx - RADIUS;
                            if (ix >= 0 && ix < width &&
                                iy >= 0 && iy < height &&
                                iz >= 0 && iz < depth) {
                                const int inputIndex = (iz * height + iy) * width + ix;
                                const int filterIndex = (fz * FILTER_WIDTH + fy) * FILTER_WIDTH + fx;
                                sum += input[inputIndex] * filter[filterIndex];
                            }
                        }
                    }
                }
                output[(z * height + y) * width + x] = sum;
            }
        }
    }
}

static void reference2D(const std::vector<float>& input,
    const std::vector<float>& filter, std::vector<float>& output,
    int width, int height)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int fy = 0; fy < FILTER_WIDTH; ++fy) {
                for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
                    const int iy = y + fy - RADIUS;
                    const int ix = x + fx - RADIUS;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        const int inputIndex = iy * width + ix;
                        const int filterIndex = fy * FILTER_WIDTH + fx;
                        sum += input[inputIndex] * filter[filterIndex];
                    }
                }
            }
            output[y * width + x] = sum;
        }
    }
}

static void print3D(const char* title, const std::vector<float>& a,
    int width, int height, int depth)
{
    std::printf("\n%s\n", title);
    for (int z = 0; z < depth; ++z) {
        std::printf("z=%d\n", z);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                std::printf("%6.0f", a[(z * height + y) * width + x]);
            }
            std::printf("\n");
        }
    }
}

static void print2D(const char* title, const std::vector<float>& a,
    int width, int height)
{
    std::printf("\n%s\n", title);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::printf("%6.0f", a[y * width + x]);
        }
        std::printf("\n");
    }
}

static bool verify(const char* title,
    const std::vector<float>& actual,
    const std::vector<float>& expected)
{
    int mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float a = actual[i];
        const float e = expected[i];
        if (!std::isfinite(a) || std::fabs(a - e) > 1e-4f) {
            if (mismatches < 5) {
                std::printf("  mismatch at flat index %zu: GPU=%g CPU=%g\n",
                    i, a, e);
            }
            ++mismatches;
        }
    }
    const bool passed = mismatches == 0;
    std::printf("%s: %s (%d mismatches)\n", title,
        passed ? "PASS" : "FAIL / not yet implemented", mismatches);
    return passed;
}

int main()
{
    // Non-square / non-cubic sizes intentionally test bounds and partial tiles.
    constexpr int W3 = 5, H3 = 4, D3 = 3;
    constexpr int W2 = 7, H2 = 6;

    std::vector<float> input3D(W3 * H3 * D3);
    std::vector<float> filter3D(FILTER_3D_SIZE, 1.0f);
    std::vector<float> expected3D(W3 * H3 * D3, 0.0f);
    std::vector<float> actual3D(W3 * H3 * D3, 0.0f);
    for (size_t i = 0; i < input3D.size(); ++i) input3D[i] = static_cast<float>(i + 1);
    // 3D filter: all 1s, except F[z=2,y=1,x=1] = 2.
    filter3D[(2 * FILTER_WIDTH + 1) * FILTER_WIDTH + 1] = 2.0f;
    reference3D(input3D, filter3D, expected3D, W3, H3, D3);

    std::vector<float> input2D(W2 * H2);
    std::vector<float> filter2D(FILTER_2D_SIZE, 1.0f);
    std::vector<float> expected2D(W2 * H2, 0.0f);
    std::vector<float> actual2D(W2 * H2, 0.0f);
    for (size_t i = 0; i < input2D.size(); ++i) input2D[i] = static_cast<float>(i + 1);
    // 2D filter: all 1s, except F[y=1,x=2] = 2.
    filter2D[1 * FILTER_WIDTH + 2] = 2.0f;
    reference2D(input2D, filter2D, expected2D, W2, H2);

    print3D("3D INPUT (5x4x3)", input3D, W3, H3, D3);
    print3D("EXPECTED 3D OUTPUT (Exercises 8, 9, 10)", expected3D, W3, H3, D3);
    print2D("2D INPUT (7x6)", input2D, W2, H2);
    print2D("EXPECTED 2D OUTPUT (Exercise 11)", expected2D, W2, H2);

    float *d_input3D = nullptr, *d_filter3D = nullptr, *d_output3D = nullptr;
    float *d_input2D = nullptr, *d_output2D = nullptr;
    const size_t bytes3D = input3D.size() * sizeof(float);
    const size_t bytes2D = input2D.size() * sizeof(float);

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc((void**)&d_input3D, bytes3D));
    CUDA_CHECK(cudaMalloc((void**)&d_filter3D, FILTER_3D_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output3D, bytes3D));
    CUDA_CHECK(cudaMalloc((void**)&d_input2D, bytes2D));
    CUDA_CHECK(cudaMalloc((void**)&d_output2D, bytes2D));

    CUDA_CHECK(cudaMemcpy(d_input3D, input3D.data(), bytes3D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter3D, filter3D.data(),
        FILTER_3D_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input2D, input2D.data(), bytes2D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(constantFilter3D, filter3D.data(),
        FILTER_3D_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(constantFilter2D, filter2D.data(),
        FILTER_2D_SIZE * sizeof(float)));

    // Exercise 8: basic 3D, one output per thread.
    {
        const dim3 block(4, 4, 2);
        const dim3 grid((W3 + block.x - 1) / block.x,
                        (H3 + block.y - 1) / block.y,
                        (D3 + block.z - 1) / block.z);
        CUDA_CHECK(cudaMemset(d_output3D, 0, bytes3D));
        exercise8_basic3D<<<grid, block>>>(d_input3D, d_filter3D, d_output3D, W3, H3, D3);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual3D.data(), d_output3D, bytes3D, cudaMemcpyDeviceToHost));
        verify("Exercise 8", actual3D, expected3D);
    }

    // Exercise 9: same launch, filter read from constant memory.
    {
        const dim3 block(4, 4, 2);
        const dim3 grid((W3 + block.x - 1) / block.x,
                        (H3 + block.y - 1) / block.y,
                        (D3 + block.z - 1) / block.z);
        CUDA_CHECK(cudaMemset(d_output3D, 0, bytes3D));
        exercise9_constant3D<<<grid, block>>>(d_input3D, d_output3D, W3, H3, D3);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual3D.data(), d_output3D, bytes3D, cudaMemcpyDeviceToHost));
        verify("Exercise 9", actual3D, expected3D);
    }

    // Exercise 10: 4x4x4 INPUT tile, producing 2x2x2 OUTPUT tile.
    {
        const dim3 block(IN_3D_TILE, IN_3D_TILE, IN_3D_TILE);
        const dim3 grid((W3 + OUT_3D_TILE - 1) / OUT_3D_TILE,
                        (H3 + OUT_3D_TILE - 1) / OUT_3D_TILE,
                        (D3 + OUT_3D_TILE - 1) / OUT_3D_TILE);
        CUDA_CHECK(cudaMemset(d_output3D, 0, bytes3D));
        exercise10_tiled3D<<<grid, block>>>(d_input3D, d_output3D, W3, H3, D3);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual3D.data(), d_output3D, bytes3D, cudaMemcpyDeviceToHost));
        verify("Exercise 10", actual3D, expected3D);
    }

    // Exercise 11: 4x4 OUTPUT threads cooperatively load a 6x6 INPUT tile.
    {
        const dim3 block(OUT_2D_TILE, OUT_2D_TILE);
        const dim3 grid((W2 + OUT_2D_TILE - 1) / OUT_2D_TILE,
                        (H2 + OUT_2D_TILE - 1) / OUT_2D_TILE);
        CUDA_CHECK(cudaMemset(d_output2D, 0, bytes2D));
        exercise11_tiled2D_multipleLoads<<<grid, block>>>(d_input2D, d_output2D, W2, H2);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual2D.data(), d_output2D, bytes2D, cudaMemcpyDeviceToHost));
        verify("Exercise 11", actual2D, expected2D);
    }

    CUDA_CHECK(cudaFree(d_input3D));
    CUDA_CHECK(cudaFree(d_filter3D));
    CUDA_CHECK(cudaFree(d_output3D));
    CUDA_CHECK(cudaFree(d_input2D));
    CUDA_CHECK(cudaFree(d_output2D));
    CUDA_CHECK(cudaDeviceReset());
    return 0; // The program runs even when unfinished kernels are reported as FAIL.
}
