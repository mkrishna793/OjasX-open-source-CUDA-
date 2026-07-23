// ═══════════════════════════════════════════════════════════════════
// TorchCL — Matrix Multiplication Kernels
// Naive version + Tiled version (uses local memory for speed)
// ═══════════════════════════════════════════════════════════════════

// ── Naive matmul: C[M,N] = A[M,K] × B[K,N] ────────────────────────
__kernel void matmul_naive_f32(__global const float* A,
                               __global const float* B,
                               __global float* C,
                               const int M,
                               const int N,
                               const int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ── Tiled matmul: uses local memory for much better performance ─────
// TILE_SIZE is defined at compile time via -DTILE_SIZE=16
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

__kernel void matmul_tiled_f32(__global const float* A,
                               __global const float* B,
                               __global float* C,
                               const int M,
                               const int N,
                               const int K) {
    // Local memory tiles
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(0);
    int col = get_global_id(1);
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tiles into local memory
        int tiledK_A = t * TILE_SIZE + localCol;
        int tiledK_B = t * TILE_SIZE + localRow;

        tileA[localRow][localCol] = (row < M && tiledK_A < K)
            ? A[row * K + tiledK_A] : 0.0f;

        tileB[localRow][localCol] = (tiledK_B < K && col < N)
            ? B[tiledK_B * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot product from this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ── Matrix + bias: C = A @ B + bias ─────────────────────────────────
__kernel void matmul_bias_f32(__global const float* A,
                              __global const float* B,
                              __global const float* bias,
                              __global float* C,
                              const int M,
                              const int N,
                              const int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum + bias[col];
    }
}

// ── Transpose: B[N,M] = A[M,N]^T ───────────────────────────────────
__kernel void transpose_f32(__global const float* A,
                            __global float* B,
                            const int M,
                            const int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < M && col < N) {
        B[col * M + row] = A[row * N + col];
    }
}

// ── Linear Forward: C[M,N] = A[M,K] @ B[N,K]^T + bias[N] ────────────────
__kernel void linear_forward_f32(__global const float* A,
                                 __global const float* B,
                                 __global const float* bias,
                                 __global float* C,
                                 const int M,
                                 const int N,
                                 const int K,
                                 const int has_bias) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];
        }
        if (has_bias) {
            C[row * N + col] = sum + bias[col];
        } else {
            C[row * N + col] = sum;
        }
    }
}

__kernel void linear_forward_tiled_f32(__global const float* A,
                                       __global const float* B,
                                       __global const float* bias,
                                       __global float* C,
                                       const int M,
                                       const int N,
                                       const int K,
                                       const int has_bias) {
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(0);
    int col = get_global_id(1);
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int tiledK_A = t * TILE_SIZE + localCol;
        int tiledK_B = t * TILE_SIZE + localRow;

        tileA[localRow][localCol] = (row < M && tiledK_A < K)
            ? A[row * K + tiledK_A] : 0.0f;

        tileB[localRow][localCol] = (col < N && tiledK_B < K)
            ? B[col * K + tiledK_B] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        if (has_bias) {
            C[row * N + col] = sum + bias[col];
        } else {
            C[row * N + col] = sum;
        }
    }
}

// ── FP16 matrix multiplication kernels ───────────────────────────────
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matmul_naive_fp16(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int N,
    const int K
) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        half sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__kernel void matmul_tiled_fp16(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int N,
    const int K
) {
    __local half tileA[TILE_SIZE][TILE_SIZE];
    __local half tileB[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(0);
    int col = get_global_id(1);
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);

    half sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int tiledK_A = t * TILE_SIZE + localCol;
        int tiledK_B = t * TILE_SIZE + localRow;

        tileA[localRow][localCol] = (row < M && tiledK_A < K)
            ? A[row * K + tiledK_A] : 0.0f;

        tileB[localRow][localCol] = (tiledK_B < K && col < N)
            ? B[tiledK_B * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
