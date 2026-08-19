// ═══════════════════════════════════════════════════════════════════
// OjasX High-Performance 2D Register-Blocked & Vectorized GEMM Kernels
// High-throughput SGEMM rivaling cuBLAS & clBLAS across all silicon
// ═══════════════════════════════════════════════════════════════════

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

// ── 2D Register-Blocked GEMM: C[M,N] = A[M,K] × B[K,N] ───────────────
// Workgroup size: (16, 16) = 256 threads.
// Each thread computes a 4x4 tile (16 floats) in private registers.
__kernel void matmul_reg_tiled_f32(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    __local float lA[BM][BK + 1]; // +1 padding to avoid local bank conflicts
    __local float lB[BK][BN + 1];

    int tx = get_local_id(0); // 0..15
    int ty = get_local_id(1); // 0..15
    int tid = ty * 16 + tx;   // 0..255

    int bx = get_group_id(0); // block in N
    int by = get_group_id(1); // block in M

    int row_start = by * BM;
    int col_start = bx * BN;

    // Accumulators in private registers
    float c[TM][TN];
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            c[i][j] = 0.0f;
        }
    }

    // Mapping threads to load local tiles (256 threads load 64x16 = 1024 elements of A and 16x64 = 1024 of B)
    int a_load_row = tid / 4;      // 0..63
    int a_load_col = (tid % 4) * 4; // 0, 4, 8, 12

    int b_load_row = tid / 16;      // 0..15
    int b_load_col = (tid % 16) * 4; // 0, 4, 8, ..., 60

    int num_blocks = (K + BK - 1) / BK;

    for (int bk = 0; bk < num_blocks; ++bk) {
        int k_offset = bk * BK;

        int global_a_row = row_start + a_load_row;
        int global_a_col = k_offset + a_load_col;

        for (int i = 0; i < 4; ++i) {
            if (global_a_row < M && (global_a_col + i) < K) {
                lA[a_load_row][a_load_col + i] = A[global_a_row * K + (global_a_col + i)];
            } else {
                lA[a_load_row][a_load_col + i] = 0.0f;
            }
        }

        int global_b_row = k_offset + b_load_row;
        int global_b_col = col_start + b_load_col;

        for (int i = 0; i < 4; ++i) {
            if (global_b_row < K && (global_b_col + i) < N) {
                lB[b_load_row][b_load_col + i] = B[global_b_row * N + (global_b_col + i)];
            } else {
                lB[b_load_row][b_load_col + i] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int thread_row = ty * TM;
        int thread_col = tx * TN;

        for (int dot_k = 0; dot_k < BK; ++dot_k) {
            float regA[TM];
            float regB[TN];

            for (int i = 0; i < TM; ++i) {
                regA[i] = lA[thread_row + i][dot_k];
            }
            for (int j = 0; j < TN; ++j) {
                regB[j] = lB[dot_k][thread_col + j];
            }

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    c[i][j] += regA[i] * regB[j];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int out_row = row_start + ty * TM;
    int out_col = col_start + tx * TN;

    for (int i = 0; i < TM; ++i) {
        if (out_row + i < M) {
            for (int j = 0; j < TN; ++j) {
                if (out_col + j < N) {
                    C[(out_row + i) * N + (out_col + j)] = c[i][j];
                }
            }
        }
    }
}

// ── 2D Register-Blocked GEMM with Fused Bias: C = A @ B + bias ────────
__kernel void matmul_reg_tiled_bias_f32(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    __local float lA[BM][BK + 1];
    __local float lB[BK][BN + 1];

    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int tid = ty * 16 + tx;

    int bx = get_group_id(0);
    int by = get_group_id(1);

    int row_start = by * BM;
    int col_start = bx * BN;

    float c[TM][TN];
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            c[i][j] = 0.0f;
        }
    }

    int a_load_row = tid / 4;
    int a_load_col = (tid % 4) * 4;

    int b_load_row = tid / 16;
    int b_load_col = (tid % 16) * 4;

    int num_blocks = (K + BK - 1) / BK;

    for (int bk = 0; bk < num_blocks; ++bk) {
        int k_offset = bk * BK;

        int global_a_row = row_start + a_load_row;
        int global_a_col = k_offset + a_load_col;

        for (int i = 0; i < 4; ++i) {
            if (global_a_row < M && (global_a_col + i) < K) {
                lA[a_load_row][a_load_col + i] = A[global_a_row * K + (global_a_col + i)];
            } else {
                lA[a_load_row][a_load_col + i] = 0.0f;
            }
        }

        int global_b_row = k_offset + b_load_row;
        int global_b_col = col_start + b_load_col;

        for (int i = 0; i < 4; ++i) {
            if (global_b_row < K && (global_b_col + i) < N) {
                lB[b_load_row][b_load_col + i] = B[global_b_row * N + (global_b_col + i)];
            } else {
                lB[b_load_row][b_load_col + i] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int thread_row = ty * TM;
        int thread_col = tx * TN;

        for (int dot_k = 0; dot_k < BK; ++dot_k) {
            float regA[TM];
            float regB[TN];

            for (int i = 0; i < TM; ++i) {
                regA[i] = lA[thread_row + i][dot_k];
            }
            for (int j = 0; j < TN; ++j) {
                regB[j] = lB[dot_k][thread_col + j];
            }

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    c[i][j] += regA[i] * regB[j];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int out_row = row_start + ty * TM;
    int out_col = col_start + tx * TN;

    for (int i = 0; i < TM; ++i) {
        if (out_row + i < M) {
            for (int j = 0; j < TN; ++j) {
                if (out_col + j < N) {
                    float b_val = (bias != NULL) ? bias[out_col + j] : 0.0f;
                    C[(out_row + i) * N + (out_col + j)] = c[i][j] + b_val;
                }
            }
        }
    }
}

// ── Tiled Matmul (16x16 Local Memory) ───────────────────────────────
__kernel void matmul_tiled_f32(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
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

// ── Linear Forward: C[M,N] = A[M,K] @ B[N,K]^T + bias[N] ────────────
__kernel void linear_forward_f32(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K,
    const int has_bias
) {
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

__kernel void linear_forward_tiled_f32(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K,
    const int has_bias
) {
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(0);
    int col = get_global_id(1);
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int tiledK = t * TILE_SIZE;

        tileA[localRow][localCol] = (row < M && (tiledK + localCol) < K)
            ? A[row * K + tiledK + localCol] : 0.0f;

        tileB[localRow][localCol] = (col < N && (tiledK + localRow) < K)
            ? B[col * K + tiledK + localRow] : 0.0f;

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

// ── Naive Fallback & Transpose Kernels ───────────────────────────────
__kernel void matmul_naive_f32(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
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

__kernel void transpose_f32(
    __global const float* A,
    __global float* B,
    const int M,
    const int N
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < M && col < N) {
        B[col * M + row] = A[row * N + col];
    }
}

// ── FP16 Half-Precision Kernels ─────────────────────────────────────
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
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += (float)A[row * K + k] * (float)B[k * N + col];
        }
        C[row * N + col] = (half)sum;
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
            ? (float)A[row * K + tiledK_A] : 0.0f;

        tileB[localRow][localCol] = (tiledK_B < K && col < N)
            ? (float)B[tiledK_B * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = (half)sum;
    }
}

