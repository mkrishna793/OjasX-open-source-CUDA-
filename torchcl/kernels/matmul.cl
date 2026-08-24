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

// Unaligned vload4 when the 4-wide window is in-bounds; scalar otherwise.
#define LOAD_A_VEC4()                                                          \
    do {                                                                       \
        int _ar = row_start + a_load_row;                                      \
        int _ac = k_offset + a_load_col;                                       \
        if (_ar < M && (_ac + 3) < K) {                                        \
            float4 va = vload4(0, A + _ar * K + _ac);                          \
            lA[a_load_row][a_load_col + 0] = va.s0;                            \
            lA[a_load_row][a_load_col + 1] = va.s1;                            \
            lA[a_load_row][a_load_col + 2] = va.s2;                            \
            lA[a_load_row][a_load_col + 3] = va.s3;                            \
        } else {                                                               \
            for (int i = 0; i < 4; ++i) {                                      \
                lA[a_load_row][a_load_col + i] =                               \
                    (_ar < M && (_ac + i) < K) ? A[_ar * K + (_ac + i)] : 0.0f;\
            }                                                                  \
        }                                                                      \
    } while (0)

#define LOAD_B_NN_VEC4()                                                       \
    do {                                                                       \
        int _br = k_offset + b_load_row;                                       \
        int _bc = col_start + b_load_col;                                      \
        if (_br < K && (_bc + 3) < N) {                                        \
            float4 vb = vload4(0, B + _br * N + _bc);                          \
            lB[b_load_row][b_load_col + 0] = vb.s0;                            \
            lB[b_load_row][b_load_col + 1] = vb.s1;                            \
            lB[b_load_row][b_load_col + 2] = vb.s2;                            \
            lB[b_load_row][b_load_col + 3] = vb.s3;                            \
        } else {                                                               \
            for (int i = 0; i < 4; ++i) {                                      \
                lB[b_load_row][b_load_col + i] =                               \
                    (_br < K && (_bc + i) < N) ? B[_br * N + (_bc + i)] : 0.0f;\
            }                                                                  \
        }                                                                      \
    } while (0)

#define GEMM_FMA_4x4()                                                         \
    do {                                                                       \
        int thread_row = ty * TM;                                              \
        int thread_col = tx * TN;                                              \
        for (int dot_k = 0; dot_k < BK; ++dot_k) {                             \
            float regA0 = lA[thread_row + 0][dot_k];                           \
            float regA1 = lA[thread_row + 1][dot_k];                           \
            float regA2 = lA[thread_row + 2][dot_k];                           \
            float regA3 = lA[thread_row + 3][dot_k];                           \
            float regB0 = lB[dot_k][thread_col + 0];                           \
            float regB1 = lB[dot_k][thread_col + 1];                           \
            float regB2 = lB[dot_k][thread_col + 2];                           \
            float regB3 = lB[dot_k][thread_col + 3];                           \
            c[0][0] += regA0 * regB0; c[0][1] += regA0 * regB1;                \
            c[0][2] += regA0 * regB2; c[0][3] += regA0 * regB3;                \
            c[1][0] += regA1 * regB0; c[1][1] += regA1 * regB1;                \
            c[1][2] += regA1 * regB2; c[1][3] += regA1 * regB3;                \
            c[2][0] += regA2 * regB0; c[2][1] += regA2 * regB1;                \
            c[2][2] += regA2 * regB2; c[2][3] += regA2 * regB3;                \
            c[3][0] += regA3 * regB0; c[3][1] += regA3 * regB1;                \
            c[3][2] += regA3 * regB2; c[3][3] += regA3 * regB3;                \
        }                                                                      \
    } while (0)


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
        LOAD_A_VEC4();
        LOAD_B_NN_VEC4();
        barrier(CLK_LOCAL_MEM_FENCE);
        GEMM_FMA_4x4();
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
        LOAD_A_VEC4();
        LOAD_B_NN_VEC4();
        barrier(CLK_LOCAL_MEM_FENCE);
        GEMM_FMA_4x4();
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

// ── Register-blocked GEMM + bias + ReLU (Linear+ReLU in one launch) ──
__kernel void matmul_reg_tiled_bias_relu_f32(
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
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            c[i][j] = 0.0f;

    int a_load_row = tid / 4;
    int a_load_col = (tid % 4) * 4;
    int b_load_row = tid / 16;
    int b_load_col = (tid % 16) * 4;
    int num_blocks = (K + BK - 1) / BK;

    for (int bk = 0; bk < num_blocks; ++bk) {
        int k_offset = bk * BK;
        LOAD_A_VEC4();
        LOAD_B_NN_VEC4();
        barrier(CLK_LOCAL_MEM_FENCE);
        GEMM_FMA_4x4();
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int out_row = row_start + ty * TM;
    int out_col = col_start + tx * TN;
    for (int i = 0; i < TM; ++i) {
        if (out_row + i < M) {
            for (int j = 0; j < TN; ++j) {
                if (out_col + j < N) {
                    float v = c[i][j] + ((bias != NULL) ? bias[out_col + j] : 0.0f);
                    C[(out_row + i) * N + (out_col + j)] = fmax(v, 0.0f);
                }
            }
        }
    }
}

// ── Register-blocked Linear: C[M,N] = A[M,K] @ W[N,K]^T (+ bias) ─────
// Weight stays in [out, in] layout — no transpose kernel per forward.
__kernel void linear_reg_tiled_f32(
    __global const float* A,
    __global const float* W,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K,
    const int has_bias
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
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            c[i][j] = 0.0f;

    int a_load_row = tid / 4;
    int a_load_col = (tid % 4) * 4;
    int b_load_row = tid / 16;
    int b_load_col = (tid % 16) * 4;
    int num_blocks = (K + BK - 1) / BK;

    for (int bk = 0; bk < num_blocks; ++bk) {
        int k_offset = bk * BK;
        LOAD_A_VEC4();
        {
            int _br = k_offset + b_load_row;
            int _bc = col_start + b_load_col;
            for (int i = 0; i < 4; ++i) {
                int n = _bc + i;
                lB[b_load_row][b_load_col + i] =
                    (n < N && _br < K) ? W[n * K + _br] : 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        GEMM_FMA_4x4();
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int out_row = row_start + ty * TM;
    int out_col = col_start + tx * TN;
    for (int i = 0; i < TM; ++i) {
        if (out_row + i < M) {
            for (int j = 0; j < TN; ++j) {
                if (out_col + j < N) {
                    float v = c[i][j];
                    if (has_bias) v += bias[out_col + j];
                    C[(out_row + i) * N + (out_col + j)] = v;
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

__kernel void matmul_bias_f32(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
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
        C[row * N + col] = sum + bias[col];
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

