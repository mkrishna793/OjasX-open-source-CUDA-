// ═══════════════════════════════════════════════════════════════════
// OjasX — Universal Fused Attention & Transformer LLM Kernels
// FlashAttention, Causal Masking, RoPE, SwiGLU
// ═══════════════════════════════════════════════════════════════════

#define MAX_N 2048

__kernel void flash_attention_f32(
    __global const float* Q,       // [B * H * M, D]
    __global const float* K,       // [B * H * N, D]
    __global const float* V,       // [B * H * N, D]
    __global float* Out,           // [B * H * M, D]
    const int B,                   // Batch size
    const int H,                   // Heads
    const int M,                   // Query seq len
    const int N,                   // Key/value seq len
    const int D,                   // Head dimension
    const float scale
) {
    int q_row = get_group_id(0);
    if (q_row >= B * H * M) return;

    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    int bh = q_row / M;
    int q_offset = q_row * D;
    int kv_offset = bh * N * D;

    __local float local_max[256];
    __local float local_sum[256];
    __local float scores[MAX_N];

    // Step 1: Compute query dot product with all keys
    for (int j = lid; j < N && j < MAX_N; j += lsize) {
        float sum = 0.0f;
        for (int k = 0; k < D; k++) {
            sum += Q[q_offset + k] * K[kv_offset + j * D + k];
        }
        scores[j] = sum * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: Softmax in local memory
    float my_max = -1e20f;
    for (int j = lid; j < N && j < MAX_N; j += lsize) {
        if (scores[j] > my_max) {
            my_max = scores[j];
        }
    }
    local_max[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float g_max = local_max[0];
        for (int i = 1; i < lsize; i++) {
            if (local_max[i] > g_max) g_max = local_max[i];
        }
        local_max[0] = g_max;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float row_max = local_max[0];

    float my_sum = 0.0f;
    for (int j = lid; j < N && j < MAX_N; j += lsize) {
        scores[j] = exp(scores[j] - row_max);
        my_sum += scores[j];
    }
    local_sum[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float g_sum = 0.0f;
        for (int i = 0; i < lsize; i++) {
            g_sum += local_sum[i];
        }
        local_sum[0] = g_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float row_sum = local_sum[0];

    for (int j = lid; j < N && j < MAX_N; j += lsize) {
        scores[j] /= (row_sum + 1e-6f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: Out = Prob * V
    for (int k = lid; k < D; k += lsize) {
        float sum = 0.0f;
        for (int j = 0; j < N && j < MAX_N; j++) {
            sum += scores[j] * V[kv_offset + j * D + k];
        }
        Out[q_offset + k] = sum;
    }
}

// ── Causal Masked FlashAttention for Autoregressive LLM Generation ───
__kernel void flash_attention_causal_f32(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* Out,
    const int B,
    const int H,
    const int M,
    const int N,
    const int D,
    const float scale
) {
    int q_row = get_group_id(0);
    if (q_row >= B * H * M) return;

    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    int bh = q_row / M;
    int q_idx = q_row % M;
    int q_offset = q_row * D;
    int kv_offset = bh * N * D;

    __local float local_max[256];
    __local float local_sum[256];
    __local float scores[MAX_N];

    for (int j = lid; j < N && j < MAX_N; j += lsize) {
        if (j <= q_idx) {
            float sum = 0.0f;
            for (int k = 0; k < D; k++) {
                sum += Q[q_offset + k] * K[kv_offset + j * D + k];
            }
            scores[j] = sum * scale;
        } else {
            scores[j] = -1e20f; // Causal mask
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float my_max = -1e20f;
    for (int j = lid; j <= q_idx && j < MAX_N; j += lsize) {
        if (scores[j] > my_max) my_max = scores[j];
    }
    local_max[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float g_max = local_max[0];
        for (int i = 1; i < lsize; i++) {
            if (local_max[i] > g_max) g_max = local_max[i];
        }
        local_max[0] = g_max;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float row_max = local_max[0];

    float my_sum = 0.0f;
    for (int j = lid; j < N && j < MAX_N; j += lsize) {
        if (j <= q_idx) {
            scores[j] = exp(scores[j] - row_max);
            my_sum += scores[j];
        } else {
            scores[j] = 0.0f;
        }
    }
    local_sum[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float g_sum = 0.0f;
        for (int i = 0; i < lsize; i++) {
            g_sum += local_sum[i];
        }
        local_sum[0] = g_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float row_sum = local_sum[0];

    for (int j = lid; j <= q_idx && j < MAX_N; j += lsize) {
        scores[j] /= (row_sum + 1e-6f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = lid; k < D; k += lsize) {
        float sum = 0.0f;
        for (int j = 0; j <= q_idx && j < MAX_N; j++) {
            sum += scores[j] * V[kv_offset + j * D + k];
        }
        Out[q_offset + k] = sum;
    }
}

// ── Rotary Positional Embeddings (RoPE) ──────────────────────────────
__kernel void rope_f32(
    __global float* X,               // [B, H, SeqLen, D]
    __global const float* cos_table, // [SeqLen, D / 2]
    __global const float* sin_table, // [SeqLen, D / 2]
    const int total_tokens,          // B * H * SeqLen
    const int half_dim               // D / 2
) {
    int token_idx = get_global_id(0);
    int dim_pair = get_global_id(1);

    if (token_idx >= total_tokens || dim_pair >= half_dim) return;

    int D = half_dim * 2;
    int base = token_idx * D;

    float x0 = X[base + dim_pair];
    float x1 = X[base + half_dim + dim_pair];

    float c = cos_table[dim_pair];
    float s = sin_table[dim_pair];

    X[base + dim_pair] = x0 * c - x1 * s;
    X[base + half_dim + dim_pair] = x0 * s + x1 * c;
}

// ── Fused SwiGLU Gated Activation ────────────────────────────────────
// Out = SiLU(Gate) * Up = (Gate / (1 + exp(-Gate))) * Up
__kernel void swiglu_f32(
    __global const float* gate,
    __global const float* up,
    __global float* out,
    const int N
) {
    int gid = get_global_id(0);
    if (gid < N) {
        float g = gate[gid];
        float u = up[gid];
        float silu_g = g / (1.0f + exp(-g));
        out[gid] = silu_g * u;
    }
}
