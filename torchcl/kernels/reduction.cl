// ═══════════════════════════════════════════════════════════════════
// OjasX High-Performance Reduction & Softmax Kernels
// Workgroup-Parallel Local SRAM Tree Reductions
// ═══════════════════════════════════════════════════════════════════

// ── Sum reduction (full tensor → scalar) ────────────────────────────
__kernel void sum_f32(__global const float* input,
                      __global float* output,
                      __local float* scratch,
                      const int n) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    scratch[lid] = (gid < n) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[get_group_id(0)] = scratch[0];
    }
}

// ── Max reduction (full tensor → scalar) ────────────────────────────
__kernel void max_f32(__global const float* input,
                      __global float* output,
                      __local float* scratch,
                      const int n) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    scratch[lid] = (gid < n) ? input[gid] : -INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[get_group_id(0)] = scratch[0];
    }
}

// ── Min reduction (full tensor → scalar) ────────────────────────────
__kernel void min_f32(__global const float* input,
                      __global float* output,
                      __local float* scratch,
                      const int n) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    scratch[lid] = (gid < n) ? input[gid] : INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = fmin(scratch[lid], scratch[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[get_group_id(0)] = scratch[0];
    }
}

// ── High-Performance Workgroup-Parallel Softmax ─────────────────────
// Each row is computed by a full workgroup (e.g. 256 threads) in parallel
__kernel void softmax_workgroup_f32(
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols
) {
    int row = get_group_id(0);
    if (row >= rows) return;

    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    __local float scratch[256];
    __local float s_max;
    __local float s_sum;

    int offset = row * cols;

    // 1. Parallel Local Max Reduction
    float my_max = -INFINITY;
    for (int j = lid; j < cols; j += group_size) {
        my_max = fmax(my_max, input[offset + j]);
    }
    scratch[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        s_max = scratch[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Parallel Local Sum Reduction of exp(x - max)
    float my_sum = 0.0f;
    for (int j = lid; j < cols; j += group_size) {
        my_sum += exp(input[offset + j] - s_max);
    }
    scratch[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        s_sum = scratch[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. Parallel Normalized Output Store
    float inv_sum = 1.0f / (s_sum > 0.0f ? s_sum : 1e-12f);
    for (int j = lid; j < cols; j += group_size) {
        output[offset + j] = exp(input[offset + j] - s_max) * inv_sum;
    }
}

// ── Fallback 1-thread Softmax ───────────────────────────────────────
__kernel void softmax_f32(__global const float* input,
                          __global float* output,
                          const int rows,
                          const int cols) {
    int row = get_global_id(0);
    if (row >= rows) return;

    int offset = row * cols;

    float max_val = -INFINITY;
    for (int j = 0; j < cols; j++) {
        max_val = fmax(max_val, input[offset + j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        float e = exp(input[offset + j] - max_val);
        output[offset + j] = e;
        sum += e;
    }

    for (int j = 0; j < cols; j++) {
        output[offset + j] /= sum;
    }
}

__kernel void sum_columns_f32(__global const float* input,
                              __global float* output,
                              const int rows,
                              const int cols) {
    int col = get_global_id(0);
    if (col >= cols) return;

    float sum = 0.0f;
    for (int r = 0; r < rows; r++) {
        sum += input[r * cols + col];
    }
    output[col] = sum;
}

__kernel void sum_conv2d_bias_f32(__global const float* input,
                                  __global float* output,
                                  const int N,
                                  const int C,
                                  const int H,
                                  const int W) {
    int c = get_global_id(0);
    if (c >= C) return;

    float sum = 0.0f;
    int spatial = H * W;
    for (int n = 0; n < N; n++) {
        int offset = n * C * spatial + c * spatial;
        for (int i = 0; i < spatial; i++) {
            sum += input[offset + i];
        }
    }
    output[c] = sum;
}
