// ═══════════════════════════════════════════════════════════════════
// TorchCL — Reduction Kernels
// sum, mean, max, min — two-pass parallel reduction
// ═══════════════════════════════════════════════════════════════════

// ── Sum reduction (full tensor → scalar) ────────────────────────────
__kernel void sum_f32(__global const float* input,
                      __global float* output,
                      __local float* scratch,
                      const int n) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    // Load into local memory
    scratch[lid] = (gid < n) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in local memory
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result of this workgroup
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

// ── Row-wise softmax: out[i,j] = exp(a[i,j]) / sum_j(exp(a[i,j])) ─
__kernel void softmax_f32(__global const float* input,
                          __global float* output,
                          const int rows,
                          const int cols) {
    int row = get_global_id(0);
    if (row >= rows) return;

    int offset = row * cols;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int j = 0; j < cols; j++) {
        max_val = fmax(max_val, input[offset + j]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        float e = exp(input[offset + j] - max_val);
        output[offset + j] = e;
        sum += e;
    }

    // Normalize
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
