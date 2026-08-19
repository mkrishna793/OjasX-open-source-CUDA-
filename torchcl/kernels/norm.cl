// ═══════════════════════════════════════════════════════════════════
// OjasX — High-Performance Normalization OpenCL Kernels
// Workgroup-Parallel LayerNorm, RMSNorm, BatchNorm
// ═══════════════════════════════════════════════════════════════════

// ── Workgroup-Parallel LayerNorm Forward ────────────────────────────
// Each row is computed by a full workgroup (256 threads) in parallel
__kernel void layer_norm_workgroup_f32(
    __global const float* input,
    __global const float* weight,   // gamma [N]
    __global const float* bias,     // beta  [N]
    __global float* output,
    __global float* mean_out,       // [M]
    __global float* rstd_out,       // [M]
    const int M,
    const int N,
    const float eps
) {
    int row = get_group_id(0);
    if (row >= M) return;

    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    __local float scratch[256];
    __local float s_mean;
    __local float s_rstd;

    int offset = row * N;

    // 1. Parallel Mean Reduction
    float my_sum = 0.0f;
    for (int j = lid; j < N; j += group_size) {
        my_sum += input[offset + j];
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
        s_mean = scratch[0] / (float)N;
        if (mean_out != NULL) mean_out[row] = s_mean;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Parallel Variance Reduction
    float my_var = 0.0f;
    for (int j = lid; j < N; j += group_size) {
        float diff = input[offset + j] - s_mean;
        my_var += diff * diff;
    }
    scratch[lid] = my_var;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        float variance = scratch[0] / (float)N;
        s_rstd = 1.0f / sqrt(variance + eps);
        if (rstd_out != NULL) rstd_out[row] = s_rstd;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. Parallel Normalized Output Store
    for (int j = lid; j < N; j += group_size) {
        float w = (weight != NULL) ? weight[j] : 1.0f;
        float b = (bias != NULL) ? bias[j] : 0.0f;
        output[offset + j] = (input[offset + j] - s_mean) * s_rstd * w + b;
    }
}

// ── Workgroup-Parallel RMSNorm Forward ──────────────────────────────
__kernel void rms_norm_workgroup_f32(
    __global const float* input,
    __global const float* weight,     // [N]
    __global float* output,
    __global float* rrms_out,         // [M]
    const int M,
    const int N,
    const float eps
) {
    int row = get_group_id(0);
    if (row >= M) return;

    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    __local float scratch[256];
    __local float s_rrms;

    int offset = row * N;

    // 1. Parallel Mean(x^2) Reduction
    float my_sq = 0.0f;
    for (int j = lid; j < N; j += group_size) {
        float v = input[offset + j];
        my_sq += v * v;
    }
    scratch[lid] = my_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        float rms = sqrt(scratch[0] / (float)N + eps);
        s_rrms = 1.0f / (rms > 0.0f ? rms : 1e-12f);
        if (rrms_out != NULL) rrms_out[row] = s_rrms;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Parallel Scale & Store
    for (int j = lid; j < N; j += group_size) {
        float w = (weight != NULL) ? weight[j] : 1.0f;
        output[offset + j] = input[offset + j] * s_rrms * w;
    }
}

// ── Fallback 1-Thread LayerNorm Forward ─────────────────────────────
__kernel void layer_norm_f32(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    __global float* mean_out,
    __global float* rstd_out,
    const int M,
    const int N,
    const float eps
) {
    int row = get_global_id(0);
    if (row >= M) return;

    int offset = row * N;

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += input[offset + j];
    }
    float mu = sum / (float)N;

    float var_sum = 0.0f;
    for (int j = 0; j < N; j++) {
        float diff = input[offset + j] - mu;
        var_sum += diff * diff;
    }
    float variance = var_sum / (float)N;
    float rstd = 1.0f / sqrt(variance + eps);

    if (mean_out != NULL) mean_out[row] = mu;
    if (rstd_out != NULL) rstd_out[row] = rstd;

    for (int j = 0; j < N; j++) {
        float w = (weight != NULL) ? weight[j] : 1.0f;
        float b = (bias != NULL) ? bias[j] : 0.0f;
        output[offset + j] = (input[offset + j] - mu) * rstd * w + b;
    }
}

// ── Fallback 1-Thread RMSNorm Forward ───────────────────────────────
__kernel void rms_norm_f32(
    __global const float* input,
    __global const float* weight,
    __global float* output,
    __global float* rrms_out,
    const int M,
    const int N,
    const float eps
) {
    int row = get_global_id(0);
    if (row >= M) return;

    int offset = row * N;

    float sq_sum = 0.0f;
    for (int j = 0; j < N; j++) {
        float v = input[offset + j];
        sq_sum += v * v;
    }
    float rms = sqrt(sq_sum / (float)N + eps);
    float rrms = 1.0f / (rms > 0.0f ? rms : 1e-12f);

    if (rrms_out != NULL) rrms_out[row] = rrms;

    for (int j = 0; j < N; j++) {
        float w = (weight != NULL) ? weight[j] : 1.0f;
        output[offset + j] = input[offset + j] * rrms * w;
    }
}

// ── LayerNorm backward (grad_input) ─────────────────────────────────
__kernel void layer_norm_backward_f32(
    __global const float* grad_out,
    __global const float* input,
    __global const float* weight,
    __global const float* mean,
    __global const float* rstd,
    __global float* grad_input,
    const int M,
    const int N
) {
    int row = get_global_id(0);
    if (row >= M) return;

    int offset = row * N;
    float mu = mean[row];
    float rs = rstd[row];

    float ds = 0.0f;
    float db = 0.0f;
    for (int j = 0; j < N; j++) {
        float g = grad_out[offset + j];
        float x_hat = (input[offset + j] - mu) * rs;
        float w = (weight != NULL) ? weight[j] : 1.0f;
        ds += g * w * x_hat;
        db += g * w;
    }

    float inv_N = 1.0f / (float)N;
    for (int j = 0; j < N; j++) {
        float x_hat = (input[offset + j] - mu) * rs;
        float g = grad_out[offset + j];
        float w = (weight != NULL) ? weight[j] : 1.0f;
        grad_input[offset + j] = rs * (g * w - inv_N * (x_hat * ds + db));
    }
}

// ── LayerNorm backward (grad_weight, grad_bias) ─────────────────────
__kernel void layer_norm_grad_weight_bias_f32(
    __global const float* grad_out,
    __global const float* input,
    __global const float* mean,
    __global const float* rstd,
    __global float* grad_weight,
    __global float* grad_bias,
    const int M,
    const int N
) {
    int col = get_global_id(0);
    if (col >= N) return;

    float gw = 0.0f;
    float gb = 0.0f;
    for (int row = 0; row < M; ++row) {
        float g = grad_out[row * N + col];
        float x_hat = (input[row * N + col] - mean[row]) * rstd[row];
        gw += g * x_hat;
        gb += g;
    }
    grad_weight[col] = gw;
    grad_bias[col] = gb;
}

// ── BatchNorm Forward ───────────────────────────────────────────────
__kernel void batch_norm_f32(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    __global float* mean_out,
    __global float* var_out,
    const int batch_size,
    const int C,
    const int spatial,
    const float eps,
    const float momentum
) {
    int c = get_global_id(0);
    if (c >= C) return;

    int count = batch_size * spatial;
    float sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        for (int s = 0; s < spatial; s++) {
            sum += input[n * C * spatial + c * spatial + s];
        }
    }
    float mu = sum / (float)count;

    float var_sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        for (int s = 0; s < spatial; s++) {
            float diff = input[n * C * spatial + c * spatial + s] - mu;
            var_sum += diff * diff;
        }
    }
    float variance = var_sum / (float)count;

    if (mean_out != NULL) mean_out[c] = mu;
    if (var_out != NULL) var_out[c] = variance;

    float rstd = 1.0f / sqrt(variance + eps);

    for (int n = 0; n < batch_size; n++) {
        for (int s = 0; s < spatial; s++) {
            int idx = n * C * spatial + c * spatial + s;
            float normed = (input[idx] - mu) * rstd;
            float w = (weight != NULL) ? weight[c] : 1.0f;
            float b = (bias != NULL) ? bias[c] : 0.0f;
            output[idx] = normed * w + b;
        }
    }
}
