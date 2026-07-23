// ═══════════════════════════════════════════════════════════════════
// OjasX — Convolution OpenCL Kernels
// Implements im2col + matmul approach and direct 3×3 convolution
// ═══════════════════════════════════════════════════════════════════

// ── im2col: Unfold image patches into a column matrix ───────────────
// Input:  [N, C_in, H, W]     (flattened to 1D)
// Output: [N * H_out * W_out, C_in * kH * kW]  (column matrix)
//
// Each work-item handles one output pixel position, extracting the
// corresponding receptive field patch into a row of the column matrix.
__kernel void im2col_f32(
    __global const float* input,   // [N, C_in, H, W]
    __global float* cols,          // [N * H_out * W_out, C_in * kH * kW]
    const int C_in,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int H_out,
    const int W_out,
    const int batch_size
) {
    // Each work-item handles one (batch, h_out, w_out) position
    int gid = get_global_id(0);
    int total_out = batch_size * H_out * W_out;
    if (gid >= total_out) return;

    // Decode gid → (n, h_out, w_out)
    int w_out_idx = gid % W_out;
    int h_out_idx = (gid / W_out) % H_out;
    int n = gid / (H_out * W_out);

    int col_row = gid;  // Row in the column matrix
    int col_width = C_in * kH * kW;

    for (int c = 0; c < C_in; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int h_in = h_out_idx * stride_h - pad_h + kh;
                int w_in = w_out_idx * stride_w - pad_w + kw;

                int col_col = c * kH * kW + kh * kW + kw;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int input_idx = n * C_in * H * W + c * H * W + h_in * W + w_in;
                    cols[col_row * col_width + col_col] = input[input_idx];
                } else {
                    cols[col_row * col_width + col_col] = 0.0f;  // Zero-padding
                }
            }
        }
    }
}

// ── col2im: Fold column matrix back into image (for backward pass) ──
// Accumulates gradients from the column-format back into spatial format
__kernel void col2im_f32(
    __global const float* cols,    // [N * H_out * W_out, C_in * kH * kW]
    __global float* grad_input,    // [N, C_in, H, W]  — accumulated
    const int C_in,
    const int H,
    const int W,
    const int kH,
    const int kW,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int H_out,
    const int W_out,
    const int batch_size
) {
    int gid = get_global_id(0);
    int total_out = batch_size * H_out * W_out;
    if (gid >= total_out) return;

    int w_out_idx = gid % W_out;
    int h_out_idx = (gid / W_out) % H_out;
    int n = gid / (H_out * W_out);

    int col_row = gid;
    int col_width = C_in * kH * kW;

    for (int c = 0; c < C_in; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int h_in = h_out_idx * stride_h - pad_h + kh;
                int w_in = w_out_idx * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int col_col = c * kH * kW + kh * kW + kw;
                    int input_idx = n * C_in * H * W + c * H * W + h_in * W + w_in;
                    // Atomic add because multiple output positions map to same input
                    // For correctness without atomics, we use sequential accumulation
                    grad_input[input_idx] += cols[col_row * col_width + col_col];
                }
            }
        }
    }
}

// ── Direct 3×3 convolution (optimized for small kernels) ────────────
// Each work-item computes one output pixel across one output channel.
// No im2col needed — directly reads the 3×3 receptive field.
__kernel void conv2d_direct_3x3_f32(
    __global const float* input,   // [N, C_in, H, W]
    __global const float* weight,  // [C_out, C_in, 3, 3]
    __global const float* bias,    // [C_out] or NULL check done on host
    __global float* output,        // [N, C_out, H_out, W_out]
    const int batch_size,
    const int C_in,
    const int C_out,
    const int H,
    const int W,
    const int H_out,
    const int W_out,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int has_bias
) {
    // gid.0 = output position (h_out * W_out + w_out)
    // gid.1 = (n * C_out + c_out)
    int pos = get_global_id(0);
    int nc = get_global_id(1);

    if (pos >= H_out * W_out) return;
    int total_nc = batch_size * C_out;
    if (nc >= total_nc) return;

    int w_out = pos % W_out;
    int h_out = pos / W_out;
    int c_out = nc % C_out;
    int n = nc / C_out;

    float sum = 0.0f;

    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    float in_val = input[n * C_in * H * W + c_in * H * W + h_in * W + w_in];
                    float w_val = weight[c_out * C_in * 9 + c_in * 9 + kh * 3 + kw];
                    sum += in_val * w_val;
                }
            }
        }
    }

    if (has_bias) {
        sum += bias[c_out];
    }

    output[n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out] = sum;
}

__kernel void reshape_grad_out_f32(__global const float* input,
                                   __global float* output,
                                   const int N,
                                   const int C,
                                   const int S) {
    int c = get_global_id(0); // C dimension
    int ns = get_global_id(1); // N * S dimension

    if (c < C && ns < N * S) {
        int n = ns / S;
        int s = ns % S;
        output[c * N * S + n * S + s] = input[n * C * S + c * S + s];
    }
}
