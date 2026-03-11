__global__ void rasterize_forward_sum_gabor(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,  //存储该tile内高斯的范围
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    const float* __restrict__ gabor_freqs_x, //[N, F]
    const float* __restrict__ gabor_freqs_y,
    const float* __restrict__ gabor_weights,
    const int num_freqs,
    // ouput
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img, 
    const float3& __restrict__ background
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];         
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            const float gs_value = __expf(-sigma);

            // 计算 Gabor 调制部分
            // float weights_sum = 0.f;
            float cos_sum = 0.f;

            // 读取 Gabor 参数
            for (int f = 0; f < num_freqs; ++f) {

                //int g_idx = t * num_freqs + f; 
                int32_t g = id_batch[t];
                int g_idx = g * num_freqs + f;

                float fx = gabor_freqs_x[g_idx];
                float fy = gabor_freqs_y[g_idx];
                
                float w = gabor_weights[g_idx];

                // weights_sum += w;
                // theta = 2 * pi * (f^T * x)
                float theta = 2.0f * M_PI * (delta.x * fx + delta.y * fy);
                cos_sum += w * __cosf(theta);
            }

            // Gabor Modulation H
            float H = (float)num_freqs + cos_sum;
            const float alpha = min(1.f, opac * gs_value * H);
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            int32_t g = id_batch[t];
            const float vis = alpha;
            const float3 c = colors[g];
            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            // T = next_T;
            cur_idx = batch_start + t;
        }
        // done = true;
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x; //+ T * background.x;
        final_color.y = pix_out.y; //+ T * background.y;
        final_color.z = pix_out.z; //+ T * background.z;
        out_img[pix_id] = final_color;
    }
}


__global__ void rasterize_backward_sum_gabor_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ gabor_freqs_x,
    const float* __restrict__ gabor_freqs_y,
    const float* __restrict__ gabor_weights,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    int num_freqs,
    // output 
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity,
    
    float* __restrict__ v_weights,
    float* __restrict__ v_freqs_x,
    float* __restrict__ v_freqs_y
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    // float T_final = final_Ts[pix_id];
    // float T = T_final;
    // the contribution from gaussians behind the current one
    // float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float3 rgbs_batch[BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];
    // const float v_out_alpha = v_output_alpha[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - BLOCK_SIZE * b;
        int batch_size = min(BLOCK_SIZE, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        int32_t g_id;
        if (idx >= range.x) {
            g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
            }

        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float2 delta;
            float3 conic;
            float gs_value;
            // 计算 Gabor 调制部分
            float weights_sum = 0.f;
            float cos_sum = 0.f;
            float sin_sum_x = 0.f;
            float sin_sum_y = 0.f;
            float H;
            float3 xy_opac;

            if(valid){
                conic = conic_batch[t];
                xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                gs_value =  __expf(-sigma);

                // 读取 Gabor 参数
                for (int f = 0; f < num_freqs; ++f) {
                    
                    //int g_idx = g_id * num_freqs + f; 
                    int32_t g = id_batch[t];
                    int g_idx = g * num_freqs + f;

                    float fx = gabor_freqs_x[g_idx];
                    float fy = gabor_freqs_y[g_idx];
                
                    float w = gabor_weights[g_idx];

                    weights_sum += w;
                    // theta = 2 * pi * (f^T * x)
                    float theta = 2.0f * M_PI * (delta.x * fx + delta.y * fy);
                    cos_sum += w * __cosf(theta);
                    sin_sum_x -= 2.0f * M_PI * w * fx * __sinf(theta);
                    sin_sum_y -= 2.0f * M_PI * w * fy * __sinf(theta);
                }

                    // Gabor Modulation H
                    H = num_freqs + cos_sum;
                    alpha = min(1.f, opac * gs_value * H);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid)){
                continue;
            }
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            float v_alpha = 0.f;

                        
            //initialize everything to 0, only set if the lane is valid
            if(valid){
                // compute the current T for this gaussian
                // float ra = 1.f / (1.f - alpha);
                // T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha;
                v_rgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};

                const float3 rgb = rgbs_batch[t];
                // contribution from this pixel
                v_alpha += rgb.x * v_out.x;
                v_alpha += rgb.y * v_out.y;
                v_alpha += rgb.z * v_out.z;

                const float v_sigma = -alpha * v_alpha;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                         v_sigma * delta.x * delta.y, 
                                        0.5f * v_sigma * delta.y * delta.y};
                        
                // v_xy_local = {v_sigma * (conic.x * xy_opac.x + conic.y * xy_opac.y) + v_alpha * opac * gs_value * sin_sum_x, 
                //                     v_sigma * (conic.y * xy_opac.x + conic.z * xy_opac.y) + v_alpha * opac * gs_value * sin_sum_y};
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y) + v_alpha * opac * gs_value * sin_sum_x, 
                                    v_sigma * (conic.y * delta.x + conic.z * delta.y) + v_alpha * opac * gs_value * sin_sum_y};
                v_opacity_local = v_alpha * gs_value * H;
            }

            for(int f = 0; f < num_freqs; ++f){
                float v_weight_local = 0.f;
                float v_freq_x_local = 0.f;
                float v_freq_y_local = 0.f;

                int32_t g = id_batch[t];
                
                if (valid) {
                    int g_idx = g * num_freqs + f;

                    float fx = gabor_freqs_x[g_idx];
                    float fy = gabor_freqs_y[g_idx]; 
                    float w = gabor_weights[g_idx]; 

                    v_weight_local = v_alpha * opac * gs_value * ( __cosf(2.0f * M_PI * (delta.x * fx+ delta.y * fy)));
                    v_freq_x_local = - v_alpha * opac * gs_value * 2.0f * M_PI * w * delta.x * __sinf(2.0f * M_PI * (delta.x * fx + delta.y * fy));
                    v_freq_y_local = - v_alpha * opac * gs_value * 2.0f * M_PI * w * delta.y * __sinf(2.0f * M_PI * (delta.x * fx + delta.y * fy));
                }

                // ===== warp reduce =====
                warpSum(v_weight_local, warp);
                warpSum(v_freq_x_local, warp);
                warpSum(v_freq_y_local, warp);

                if (warp.thread_rank() == 0) {
                    atomicAdd(v_weights + g * num_freqs + f, v_weight_local);
                    atomicAdd(v_freqs_x + g * num_freqs + f, v_freq_x_local);
                    atomicAdd(v_freqs_y + g * num_freqs + f, v_freq_y_local);
                }
            }

            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
                atomicAdd(v_opacity + g, v_opacity_local);
            }
        }
    }
}