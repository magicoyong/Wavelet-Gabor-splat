__global__ void rasterize_backward_sum_kernel(
    const dim3 tile_bounds,                   // tile 的数量, 类型 dim3 (x: tile列数, y: tile行数, z=1)
    const dim3 img_size,                      // 图像尺寸, 类型 dim3 (x: width, y: height, z=1)
    const int32_t* __restrict__ gaussian_ids_sorted, // 排序后的 Gaussian 索引, 类型 int32_t[N]
    const int2* __restrict__ tile_bins,      // 每个 tile 包含的 gaussian 范围, 类型 int2[num_tiles]，x=start, y=end
    const float2* __restrict__ xys,          // 每个 gaussian 的中心坐标, 类型 float2[N] (x, y)
    const float3* __restrict__ conics,       // 每个 gaussian 的椭圆参数, 类型 float3[N] (a, b, c)，椭圆矩阵 [[a, b], [b, c]]
    const float3* __restrict__ rgbs,         // 每个 gaussian 的颜色, 类型 float3[N] (r, g, b)
    const float* __restrict__ opacities,     // 每个 gaussian 的透明度, 类型 float[N]
    const float3& __restrict__ background,   // 背景颜色, 类型 float3 (r, g, b)
    const float* __restrict__ final_Ts,      // 每个像素最终累积透射率 T, 类型 float[H*W]
    const int* __restrict__ final_index,     // 每个像素最后一个贡献的 gaussian 索引, 类型 int[H*W]
    const float3* __restrict__ v_output,     // 每个像素对输出梯度的贡献, 类型 float3[H*W]
    const float* __restrict__ v_output_alpha,// 每个像素 alpha 对输出梯度的贡献, 类型 float[H*W] (未使用)
    float2* __restrict__ v_xy,               // 每个 gaussian 的 xy 梯度, 类型 float2[N] (dx, dy)
    float3* __restrict__ v_conic,            // 每个 gaussian 的 conic 梯度, 类型 float3[N] (a, b, c)
    float3* __restrict__ v_rgb,              // 每个 gaussian 的 rgb 梯度, 类型 float3[N] (r, g, b)
    float* __restrict__ v_opacity             // 每个 gaussian 的 opacity 梯度, 类型 float[N]
) {
    // 获取当前 CUDA block 对象
    auto block = cg::this_thread_block();

    // 计算当前线程所在 tile 的 ID (行优先)
    int32_t tile_id = block.group_index().y * tile_bounds.x + block.group_index().x;

    // 当前线程在图像坐标中的 i,j
    unsigned i = block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j = block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j; // 当前像素 x 坐标
    const float py = (float)i; // 当前像素 y 坐标

    // 当前像素在一维数组中的索引
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // 当前线程是否在图像范围内 (越界线程不参与 rasterization)
    const bool inside = (i < img_size.y && j < img_size.x);

    // 当前像素最后贡献 gaussian 的索引
    const int bin_final = inside ? final_index[pix_id] : 0;

    // 当前 tile 的 gaussian 范围 (start, end)
    const int2 range = tile_bins[tile_id];

    // 当前 tile 内需要处理的 batch 数量
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 共享内存，用于存储当前 batch 的数据
    __shared__ int32_t id_batch[BLOCK_SIZE];        // batch gaussian 的 ID, int32
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];// batch gaussian 的 xy + opacity, float3(x, y, opacity)
    __shared__ float3 conic_batch[BLOCK_SIZE];     // batch gaussian 的 conic, float3(a, b, c)
    __shared__ float3 rgbs_batch[BLOCK_SIZE];      // batch gaussian 的 rgb, float3(r,g,b)

    // 当前像素的输出梯度 v_out
    const float3 v_out = v_output[pix_id];

    // 当前线程在 block 内的线程编号
    const int tr = block.thread_rank();

    // 将 block 分成 warp，每 warp 内使用 collective reduction
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // 当前 warp 内计算 bin_final 的最大值
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

    // 遍历 tile 内所有 batch
    for (int b = 0; b < num_batches; ++b) {
        block.sync(); // 所有线程同步

        // 当前 batch 的最后一个 gaussian 索引
        const int batch_end = range.y - 1 - BLOCK_SIZE * b;

        // batch 大小
        int batch_size = min(BLOCK_SIZE, batch_end + 1 - range.x);

        // 当前线程要加载的 gaussian 索引
        const int idx = batch_end - tr;

        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx]; // global gaussian ID
            id_batch[tr] = g_id;

            // 从全局内存读取 gaussian 属性
            const float2 xy = xys[g_id];          // gaussian 中心 (x,y)
            const float opac = opacities[g_id];   // gaussian opacity
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};

            conic_batch[tr] = conics[g_id];       // conic a,b,c
            rgbs_batch[tr] = rgbs[g_id];          // rgb
        }

        block.sync(); // 等待所有线程将 batch 数据写入 shared memory

        // 遍历 batch 内每个 gaussian
        for (int t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside; // 是否处理当前像素

            // 如果当前 gaussian 在像素的前面，跳过
            if (batch_end - t > bin_final) {
                valid = 0;
            }

            // 声明局部变量
            float alpha;          // alpha
            float opac;           // opacity
            float2 delta;         // 像素到 gaussian 中心的偏移 (dx, dy)
            float3 conic;         // conic 参数
            float vis;            // Gaussian 可见性 (exp(-sigma))

            if(valid){
                // 读取 batch 数据
                conic = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;

                // 计算像素偏移
                delta = {xy_opac.x - px, xy_opac.y - py};

                // Gaussian 高斯项 sigma = 0.5 * [a dx^2 + 2b dx dy + c dy^2]
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                       conic.z * delta.y * delta.y) +
                              conic.y * delta.x * delta.y;

                // 高斯衰减值
                vis = __expf(-sigma);

                // 当前像素 alpha
                alpha = min(1.f, opac * vis);

                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0; // 不显著的 gaussian 不处理
                }
            }

            // 如果 warp 内所有线程无效，跳过
            if(!warp.any(valid)){
                continue;
            }

            // 局部梯度变量初始化
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;

            if(valid){
                const float fac = alpha; // 颜色梯度缩放
                float v_alpha = 0.f;

                // 当前像素对 gaussian 颜色梯度
                v_rgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};

                const float3 rgb = rgbs_batch[t];

                // alpha 对颜色的梯度累积
                v_alpha += rgb.x * v_out.x + rgb.y * v_out.y + rgb.z * v_out.z;

                // conic 梯度
                const float v_sigma = -opac * vis * v_alpha;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                 0.5f * v_sigma * delta.x * delta.y, 
                                 0.5f * v_sigma * delta.y * delta.y};

                // xy 梯度
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                              v_sigma * (conic.y * delta.x + conic.z * delta.y)};

                // opacity 梯度
                v_opacity_local = vis * v_alpha;
            }

            // warp 内求和
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum(v_opacity_local, warp);

            // warp 内第0个线程写回全局梯度
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];

                // 写回 rgb
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);

                // 写回 conic
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);

                // 写回 xy
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);

                // 写回 opacity
                atomicAdd(v_opacity + g, v_opacity_local);
            }
        }
    }
}
