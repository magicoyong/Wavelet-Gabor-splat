# Wavelet Gabor Image

## primary

$$
Gabor = exp(-\sigma)[(1 - \sum_{i=1}^F \omega_i) + \sum_i^{F}\omega_i cos(2\pi(\mathbf{f}_i^T\mathbf{d}))]\\
=exp(-\sigma)(1 - \sum_{i=1}^F \omega_i) + \sum_i^{F}\omega_i  exp(-\sigma)cos(2\pi(\mathbf{f}_i^T\mathbf{d}))
$$





## Gradient Compute

### Gaussian Image


$$
C_k = \sum_{n=1}^N c_n^k o_n exp(-\sigma_n)
$$
where $\sigma_n = \frac{1}{2} \mathbf{d}_n^T \Sigma^{-1} \mathbf{d}_n$ and $\mathbf{d}_n = (\mu_n^x - x, \mu_n^y- y)$

## Gabor Image

$$
C_k = \sum_{n=1}^N c_n^k o_n exp(-\sigma_n)[(1 - \sum_{i=1}^F \omega_i) + \sum_i^{F}\omega_i cos(2\pi(\mathbf{f}_i^T\mathbf{d}_n))]\\
=  \sum_{n=1}^N c_n^k \alpha_n
$$

 

#### Gabor Gradient:

$n$ 代表第$n$个Gabor，$k$ 代表第$k$条颜色通道

##### out gradient

$$
\frac{\partial L}{\partial C_k}
$$

##### $c_i^k$ gradient

$$
\frac{\partial L}{\partial c_n^k} =\frac{\partial L}{\partial C_k} \frac{\partial C_k}{\partial c_n^k}\\
= \frac{\partial L}{\partial C_k} \alpha_n
$$

##### $\alpha_n$ gradient

$$
\frac{\partial L}{\partial \alpha_n} = \frac{\partial L}{\partial C_1} \frac{\partial C_1}{\partial \alpha_n} +  \frac{\partial L}{\partial C_2} \frac{\partial C_2}{\partial \alpha_n} +  \frac{\partial L}{\partial C_3} \frac{\partial C_3}{\partial \alpha_n}\\
=\frac{\partial L}{\partial C_1} c_n^1 + \frac{\partial L}{\partial C_1} c_n^2 + \frac{\partial L}{\partial C_1} c_n^3
$$



##### $\sigma_n$ gradient

$$
\frac{\partial L}{\partial \sigma_n} =  \frac{\partial L}{\partial \alpha_n} \frac{\partial \alpha_n}{\partial \sigma_n} \\
\frac{\partial \alpha_n}{\partial \sigma_n} = - o_n \exp{(-\sigma_n)}[(1 - \sum_{i=1}^F \omega_i) + \sum_i^{F}\omega_i cos(2\pi(\mathbf{f}_i^T\mathbf{d}_n))] = -\alpha_n
$$

##### $\mu_n$ gradient

*Recall:*
$$
\alpha_n = o_n exp(-\sigma_n)[(1 - \sum_{i=1}^F \omega_i) + \sum_i^{F}\omega_i cos(2\pi(\mathbf{f}_i^T\mathbf{d}_n))]\\
=o_n exp(-\sigma_n) Gabor_n\\
Gabor_n = (1 - \sum_{i=1}^F \omega_i) + \sum_i^{F}\omega_i cos(2\pi(\mathbf{f}_i^T\mathbf{d}_n))
$$
*Full gradient:*
$$
\frac{\partial L}{\partial \mu_n} = \frac{\partial L}{\partial \alpha_n}( \frac{\partial \alpha_n}{\partial \sigma_n} \frac{\partial \sigma_n}{\partial \mu_n}+ \frac{\partial \alpha_n}{\partial Gabor_n}\frac{\partial Gabor_n}{\partial \mu_n} )\\
$$
$\frac{\partial \sigma_n}{\partial \mu_n}$: 
$$
\frac{\partial \sigma_n}{\partial \mu_n} = \frac{\partial \sigma_n}{\partial d_n} \frac{\partial d_n}{\partial \mu_n}=\Sigma_n^{-1}d_n\\
$$
$\frac{\partial \alpha_n}{\partial Gabor_n}\frac{\partial Gabor_n}{\partial \mu_n}$
$$
\frac{\partial \alpha_n}{\partial Gabor_n} = o_n exp(-\sigma_n)\\
\frac{\partial Gabor_n}{\partial \mu_n} = \frac{\partial Gabor_n}{\partial d_n} \frac{\partial d_n}{\partial \mu_n}=-\sum_i^F \omega_i 2\pi \mathbf{f}_isin(2\pi(\mathbf{f}_i^T\mathbf{d}_n))
$$


##### $\Sigma_n^{-1} gradient$

*Notation:*
$$
\Sigma_n^{-1} = \begin{bmatrix} a & b \\ b & c\\ \end{bmatrix}\\
\Delta x_n = \mu_n^x - x_n\\
\Delta y_n = \mu_n^y - y_n
$$

$$
\frac{\partial L}{\partial \Sigma_n^{-1}} =  \frac{\partial L}{\partial \alpha_n} \frac{\partial \alpha_n}{\partial \sigma_n}\frac{\partial \sigma_n}{\partial \Sigma_n^{-1}}
$$

*Recall:*
$$
\sigma_n = \frac{1}{2} \mathbf{d}_n^T \Sigma^{-1} \mathbf{d}_n \\
= \frac{1}{2} [\Delta x, \Delta y] \begin{bmatrix} a & b \\ b & c\\ \end{bmatrix}\begin{bmatrix} \Delta x\\ \Delta y\\ \end{bmatrix} \\
=\frac{1}{2} (\Delta x^2 a + \Delta y^2 c + 2\Delta x \Delta y b)
$$


*gradient of $\Sigma_n^{-1}$:*
$$
\frac{\partial \sigma_n}{\partial a} = \frac{1}{2} \Delta x^2 \\
\frac{\partial \sigma_n}{\partial b} =  \Delta x \Delta y \\
\frac{\partial \sigma_n}{\partial c} =\frac{1}{2} \Delta y^2
$$


##### $o_n$ Gradient

$$
\frac{\partial L}{\partial o_n} = \frac{\partial L}{\partial \alpha_n} \frac{\partial \alpha_n}{\partial o_n}\\
=\frac{\partial L}{\partial \alpha_n} exp(-\sigma_n)[(1 - \sum_{i=1}^F \omega_i) + \sum_i^{F}\omega_i cos(2\pi(\mathbf{f}_i^T\mathbf{d}_n))]
$$



##### $\omega_i$ Gradient

$$
\frac{\partial L}{\partial \omega_i} = \frac{\partial L}{\partial \alpha_n} \frac{\partial \alpha_n}{\partial Gabor_n} \frac{\partial Gabor_n}{\partial \omega_i}
$$

$$
\frac{\partial Gabor_n}{\partial \omega_i} = -1 + cos(2\pi(\mathbf{f}_i^T\mathbf{d}_n))
$$

##### $\mathbf{f}_i$ Gradient

$$
\frac{\partial L}{\partial \mathbf{f}_i} = \frac{\partial L}{\partial \alpha_n} \frac{\partial \alpha_n}{\partial Gabor_n} \frac{\partial Gabor_n}{\partial \mathbf{f}_i}
$$

$$
\frac{\partial Gabor_n}{\partial \mathbf{f}_i} = -2\pi \omega_i \mathbf{d}_n sin(2\pi(\mathbf{f}_i^T\mathbf{d}_n))
$$



##  Gaussian Image Code 

### Gaussian Image model

```python
## gaussianimage_cholesky.py
def forward(self):
    self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
    out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit, self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        
     out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
     out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
     return {"render": out_img}
```

#### `project_gaussians_2d`:



| parameter:                   | shape  | meaning   |
| ---------------------------- | ------ | --------- |
| `self.get_xyz`               | [N, 2] | 均值      |
| `self.get_cholesky_elements` | [N, 3] | cholesky  |
| `self.H`                     | int    | 高        |
| `self.W`                     | int    | 宽        |
| `self.tile_bounds`           | [3, 1] | tile size |

```python
self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) 
```



| output          | shape           | meaning                           |
| --------------- | --------------- | --------------------------------- |
| `self.xys`      | [N, 2]          | 投影后的中心位置（image坐标系下） |
| `depths`        | float           | depth                             |
| `self.radii`    | [N]             | Gaussian的屏幕坐标半径            |
| `conics`        | [num_points, 3] | 二次型系数$\Sigma^{-1}_n$         |
| `num_tiles_hit` | [num_points]    | 覆盖的瓦片数量                    |



```c++
// kernel function for projecting each gaussian on device
// each thread processes one gaussian
// 这个函数应该包含的过程：计算2d cov，num_tiles_hit tile_bounds blabla 然后接入rasterize_forward？
__global__ void project_gaussians_2d_forward_kernel(
    const int num_points,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    // Retrieve the 2D Gaussian parameters
    // printf("means2d %d, %.2f %.2f \n", idx, means2d[idx].x, means2d[idx].y);
    // float clamped_x = max(-1.0f, min(1.0f, means2d[idx].x)); // Clamp x between -1 and 1
    // float clamped_y = max(-1.0f, min(1.0f, means2d[idx].y)); // Clamp y between -1 and 1

    float2 center = {0.5f * img_size.x * means2d[idx].x + 0.5f * img_size.x,
                     0.5f * img_size.y * means2d[idx].y + 0.5f * img_size.y};
    // Assuming L is packed row-wise in a 1D array: [l11, l21, l22]
    float l11 = L_elements[idx].x; // scale_x
    float l21 = L_elements[idx].y; // covariance_xy
    float l22 = L_elements[idx].z; // scale_y

    // Construct the 2x2 covariance matrix from L
    // float2x2 Cov2D = make_float2x2(l11*l11, l11*l21,
                                //    l11*l21, l21*l21 + l22*l22);
    float3 cov2d = make_float3(l11*l11, l11*l21, l21*l21 + l22*l22);
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok)
        return; // zero determinant
    // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    conics[idx] = conic;
    xys[idx] = center;
    radii[idx] = (int)radius;
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        // printf("%d point bbox outside of bounds\n", idx);
        return;
    }
    num_tiles_hit[idx] = tile_area;
    // 先给一个固定的depth，为了后面的函数调用方便
    depths[idx] = 0.0f;
}
```

#### `project_gaussians_2d_backward`:

也要改，计算了project过程产生的梯度





#### `rasterize_gaussians_sum`:

*Recall:*

```python
 def forward(self):
        ## TODO: add gabor
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
```

##### input:

| parameter                      | shape   | meaning         |
| ------------------------------ | ------- | --------------- |
| `self.xys`                     | [N, 2]  | 屏幕投影中心    |
| `depths`                       | [N]     | 0.0f            |
| `self.radii`                   | [N]     | 影响半径        |
| `conics`                       | [N, 3]  | $\Sigma_n^{-1}$ |
| `num_tiles_hit`                | [N]     | 覆盖的瓦片数量  |
| `self.get_features`            | [N, 3]  | rgb color       |
| `self._opacity`                | [N]     | 不透明度        |
| `self.H`, `self.W`             | int     | 图像大小        |
| `self.BLOCK_H`, `self.BLOCK_W` | int     | tile大小        |
| `self.background`              | [3]     | 背景颜色        |
| `return_alpha=False`           | boolean |                 |



## Experiment

+ 需要查看训练结果
+ 检查代码
  + 梯度计算
  + 优化

+ DWT

可以试一下用GaussianImage计算出图像（$\mu$，conic，color, opacity），然后加上频率再次优化

对比下Gaussian和Gabor的位置什么的？



