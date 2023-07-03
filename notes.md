# CUDA to WebGPU

- CUDA `thread` = webgpu `invocation`
- CUDA `block` = webgpu `workgroup`
- CUDA `grid` = webgpu `dispatch`

- `<<<numBlocks, threadsPerBlocks>>>` = `<<<dispatchWorkgroups(...), @workgroup_size(...)>>>` = <<<dispatch size, workgroup size>>>

- `blockIdx` = `workgroup_id` = workgroup id
- `gridDim` = `num_workgroups` = dispatch size
- `threadIdx` = `local_invocation_id` = local invocation id
- `blockDim` = `@workgroup_size(...)` = grid size