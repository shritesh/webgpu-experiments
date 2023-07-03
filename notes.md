# CUDA to WebGPU

- CUDA `thread` = webgpu `invocation`
- CUDA `block` = webgpu `workgroup`
- CUDA `grid` = webgpu `dispatch`

- `<<<numBlocks, threadsPerBlocks>>>` = `<<<dispatchWorkgroups(...), @workgroup_size(...)>>>` = <<<dispatch size, workgroup size>>>

- `threadIdx` = `local_invocation_id` = current invocation id in the workgroup
- `blockDim` = `@workgroup_size(...)` = number/size of invocations in a workgroup
- `blockIdx` = `workgroup_id` = current workgroup id in the dispatch
- `gridDim` = `num_workgroups` = number/size of workgroups dispatched