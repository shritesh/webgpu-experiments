# Notes
- [Out of bounds writes are clamped to the last element](https://surma.dev/things/webgpu/index.html#:~:text=Luckily%2C%20accessing%20an%20array%20is%20safe%2Dguarded%20by%20an%20implicit%20clamp%2C%20so%20every%20write%20past%20the%20end%20of%20the%20array%20will%20end%20up%20writing%20to%20the%20last%20element%20of%20the%20array)
- `uniform` memory has to be 16-byte aligned.
- Only `storage` memory can have runtime-defined array length.
- `Atomic` only supports `u32` and `i32`.
- `Atomic` memory ordering is relaxed, so it can't be used to build a mutex for `f32`.
- Enable `timestamp-query` feature in Chrome with `--disable-dawn-features=disallow_unsafe_apis` command-line flag.

## CUDA to WebGPU

- In most cases, you can use `@builtin(global_invocation_id)` instead of computing the index manually.

- CUDA `thread` = webgpu `invocation`
- CUDA `block` = webgpu `workgroup`
- CUDA `grid` = webgpu `dispatch`

- `<<<numBlocks, threadsPerBlocks>>>` = `<<<dispatchWorkgroups(...), @workgroup_size(...)>>>` = <<<dispatch size, workgroup size>>>

- `threadIdx` = `local_invocation_id` = current invocation id in the workgroup
- `blockDim` = `@workgroup_size(...)` = number/size of invocations in a workgroup
- `blockIdx` = `workgroup_id` = current workgroup id in the dispatch
- `gridDim` = `num_workgroups` = number/size of workgroups dispatched