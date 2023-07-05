const dim = 1024

export async function run () {
  const adapter = await navigator.gpu?.requestAdapter()

  const device = await adapter?.requestDevice()
  if (!device) {
    throw Error('WebGPU not available')
  }

  const canvas = document.createElement('canvas')
  canvas.width = dim
  canvas.height = dim

  const ctx = canvas.getContext('2d')

  const module = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read_write> data: array<u32>;

      var<workgroup> memory: array<array<f32, 16>, 16>;
  
      @compute @workgroup_size(16, 16)
      fn draw(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) iid: vec3<u32>, @builtin(num_workgroups) dsize: vec3<u32>) {
        const pi = 3.1415926535897932f;
        const period = 128f;

        memory[iid.x][iid.y] = (sin(f32(id.x) * 2f * pi / period) + 1f) * (sin(f32(id.y) * 2f * pi / period) + 1f) / 4f;
        workgroupBarrier();

        data[id.x + id.y * dsize.x * 16] = pack4x8unorm(vec4(0f, memory[15 - iid.x][15 - iid.y], 0f, 1.0f));
      }
      `
  })

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'draw' }
  })

  const size = Uint32Array.BYTES_PER_ELEMENT * dim * dim

  const workBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  })

  const resultBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  })

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: workBuffer } }]
  })

  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(dim / 16, dim / 16)
  pass.end()

  encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, size)

  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])

  await resultBuffer.mapAsync(GPUMapMode.READ)
  const result = new Uint8ClampedArray(resultBuffer.getMappedRange())
  ctx.putImageData(new ImageData(result, dim, dim), 0, 0)
  resultBuffer.unmap()

  return canvas
}
