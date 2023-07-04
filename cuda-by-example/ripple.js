const dim = 1024
const sizeofU32 = Uint32Array.BYTES_PER_ELEMENT

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
    override dim: u32;

    @group(0) @binding(0) var<uniform> ticks: u32;
    @group(0) @binding(1) var<storage, read_write> data: array<u32>;

    @compute @workgroup_size(16, 16)
    fn ripple(@builtin(global_invocation_id) id: vec3<u32>) {
      let fx = f32(id.x) - f32(dim)/2;
      let fy = f32(id.y) - f32(dim)/2;
      let d = sqrt(fx * fx + fy * fy);
        
      let grey = cos(d / 10 - f32(ticks) / 7);
        
      data[id.x + id.y * dim] = pack4x8unorm(vec4(grey, grey, grey, 1));
    }
    `
  })

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'ripple',
      constants: { dim }
    }
  })

  const uniformBuffer = device.createBuffer({
    size: sizeofU32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  })

  const workBuffer = device.createBuffer({
    size: sizeofU32 * dim * dim,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  })

  const resultBuffer = device.createBuffer({
    size: sizeofU32 * dim * dim,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  })

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: workBuffer } }
    ]
  })

  const uniform = new Uint32Array(1)

  async function render () {
    uniform[0] += 1

    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(dim / 16, dim / 16)
    pass.end()

    encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, sizeofU32 * dim * dim)

    const commandBuffer = encoder.finish()

    device.queue.writeBuffer(uniformBuffer, 0, uniform)
    device.queue.submit([commandBuffer])

    await resultBuffer.mapAsync(GPUMapMode.READ)
    const result = new Uint8ClampedArray(resultBuffer.getMappedRange())
    ctx.putImageData(new ImageData(result, dim, dim), 0, 0)
    resultBuffer.unmap()

    requestAnimationFrame(render)
  }

  render()

  return canvas
}
