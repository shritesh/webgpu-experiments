const dim = 1024

export async function run () {
  const adapter = await navigator.gpu?.requestAdapter()

  const device = await adapter?.requestDevice({ requiredFeatures: ['bgra8unorm-storage'] })
  if (!device) {
    throw Error('WebGPU not available')
  }

  const canvas = document.createElement('canvas')
  canvas.width = dim
  canvas.height = dim

  const format = navigator.gpu.getPreferredCanvasFormat()
  const ctx = canvas.getContext('webgpu')
  ctx.configure({
    device,
    format,
    usage: GPUTextureUsage.STORAGE_BINDING
  })

  const module = device.createShaderModule({
    code: `
      @group(0) @binding(0) var texture: texture_storage_2d<bgra8unorm, write>;

      var<workgroup> memory: array<array<f32, 16>, 16>;
  
      @compute @workgroup_size(16, 16)
      fn draw(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) iid: vec3<u32>) {
        const pi = 3.1415926535897932f;
        const period = 128f;

        memory[iid.x][iid.y] = (sin(f32(id.x) * 2f * pi / period) + 1f) * (sin(f32(id.y) * 2f * pi / period) + 1f) / 4f;
        workgroupBarrier();

        textureStore(texture, id.xy, vec4(0f, memory[15 - iid.x][15 - iid.y], 0f, 1.0f));
      }
      `
  })

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'draw' }
  })

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: ctx.getCurrentTexture().createView() }]
  })

  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(dim / 16, dim / 16)
  pass.end()

  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])

  return canvas
}
