export async function run () {
  const adapter = await navigator.gpu?.requestAdapter()

  const device = await adapter?.requestDevice({ requiredFeatures: ['bgra8unorm-storage'] })
  if (!device) {
    throw Error('WebGPU not available')
  }

  const canvas = document.createElement('canvas')
  canvas.width = 1000
  canvas.height = 1000

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

    @compute @workgroup_size(8, 8)
    fn draw(@builtin(global_invocation_id) id: vec3<u32>) {
      let color = vec2f(id.xy) / vec2f(textureDimensions(texture).xy);
      textureStore(texture, id.xy, vec4(color, 0.0f, 1.0f));
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
  pass.dispatchWorkgroups(1000 / 8, 1000 / 8)
  pass.end()

  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])

  return canvas
}
