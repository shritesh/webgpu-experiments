export async function run () {
  const adapter = await navigator.gpu?.requestAdapter()
  const device = await adapter?.requestDevice()
  if (!device) {
    throw Error('WebGPU not available')
  }

  const canvas = document.createElement('canvas')
  canvas.width = 1000
  canvas.height = 1000

  const format = navigator.gpu.getPreferredCanvasFormat()
  const ctx = canvas.getContext('webgpu')
  ctx.configure({ device, format })

  const module = device.createShaderModule({
    code: `
    @vertex
    fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        const pos = array(
            vec2f( 0.0,  0.5), // top center
            vec2f(-0.5, -0.5), // bottom left
            vec2f( 0.5, -0.5)  // bottom right
        );

        return vec4f(pos[i], 0.0, 1.0);
    }

    @fragment
    fn fs() -> @location(0) vec4f {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
    `
  })

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] }
  })

  const encoder = device.createCommandEncoder()
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: [0.3, 0.3, 0.3, 1.0],
      loadOp: 'clear',
      storeOp: 'store'
    }]
  })
  pass.setPipeline(pipeline)
  pass.draw(3)
  pass.end()

  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])

  return canvas
}
