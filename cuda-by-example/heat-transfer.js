const dim = 1024

const maxTemp = 1
const minTemp = 0.0001
const speed = 0.25

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
    override speed: f32;

    @group(0) @binding(0) var tex_in: texture_2d<f32>;
    @group(0) @binding(1) var tex_out: texture_storage_2d<r32float, write>;

    @group(0) @binding(2) var<storage, read_write> data: array<u32>;

    @compute @workgroup_size(16, 16)
    fn copy(@builtin(global_invocation_id) id: vec3<u32>) {
      let c = textureLoad(tex_in, id.xy, 0);

      if (c.x != 0) {
        textureStore(tex_out, id.xy, c);
      }
    }

    @compute @workgroup_size(16, 16)
    fn transfer(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) dsize: vec3<u32>) {
      let t = textureLoad(tex_in, vec2(id.x, id.y - 1), 0).x;
      let l = textureLoad(tex_in, vec2(id.x - 1, id.y), 0).x;
      let c = textureLoad(tex_in, vec2(id.x, id.y), 0).x;
      let r = textureLoad(tex_in, vec2(id.x + 1, id.y), 0).x;
      let b = textureLoad(tex_in, vec2(id.x, id.y + 1), 0).x;

      let value = c + speed * (t + b + r + l - 4 * c);
      textureStore(tex_out, id.xy, vec4(value, 0, 0, 0));

      data[id.x + id.y * dsize.x * 16] = pack4x8unorm(vec4f(value, value, value, 1));
    }
    `
  })

  const size = Uint32Array.BYTES_PER_ELEMENT * dim * dim

  const copyPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'copy'
    }
  })

  const transferPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'transfer',
      constants: { speed }
    }
  })

  const textureConst = device.createTexture({
    format: 'r32float',
    size: [dim, dim],
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  })

  const texture0 = device.createTexture({
    format: 'r32float',
    size: [dim, dim],
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  })

  const texture1 = device.createTexture({
    format: 'r32float',
    size: [dim, dim],
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
  })

  const workBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  })

  const resultBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  })
  const copyBindGroup0 = device.createBindGroup({
    layout: copyPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: textureConst.createView() },
      { binding: 1, resource: texture0.createView() }
    ]
  })

  const copyBindGroup1 = device.createBindGroup({
    layout: copyPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: textureConst.createView() },
      { binding: 1, resource: texture1.createView() }
    ]
  })

  const transferBindGroup0 = device.createBindGroup({
    layout: transferPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: texture0.createView() },
      { binding: 1, resource: texture1.createView() },
      { binding: 2, resource: { buffer: workBuffer } }
    ]
  })

  const transferBindGroup1 = device.createBindGroup({
    layout: transferPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: texture1.createView() },
      { binding: 1, resource: texture0.createView() },
      { binding: 2, resource: { buffer: workBuffer } }
    ]

  })

  const data = new Float32Array(dim * dim)
  for (let i = 0; i < dim * dim; i++) {
    const x = i % dim
    const y = i / dim
    if (x > 300 && x < 600 && y > 310 && y < 601) {
      data[i] = maxTemp
    }
  }
  data[dim * 100 + 100] = (maxTemp + minTemp) / 2
  data[dim * 700 + 100] = minTemp
  data[dim * 300 + 300] = minTemp
  data[dim * 200 + 700] = minTemp
  for (let y = 800; y < 900; y++) {
    for (let x = 400; x < 500; x++) {
      data[x + y * dim] = minTemp
    }
  }
  device.queue.writeTexture({ texture: textureConst }, data, { bytesPerRow: Float32Array.BYTES_PER_ELEMENT * dim }, { width: dim, height: dim })

  data.fill(maxTemp)
  device.queue.writeTexture({ texture: texture0 }, data, { bytesPerRow: Float32Array.BYTES_PER_ELEMENT * dim }, { width: dim, height: dim })

  let toggle = false
  async function render () {
    const encoder = device.createCommandEncoder()
    for (let i = 0; i < 90; i++) {
      {
        const pass = encoder.beginComputePass()
        pass.setPipeline(copyPipeline)
        pass.setBindGroup(0, toggle ? copyBindGroup0 : copyBindGroup1)
        pass.dispatchWorkgroups(dim / 16, dim / 16)
        pass.end()
      }
      {
        const pass = encoder.beginComputePass()
        pass.setPipeline(transferPipeline)
        pass.setBindGroup(0, toggle ? transferBindGroup0 : transferBindGroup1)
        pass.dispatchWorkgroups(dim / 16, dim / 16)
        pass.end()
      }
      toggle = !toggle
    }

    encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, size)

    const commandBuffer = encoder.finish()
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
