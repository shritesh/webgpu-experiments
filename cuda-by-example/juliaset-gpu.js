const dim = 1000

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
  const byteLength = dim * dim * Uint8ClampedArray.BYTES_PER_ELEMENT * 4

  const module = device.createShaderModule({
    code: `
      override dim: u32;
      
      @group(0) @binding(0) var<storage, read_write> data: array<u32>;

      @compute @workgroup_size(1)
      fn kernel(@builtin(global_invocation_id) id: vec3<u32>) {
        let value = julia(id.x, id.y);
        let color = vec4(value, 0.0, 0.0, 1.0);

        data[id.x + id.y * dim] = pack4x8unorm(color);
      }

      fn julia(x: u32, y: u32) -> f32 {
        const scale = 1.5;

        let d_half = f32(dim) / 2.0;
        
        let jx = scale * (d_half - f32(x)) / d_half;
        let jy = scale * (d_half - f32(y)) / d_half;

        const c = vec2(-0.8, 0.156);
        var a = vec2(jx, jy);

        for (var i = 0; i < 200; i++) {
            let real = a[0] * a[0] - a[1] * a[1] + c[0];
            let imag = 2 * a[0] * a[1] + c[1];

            a = vec2(real, imag);

            if (real * real + imag * imag > 1000) {
                return f32(i) / 200.0;
            }
        }

        return 1;
      }
     `
  })

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'kernel', constants: { dim } }
  })

  const workBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  })

  const resultBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  })

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: workBuffer } }
    ]
  })

  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(dim, dim)
  pass.end()

  encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, byteLength)

  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])

  await resultBuffer.mapAsync(GPUMapMode.READ)
  const result = new Uint8ClampedArray(resultBuffer.getMappedRange())
  ctx.putImageData(new ImageData(result, dim, dim), 0, 0)
  resultBuffer.unmap()

  return canvas
}
