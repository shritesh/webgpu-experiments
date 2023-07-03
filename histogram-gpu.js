export async function run () {
  const adapter = await navigator.gpu?.requestAdapter()
  const device = await adapter?.requestDevice()

  if (!device) {
    throw Error('WebGPU not available')
  }

  const request = await fetch('./test.txt')
  let buffer = await request.arrayBuffer()

  // Pad right with 0 to make it a multiple of 4 bytes = 32 bits
  // We skip counting 0 (null byte)
  const remainder = buffer.byteLength % 4
  if (remainder !== 0) {
    const newBuffer = new ArrayBuffer(buffer.byteLength + 4 - remainder)
    new Uint8Array(newBuffer).set(new Uint8Array(buffer))

    buffer = newBuffer
  }

  const module = device.createShaderModule({
    code: `
        @group(0) @binding(0) var<storage> data: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<atomic<u32>, 256>;

        @compute @workgroup_size(1)
        fn histogram(@builtin(global_invocation_id) id: vec3<u32>) {
          let bytes = data[id.x];

          atomicAdd(&output[bytes & 0xff], 1);
          atomicAdd(&output[(bytes & 0xff00) >> 8], 1);
          atomicAdd(&output[(bytes & 0xff0000) >> 16], 1);
          atomicAdd(&output[(bytes & 0xff000000) >> 24], 1);
        }
        `
  })

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'histogram' }
  })

  const dataBuffer = device.createBuffer({
    size: buffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  device.queue.writeBuffer(dataBuffer, 0, buffer)

  const workBuffer = device.createBuffer({
    size: 256 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  })

  const resultBuffer = device.createBuffer({
    size: 256 * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  })

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: dataBuffer } },
      { binding: 1, resource: { buffer: workBuffer } }
    ]
  })

  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(buffer.byteLength / 4)
  pass.end()

  encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, 256 * 4)

  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])

  await resultBuffer.mapAsync(GPUMapMode.READ)
  const histo = new Uint32Array(resultBuffer.getMappedRange())

  // skip 0
  const histoSum = histo.slice(1).reduce((a, b) => a + b, 0)
  resultBuffer.unmap()

  const el = document.createElement('code')
  el.innerText = `Histogram Sum: ${histoSum}`
  return el
}
