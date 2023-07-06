export async function run () {
  const adapter = await navigator.gpu?.requestAdapter()
  const device = await adapter?.requestDevice()

  if (!device) {
    throw Error('WebGPU not available')
  }

  const request = await fetch('./sherlock.txt')
  let buffer = await request.arrayBuffer()

  // Pad right with 0 to make it a multiple of 4 bytes = 32 bits
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

        var<workgroup> temp: array<atomic<u32>, 256>;

        @compute @workgroup_size(256)
        fn histogram(@builtin(local_invocation_id) iid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) dsize: vec3<u32>) {
          var i = iid.x + wid.x * 256;
          let offset = 256 * dsize.x;

          while (i < arrayLength(&data)) {
            let bytes = data[i];
            atomicAdd(&temp[bytes & 0xff], 1);
            atomicAdd(&temp[(bytes & 0xff00) >> 8], 1);
            atomicAdd(&temp[(bytes & 0xff0000) >> 16], 1);
            atomicAdd(&temp[(bytes & 0xff000000) >> 24], 1);

            i += offset;
          }

          workgroupBarrier();

          atomicAdd(&output[iid.x], atomicLoad(&temp[iid.x]));
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
  pass.dispatchWorkgroups(32)
  pass.end()

  encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, 256 * 4)

  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])

  await resultBuffer.mapAsync(GPUMapMode.READ)
  const histo = new Uint32Array(resultBuffer.getMappedRange())

  const counts = {}
  // skip 0
  for (let i = 1; i < 256; i++) {
    if (histo[i] !== 0) {
      counts[String.fromCharCode(i)] = histo[i]
    }
  }
  resultBuffer.unmap()

  const el = document.createElement('pre')
  el.innerText = JSON.stringify(counts, null, 2)
  return el
}
