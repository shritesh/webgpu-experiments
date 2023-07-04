export async function run () {
  const adapter = await navigator.gpu?.requestAdapter()
  const device = await adapter?.requestDevice()

  if (!device) {
    throw Error('WebGPU not available')
  }

  const module = device.createShaderModule({
    code: `
          override n: u32;

          @group(0) @binding(0) var<storage, read> a: array<i32>;
          @group(0) @binding(1) var<storage, read> b: array<i32>;
          @group(0) @binding(2) var<storage, read_write> c: array<i32>;
  
          @compute @workgroup_size(n)
          fn add(@builtin(local_invocation_id) id: vec3<u32>) {
              let i = id.x;
              c[i] = a[i] + b[i];
          }
          `
  })

  const n = 10

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module, entryPoint: 'add', constants: { n }
    }
  })

  const byteLength = Int32Array.BYTES_PER_ELEMENT * n
  const a = new Int32Array(n)
  const b = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    a[i] = -i
    b[i] = i * i
  }

  const aBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  device.queue.writeBuffer(aBuffer, 0, a)

  const bBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  device.queue.writeBuffer(bBuffer, 0, b)

  const cBuffer = device.createBuffer({
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
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: cBuffer } }
    ]
  })

  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(1)
  pass.end()

  encoder.copyBufferToBuffer(cBuffer, 0, resultBuffer, 0, byteLength)

  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])

  await resultBuffer.mapAsync(GPUMapMode.READ)
  const c = new Int32Array(resultBuffer.getMappedRange().slice())
  resultBuffer.unmap()

  let text = ''
  for (let i = 0; i < n; i++) {
    text += `${a[i]} + ${b[i]} = ${c[i]}\n`
  }

  const container = document.createElement('pre')
  container.innerText = text
  return container
}
