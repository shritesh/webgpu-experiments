const n = 33 * 1024
const wsize = 256
const dsize = Math.min(32, Math.trunc((n + wsize - 1) / wsize))

const sizeofF32 = Float32Array.BYTES_PER_ELEMENT

export async function run () {
  const adapter = await navigator.gpu?.requestAdapter()
  const device = await adapter?.requestDevice()

  if (!device) {
    throw Error('WebGPU not available')
  }

  const module = device.createShaderModule({
    code: `
    override wsize: u32;

    var<workgroup> cache: array<f32, wsize>;
    @group(0) @binding(0) var<storage> a: array<f32>;
    @group(0) @binding(1) var<storage> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> c: array<f32>;

    @compute @workgroup_size(wsize) 
    fn dot(@builtin(local_invocation_id) iid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) dsize: vec3<u32>) {
        let n = arrayLength(&a);

        var id = iid.x + wid.x * wsize;
        let cache_id = iid.x;

        var temp = 0f;
        while (id < n) {
            temp += a[id] * b[id];
            id += wsize * dsize.x;
        }

        cache[cache_id] = temp;
        workgroupBarrier();

        var i = wsize / 2;
        while (i != 0) {
            if (cache_id < i) {
                cache[cache_id] += cache[cache_id + i];
            }
            workgroupBarrier();
            i /= 2;
        }

        c[wid.x] = cache[0];
    }
    `
  })

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'dot', constants: { wsize } }
  })

  const aBuffer = device.createBuffer({
    size: n * sizeofF32,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true
  })
  const a = new Float32Array(aBuffer.getMappedRange())
  for (let i = 0; i < n; i += 1) {
    a[i] = i
  }
  aBuffer.unmap()

  const bBuffer = device.createBuffer({
    size: n * sizeofF32,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true
  })
  const b = new Float32Array(bBuffer.getMappedRange())
  for (let i = 0; i < n; i += 1) {
    b[i] = i * 2
  }
  bBuffer.unmap()

  const partialcBuffer = device.createBuffer({
    size: dsize * sizeofF32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  })

  const resultBuffer = device.createBuffer({
    size: dsize * sizeofF32,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  })

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: partialcBuffer } }
    ]
  })

  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(dsize)
  pass.end()

  encoder.copyBufferToBuffer(partialcBuffer, 0, resultBuffer, 0, sizeofF32 * dsize)

  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])

  await resultBuffer.mapAsync(GPUMapMode.READ)
  const gpuResult = (new Float32Array(resultBuffer.getMappedRange()).reduce((a, b) => a + b, 0))
  resultBuffer.unmap()

  let cpuResult = 0
  for (let i = 0; i < n; i++) {
    cpuResult += i * 2 * i
  }

  const container = document.createElement('pre')
  // "%.6g" formatting like the book
  container.innerText = `GPU: ${gpuResult.toExponential(6)} CPU: ${cpuResult.toExponential(6)}`
  return container
}
