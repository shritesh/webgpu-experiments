const dim = 1024
const count = 20

export async function run () {
  const adapter = await navigator.gpu?.requestAdapter()

  const requiredFeatures = []
  const hasTimestampQuery = adapter.features.has('timestamp-query')
  if (hasTimestampQuery) {
    requiredFeatures.push('timestamp-query')
  }

  const device = await adapter?.requestDevice({ requiredFeatures })
  if (!device) {
    throw Error('WebGPU not available')
  }

  const canvas = document.createElement('canvas')
  canvas.width = dim
  canvas.height = dim

  const ctx = canvas.getContext('2d')
  const byteLength = dim * dim * Uint32Array.BYTES_PER_ELEMENT

  const module = device.createShaderModule({
    code: `
    override dim: u32;
    const inf = 2e10f;

    struct Sphere {
        r: f32,
        g: f32,
        b: f32,
        radius: f32,
        x: f32,
        y: f32,
        z: f32,
    }

    @group(0) @binding(0) var<storage> spheres: array<Sphere>;
    @group(0) @binding(1) var<storage, read_write> data: array<u32>;

    fn hit(sphere: Sphere, o: vec2<f32>, n: ptr<function, f32>) -> f32 {
        let dx = o.x - sphere.x;
        let dy = o.y - sphere.y;

        let r2 = sphere.radius * sphere.radius;

        if (dx * dx + dy * dy < r2) {
            let dz = sqrt(r2 - dx * dx - dy * dy);
            *n = dz / sqrt(r2);
            return dz + sphere.z;
        }

        return -inf;
    }

    @compute @workgroup_size(16, 16)
    fn raytrace(@builtin(global_invocation_id) id: vec3<u32>) {
        let o = vec2f(f32(id.x) - f32(dim / 2), f32(id.y) - f32(dim / 2));
        var color = vec3f(0, 0, 0);

        var maxz = -inf;
        for(var i = 0u; i < arrayLength(&spheres); i++) {
          var n = 0.0;
          let t = hit(spheres[i], o, &n);
          if (t > maxz) {
            color.r = spheres[i].r * n;
            color.g = spheres[i].g * n;
            color.b = spheres[i].b * n;
            maxz = t;
          }
        }

        data[id.x + id.y * dim] = pack4x8unorm(vec4f(color, 1.0));
      }
    `
  })

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'raytrace', constants: { dim } }
  })

  // JS doesn't have a seedable rng.
  // https://observablehq.com/@fil/linear-congruential-generator
  const LCG = function (seed) {
    const a = 1664525
    const c = 1013904223
    const m = 4294967296 // 2^32
    let s = Math.abs(a * +seed) || 1
    return () => (s = (a * s + c) % m) / m
  }
  const rand = LCG()
  const spheres = new Float32Array(count * 7) // 7 fields
  for (let i = 0; i < count; i++) {
    spheres[i * 7 + 0] = rand() // r
    spheres[i * 7 + 1] = rand() // g
    spheres[i * 7 + 2] = rand() // b
    spheres[i * 7 + 3] = 100 * rand() + 20 // radius
    spheres[i * 7 + 4] = 1000 * rand() - 500 // x
    spheres[i * 7 + 5] = 1000 * rand() - 500 // y
    spheres[i * 7 + 6] = 1000 * rand() - 500 // z
  }

  const sphereBuffer = device.createBuffer({
    size: spheres.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true
  })
  const mappedSphereBuffer = new Float32Array(sphereBuffer.getMappedRange())
  mappedSphereBuffer.set(spheres)
  sphereBuffer.unmap()

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
      { binding: 0, resource: { buffer: sphereBuffer } },
      { binding: 1, resource: { buffer: workBuffer } }
    ]
  })

  let querySet, queryBuffer, queryReadBuffer
  if (hasTimestampQuery) {
    querySet = device.createQuerySet({ type: 'timestamp', count: 2 })
    queryBuffer = device.createBuffer({ size: 8 * 2, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    queryReadBuffer = device.createBuffer({ size: 8 * 2, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  }

  const encoder = device.createCommandEncoder()
  if (hasTimestampQuery) {
    encoder.writeTimestamp(querySet, 0)
  }

  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(dim / 16, dim / 16)
  pass.end()

  if (hasTimestampQuery) {
    encoder.writeTimestamp(querySet, 1)
    encoder.resolveQuerySet(querySet, 0, 2, queryBuffer, 0)
    encoder.copyBufferToBuffer(queryBuffer, 0, queryReadBuffer, 0, 2 * 8)
  }

  encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, byteLength)

  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])

  await resultBuffer.mapAsync(GPUMapMode.READ)
  const result = new Uint8ClampedArray(resultBuffer.getMappedRange())
  ctx.putImageData(new ImageData(result, 1024, 1024), 0, 0)
  resultBuffer.unmap()

  const el = document.createElement('div')
  if (hasTimestampQuery) {
    await queryReadBuffer.mapAsync(GPUMapMode.READ)
    const timestamps = new BigUint64Array(queryReadBuffer.getMappedRange()).slice()
    queryReadBuffer.unmap()

    el.innerText = `Render time: ${timestamps[1] - timestamps[0]}\n`
  } else {
    el.innerText = '"timestamp-query" feature unavailable\n'
  }
  el.appendChild(canvas)
  return el
}
