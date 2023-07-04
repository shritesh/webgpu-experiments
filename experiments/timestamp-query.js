export async function run () {
  const adapter = await navigator.gpu?.requestAdapter()

  if (!adapter) {
    throw Error('WebGPU unsupported!')
  }

  if (!adapter.features.has('timestamp-query')) {
    throw Error('"timestamp-query" feature not supported')
  }
  const device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] })

  const module = device.createShaderModule({
    code: `
        @compute @workgroup_size(64) fn spin() {
          var total = 0;
          for (var i = 0; i < 10000; i++) {
            total += i;
          }
        }
      `
  })

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'spin' }
  })

  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 })
  const queryBuffer = device.createBuffer({ size: 8 * 2, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })

  const queryReadBuffer = device.createBuffer({ size: 8 * 2, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })

  const encoder = device.createCommandEncoder()
  encoder.writeTimestamp(querySet, 0)

  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.dispatchWorkgroups(1000)
  pass.end()

  encoder.writeTimestamp(querySet, 1)

  encoder.resolveQuerySet(querySet, 0, 2, queryBuffer, 0)
  encoder.copyBufferToBuffer(queryBuffer, 0, queryReadBuffer, 0, 2 * 8)

  const commandBuffer = encoder.finish()
  device.queue.submit([commandBuffer])

  await queryReadBuffer.mapAsync(GPUMapMode.READ)
  const timestamps = new BigUint64Array(queryReadBuffer.getMappedRange()).slice()
  queryReadBuffer.unmap()

  const el = document.createElement('code')
  el.innerText = `Start: ${timestamps[0]}; End: ${timestamps[1]}; Elapsed: ${timestamps[1] - timestamps[0]}`
  return el
}
