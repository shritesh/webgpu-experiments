export async function run() {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();

    if (!device) {
        throw Error('WebGPU unsupported!');
    }

    const module = device.createShaderModule({
        code: `
        @group(0) @binding(0) var<storage, read_write> data: array<u32, 3>;
  
        @compute @workgroup_size(1) fn add() {
          data[2] = data[0] + data[1];
        }
      `,
    });

    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'add', },
    });

    const dataBuffer = device.createBuffer({
        size: 3 * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const input = new Uint32Array([2, 7]);
    device.queue.writeBuffer(dataBuffer, 0, input);

    const resultReadBuffer = device.createBuffer({
        size: Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: dataBuffer } },
        ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();

    encoder.copyBufferToBuffer(dataBuffer, 2 * Uint32Array.BYTES_PER_ELEMENT, resultReadBuffer, 0, Uint32Array.BYTES_PER_ELEMENT);

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    await resultReadBuffer.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(resultReadBuffer.getMappedRange())[0];
    resultReadBuffer.unmap();

    document.querySelector("main").innerText = `${input[0]} + ${input[1]} = ${result}`
}