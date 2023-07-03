export async function run () {
  const request = await fetch('./test.txt')
  const buffer = await request.arrayBuffer()
  const data = new Uint8Array(buffer)

  const histo = new Uint32Array(256)

  for (let i = 0; i < data.byteLength; i++) {
    const c = data[i]
    histo[c] += 1
  }
  // skip '\0', similar to the GPU implementation
  const histoSum = histo.slice(1).reduce((a, b) => a + b, 0)

  const el = document.createElement('code')
  el.innerText = `Histogram Sum: ${histoSum}`
  return el
}
