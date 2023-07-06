export async function run () {
  const request = await fetch('./sherlock.txt')
  const buffer = await request.arrayBuffer()
  const data = new Uint8Array(buffer)

  const histo = new Uint32Array(256)

  for (let i = 0; i < data.byteLength; i++) {
    const c = data[i]
    histo[c] += 1
  }

  const counts = {}
  // skip 0
  for (let i = 1; i < 256; i++) {
    if (histo[i] !== 0) {
      counts[String.fromCharCode(i)] = histo[i]
    }
  }

  const el = document.createElement('pre')
  el.innerText = JSON.stringify(counts, null, 2)
  return el
}
