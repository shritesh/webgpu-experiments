const dim = 1000

export async function run () {
  const canvas = document.createElement('canvas')
  canvas.width = dim
  canvas.height = dim

  const ctx = canvas.getContext('2d')

  const imageData = new ImageData(dim, dim)
  const data = imageData.data

  for (let y = 0; y < dim; y++) {
    for (let x = 0; x < dim; x++) {
      const offset = x + y * dim

      const value = julia(x, y)
      data[offset * 4 + 0] = 255 * value
      data[offset * 4 + 3] = 255
    }
  }

  ctx.putImageData(imageData, 0, 0)

  return canvas
}

function julia (x, y) {
  const scale = 1.5

  const jx = scale * (dim / 2 - x) / (dim / 2)
  const jy = scale * (dim / 2 - y) / (dim / 2)

  // Complex numbers
  const c = [-0.8, 0.156]
  let a = [jx, jy]

  for (let i = 0; i < 200; i++) {
    // a = a * a + c
    const real = a[0] * a[0] - a[1] * a[1]
    const imag = 2 * a[0] * a[1]
    a = [real, imag]

    a[0] += c[0]
    a[1] += c[1]

    const magnitude = Math.sqrt(a[0] * a[0] + a[1] * a[1])

    if (magnitude > 1000) {
      return 0
    }
  }

  return 1
}
