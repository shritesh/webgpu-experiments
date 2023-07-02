// TODO: hash navigation
const files = [
  // "CUDA by Example: An Introduction to General-Purpose GPU Programming"
  'add.js',
  'parallelsum.js',
  'juliaset-cpu.js',
  'juliaset-gpu.js'
]

const buttons = document.createElement('div')

for (const file of files) {
  const btn = document.createElement('button')
  btn.innerText = file
  btn.onclick = async function (e) {
    e.preventDefault()
    const module = await import(`./${file}`)
    try {
      let content = document.querySelector('#playground-content')
      if (content) {
        content.remove()
      }

      content = await module.run()
      content.id = 'playground-content'

      document.body.appendChild(content)
    } catch (e) {
      console.error(e)
      alert(e)
    }
  }
  buttons.appendChild(btn)
}

document.body.appendChild(buttons)
