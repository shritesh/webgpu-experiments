// TODO: hash navigation
const files = [
  // "CUDA by Example: An Introduction to General-Purpose GPU Programming"
  'add.js',
  'parallelsum.js',
  'juliaset-cpu.js',
  'juliaset-gpu.js',
  'histogram-cpu.js',
  'histogram-gpu.js'
]

const buttons = document.createElement('div')
const details = document.createElement('details')
details.hidden = true

const summary = document.createElement('summary')
details.appendChild(summary)

const pre = document.createElement('pre')
const code = document.createElement('code')

pre.appendChild(code)
details.appendChild(pre)

for (const file of files) {
  const btn = document.createElement('button')
  btn.innerText = file
  btn.onclick = async function (e) {
    e.preventDefault()
    const module = await import(`./${file}`)
    const req = await fetch(`./${file}`)
    const src = await req.text()
    summary.innerText = file
    code.innerText = src
    details.hidden = false

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
document.body.appendChild(details)
