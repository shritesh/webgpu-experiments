const experiments = {
  experiments: ['compute-render.js'],
  'cuda-by-example': [
    'add.js',
    'parallelsum-workgroups.js',
    'juliaset-cpu.js',
    'juliaset-gpu.js',
    'parallelsum-invocations.js',
    'parallelsum-both.js',
    'ripple.js',
    'dotproduct.js',
    'histogram-cpu.js',
    'histogram-gpu.js'
  ],
  fundamentals: ['render.js']
}

const details = document.createElement('details')
details.hidden = true

const summary = document.createElement('summary')
details.appendChild(summary)

const pre = document.createElement('pre')
const code = document.createElement('code')

pre.appendChild(code)
details.appendChild(pre)

for (const dir in experiments) {
  const buttons = document.createElement('div')
  buttons.innerText = dir
  for (const sample of experiments[dir]) {
    const file = `./${dir}/${sample}`
    const btn = document.createElement('button')
    btn.innerText = sample
    btn.onclick = async function (e) {
      e.preventDefault()
      const module = await import(file)
      const req = await fetch(file)
      const src = await req.text()
      summary.innerText = file
      code.innerText = src
      details.hidden = false

      try {
        let content = document.querySelector('#experiment')
        if (content) {
          content.remove()
        }

        content = await module.run()
        content.id = 'experiment'

        document.body.appendChild(content)
      } catch (e) {
        console.error(e)
        alert(e)
      }
    }
    buttons.appendChild(btn)
    document.body.appendChild(buttons)
  }
}

document.body.appendChild(details)
