const express = require('express')
const app = express()
const port = process.env.PORT || 3000

app.use(express.static('web/public'))

app.get('/', (req, res) => {
  res.sendFile(__dirname + 'web/public/index.html')
})

app.listen(port, () => {
  console.log(`http://localhost:${port}`)
})