
// server.js
const express = require('express');
const multer  = require('multer');
const path = require('path');
const fs = require('fs');
const { execFile } = require('child_process');

const app = express();
const PORT = 5000;

// Ensure uploads directory exists
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

app.use(express.json());

// Multer config: accept only Python files
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadsDir),
  filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`)
});
const fileFilter = (req, file, cb) => {
  if (path.extname(file.originalname) !== '.py') {
    return cb(new Error('Only Python files are allowed'), false);
  }
  cb(null, true);
};
const upload = multer({ storage, fileFilter });

// Built-in AI: chooses a random valid column
// (Assumes board is a nested list with top row at index 0; if board[0][col] !== 0 then that column is full)
app.post('/api/move', (req, res) => {
  const { board, aiPlayer } = req.body;
  if (!board || !Array.isArray(board)) {
    return res.status(400).json({ error: 'Invalid board state' });
  }
  const numCols = board[0].length;
  const validColumns = [];
  for (let col = 0; col < numCols; col++) {
    if (board[0][col] === 0) validColumns.push(col);
  }
  if (!validColumns.length)
    return res.status(400).json({ error: 'No valid moves available' });
  
  const chosenColumn = validColumns[Math.floor(Math.random() * validColumns.length)];
  res.json({ column: chosenColumn });
});

// Upload a custom Python AI model
app.post('/api/upload', upload.single('model'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  res.json({ modelId: req.file.filename });
});

// Execute a user-uploaded Python AI model
app.post('/api/ai-move/:modelId', (req, res) => {
  const { modelId } = req.params;
  const { board, aiPlayer } = req.body;
  if (!board || !Array.isArray(board)) {
    return res.status(400).json({ error: 'Invalid board state' });
  }
  const modelPath = path.join(uploadsDir, modelId);
  if (!fs.existsSync(modelPath)) {
    return res.status(404).json({ error: 'Model not found' });
  }
  // Pass board as a JSON string. Note: the custom AI should convert it to a numpy array.
  const boardArg = JSON.stringify(board);
  execFile('python', [modelPath, '--board', boardArg, '--player', aiPlayer.toString()], { timeout: 5000 }, (error, stdout, stderr) => {
    if (error) {
      console.error('Error executing Python AI:', error);
      return res.status(500).json({ error: 'Error executing AI model' });
    }
    const output = stdout.trim();
    const chosenColumn = parseInt(output, 10);
    if (isNaN(chosenColumn)) {
      return res.status(500).json({ error: 'Invalid move returned by AI model' });
    }
    res.json({ column: chosenColumn });
  });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

