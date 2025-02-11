
// my_ai_model.js
module.exports.getMove = function(board, player) {
  // Find valid columns (top row cell is 0 if column is not full)
  let validColumns = [];
  for (let col = 0; col < board[0].length; col++) {
    if (board[0][col] === 0) validColumns.push(col);
  }
  
  // Return a random valid column
  return validColumns[Math.floor(Math.random() * validColumns.length)];
};
