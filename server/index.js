const express = require('express');
const socketio = require('socket.io');
const http = require('http');

const cors = require('cors');
const router = require('./router');

const PORT = process.env.PORT || 5000;

const app = express();
const server = http.createServer(app);
const io = socketio(server);
app.use(cors());
app.use(router);
io.on('connection', (socket) => {
  console.log('새로운 유저가 접속했습니다.');
  socket.on('join', ({ name, room }, callback) => {});
  socket.on('disconnect', () => {
    console.log('유저가 나갔습니다.');
  });
});

server.listen(PORT, () => console.log(`서버가 ${PORT} 에서 시작되었어요`));
