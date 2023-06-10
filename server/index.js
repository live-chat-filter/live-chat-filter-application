const OpenAI = require("openai");
const { Configuration, OpenAIApi } = OpenAI;

const express = require("express");
const socketio = require("socket.io");
const http = require("http");

const cors = require("cors");
const router = require("./router");
const { addUser, removeUser, getUser, getUsersInRoom } = require("./users");

const { spawn } = require("child_process");
const pythonProcess = spawn("python", ["compm.py"]);
process.stdin.setEncoding("utf8");
process.stdout.setEncoding("utf8");

const PORT = process.env.PORT || 5000;

const configuration = new Configuration({
  organization: "org-VSrAleue66GAKYoP0xrFLcXB",
  apiKey: "",
});
const openai = new OpenAIApi(configuration);

const app = express();
const server = http.createServer(app);
const io = socketio(server);
app.use(cors());
app.use(router);

let dialogue = 0;
let userCount = 0;

io.on("connection", (socket) => {
  console.log("새로운 유저가 접속했습니다.");
  userCount++;
  console.log("현재 접속 중인 유저 수:", userCount);
  console.log("현재 dialogue:", dialogue);
  socket.on("join", ({ name, room }, callback) => {
    const { error, user } = addUser({ id: socket.id, name, room });
    if (error) return callback({ error: "에러가 발생했습니다." });

    socket.join(user.room);
    io.to(user.room).emit("roomData", {
      room: user.room,
      users: getUsersInRoom(user.room),
    });

    socket.emit("message", {
      user: "admin",
      text: `${user.name}, ${user.room}에 오신 것을 환영합니다.`,
    });

    callback();
  });

  socket.on("sendMessage", async (message, callback) => {
    try {
      console.log("message: ", message); // 입력된 message

      // 새로운 pythonProcess 생성
      const pythonProcess = spawn("python", ["compm.py"]);
      pythonProcess.stdin.setEncoding("utf8");
      pythonProcess.stdout.setEncoding("utf8");

      const data = {
        speaker: `${socket.id}`,
        origin_text: `${message}`,
        types: "neutral",
        dialogue: dialogue,
      };

      const jsonData = JSON.stringify(data);
      pythonProcess.stdin.write(jsonData);
      pythonProcess.stdin.end();

      let pythonOutput = ""; // Python의 출력 데이터를 저장할 변수

      pythonProcess.stdout.on("data", (data) => {
        pythonOutput += data; // 데이터를 pythonOutput에 추가로 저장
      });

      pythonProcess.stdout.on("end", async () => {
        console.log(`Python 출력: ${pythonOutput}`);

        if (pythonOutput.trim() === "hate") {
          // 순화 - chat gpt
          try {
            const response = await openai.createCompletion({
              model: "text-davinci-003",
              prompt: `"${message}"를 바른 말로 바꿔줘`,
              max_tokens: 4000,
              temperature: 0,
            });
            // 응답 처리
            message = response.data.choices[0].text;
            console.log("순화된 표현: ", message);
          } catch (error) {
            console.error("오류 발생:", error);
          }
        }

        const user = getUser(socket.id); // 수정: socket.id를 파라미터로 전달

        io.to(user.room).emit("message", {
          user: user.name,
          text: message,
        });
        callback();
      });
    } catch (error) {
      console.error("오류 발생:", error);
    }
  });
  socket.on("disconnect", () => {
    const user = removeUser(socket.id);
    if (user) {
      io.to(user.room).emit("message", {
        user: "admin",
        text: `${user.name}님이 퇴장하셨습니다.`,
      });
      io.to(user.room).emit("roomData", {
        room: user.room,
        users: getUsersInRoom(user.room),
      });
    }
    console.log("유저가 나갔습니다.");
    userCount--;
    console.log("현재 접속 중인 유저 수:", userCount);
  });
});

server.listen(PORT, () => console.log(`서버가 ${PORT} 에서 시작되었어요`));

function increaseDialogue() {
  dialogue++;
  console.log("현재 dialogue:", dialogue);
}

setInterval(() => {
  if (userCount >= 2) {
    increaseDialogue();
  }
}, 10 * 60 * 1000); // 10분마다 dialogue +1
