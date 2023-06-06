import React, { useState, useEffect } from 'react';
import queryString from 'query-string';
import io from 'socket.io-client';

import './Chat.css';
import InfoBar from '../components/InfoBar/InfoBar';
import Messages from '../components/Messages/Messages';
import TextContainer from '../components/TextContainer/TextContainer';
import Input from '../components/Input/Input';

const ENDPOINT = 'http://localhost:5000';
let socket;

function Chat() {
  const [name, setName] = useState('');
  const [room, setRoom] = useState('');
  const [users, setUsers] = useState('');
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const { name, room } = queryString.parse(window.location.search);
    socket = io(ENDPOINT);
    setName(name);
    setRoom(room);
    socket.emit('join', { name, room }, (err) => {
      if (err) {
        alert(err);
      }
    });
    return () => {
      socket.emit('disconnect');
      socket.off();
    };
  }, [ENDPOINT, window.location.search]);

  useEffect(() => {
    socket.on('message', (message) => {
      setMessages([...messages, message]);
    });
    socket.on('roomData', ({ users }) => {
      setUsers(users);
    });
  }, []);

  const sendMessage = (event) => {
    event.preventDefault();
    if (message) {
      socket.emit('sendMessage', message, () => setMessage(''));
    }
  };

  return (
    <div className='outerContainer'>
      <div className='container'>
        <InfoBar room={room} />
        <Messages messages={messages} name={name} />
        <Input
          message={message}
          setMessage={setMessage}
          sendMessage={sendMessage}
        />
      </div>
      <TextContainer users={users} />
    </div>
  );
}

export default Chat;
