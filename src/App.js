import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Chat from './Pages/Chat';
import Join from './Pages/Join';

function App() {
  return (
    <>
      {/* <NavBar /> */}
      <Routes>
        <Route path='/chat' element={<Chat />} />
        <Route path='/' element={<Join />} />
      </Routes>
    </>
  );
}

export default App;
