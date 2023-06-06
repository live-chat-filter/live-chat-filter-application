import React from 'react';

import onlineIcon from '../../icons/onlineIcon.png';

import './TextContainer.css';

function TextContainer({ users }) {
  return (
    <div className='textContainer'>
      <div>
        <h1>
          실시간 채팅 프로그램{' '}
          <span role='img' aria-label='emoji'>
            💬
          </span>
        </h1>
      </div>
      {users ? (
        <div>
          <h1>현재 채팅중인 사람들 : </h1>
          <div className='activeContainer'>
            <h2>
              {users.map(({ name }) => (
                <div key={name} className='activeItem'>
                  {name}
                  <img alt='Online Icon' src={onlineIcon} />
                </div>
              ))}
            </h2>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default TextContainer;
