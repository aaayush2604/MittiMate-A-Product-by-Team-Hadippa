import React, { useState, useEffect, useRef } from 'react';
import './ChatBot.css'; 

const ChatBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [userName, setUserName] = useState('');
  const chatboxRef = useRef(null);
  const hasInitialized = useRef(false); 

  useEffect(() => {
    if (messages.length === 0 && !hasInitialized.current) {
      hasInitialized.current = true; 
      const displayWelcomeMessage = () => {
        addMessage("Welcome to mittiMate! What's your name?", 'incoming');
      };
      displayWelcomeMessage();
    }
  }, [messages.length]);

  useEffect(() => {
    if (chatboxRef.current) {
      chatboxRef.current.scrollTop = chatboxRef.current.scrollHeight;
    }
  }, [messages]);

  const addMessage = (text, type) => {
    setMessages(prevMessages => [...prevMessages, { text, type }]);
  };

  const handleSendMessage = () => {
    if (!userInput.trim()) return;

    addMessage(userInput, 'outgoing');

    if (!userName) {
      setUserName(userInput);
      displayFirstQuestion(userInput);
    }

    setUserInput('');
  };

  const displayFirstQuestion = (name) => {
    const question = `${name}, Please choose your preferred language from the options below`;
    const choices = ["English", "Hindi", "Tamil", "Telugu", "Malayalam"];

    addMessage(question, 'incoming');
    setMessages(prevMessages => [
      ...prevMessages,
      { type: 'choices', choices, onSelect: (choice) => handleLanguageChoice(choice) }
    ]);
  };

  const handleLanguageChoice = (choice) => {
    addMessage(choice, 'outgoing');
    displaySecondQuestion(userName);
  };

  const displaySecondQuestion = () => {
    const question = `How can I help you today?`;
    const choices = [
      { text: "Fertilizer recommendation", link: "https://example.com/fertilizer" },
      { text: "Book an appointment for soil profile check", link: "https://example.com/soil-profile" }
    ];

    addMessage(question, 'incoming');
    setMessages(prevMessages => [
      ...prevMessages,
      { type: 'choices', choices, onSelect: (choice) => handleHelpChoice(choice) }
    ]);
  };

  const handleHelpChoice = (choice) => {
    addMessage(choice.text, 'outgoing');
    const responseMessage = `You have selected ${choice.text}. Click [here](${choice.link}) for more information.`;
    addMessage(responseMessage, 'incoming');
  };

  return (
    <>
      <button className="chatbot-toggle" onClick={() => setIsOpen(!isOpen)}>
        <span className="material-symbols-outlined">
          {isOpen ? 'close' : 'mode_comment'}
        </span>
      </button>

      {isOpen && (
        <div className="chatbot">
          <header>
            <h2>ChatBot</h2>
          </header>
          <ul className="chatbox" ref={chatboxRef}>
            {messages.map((message, index) => (
              <li key={index} className={`chat ${message.type}`}>
                {message.type === 'incoming' && (
                  <span className="material-symbols-outlined">chat</span>
                )}
                {message.type === 'choices' ? (
                  <div className="choice-buttons">
                    {message.choices.map((choice, choiceIndex) => (
                      <button
                        key={choiceIndex}
                        className="choice-button"
                        onClick={() => message.onSelect(choice)}
                      >
                        {choice.text || choice}
                      </button>
                    ))}
                  </div>
                ) : (
                  <p>{message.text}</p>
                )}
              </li>
            ))}
          </ul>
          <div className="chat-input">
            <textarea
              placeholder="Enter the message..."
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
            />
            <span className="material-symbols-outlined" onClick={handleSendMessage}>
              send
            </span>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatBot;
