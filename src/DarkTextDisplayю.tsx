import React, { useState } from 'react';

// Стили, определённые прямо в компоненте
const stylesDarkTextDisplay = {
  container: {
    backgroundColor: '#1e1e1e',
    color: '#ffffff',
    minHeight: '100vh',
    padding: '20px',
    fontFamily: 'Arial, sans-serif'
  },
  header: {
    marginBottom: '20px'
  },
  inputRow: {
    display: 'flex',
    gap: '10px',
    alignItems: 'center'
  },
  textField: {
    flex: 1,
    padding: '10px',
    backgroundColor: '#2d2d2d',
    color: '#ffffff',
    border: '1px solid #444',
    borderRadius: '4px',
    fontSize: '16px'
  },
  button: {
    padding: '10px 20px',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: 'bold' as const,
    transition: 'all 0.2s ease-in-out',
    transform: 'translateY(0)',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)'
  },
  displayArea: {
    marginTop: '20px',
    padding: '15px',
    backgroundColor: '#2d2d2d',
    borderRadius: '4px',
    borderLeft: '4px solid #4CAF50',
    minHeight: '50px',
  },
  probsList: {
    display: 'flex',
    gap: "5px",
    flexWrap: "wrap",
    marginTop: "40px"
  }
};

const DarkTextDisplay: React.FC = () => {
  const [inputText, setInputText] = useState('ноль');
  const [displayText, setDisplayText] = useState('ноль');
  const [isProcessing, setIsProcessing] = useState(false);

  const handleStart = async () => {
    if (isProcessing) return;

    setIsProcessing(true);
    let currentString = inputText;
    setDisplayText(currentString);

    /*while (currentString.length < 64) {
      try {
        const newToken = await Promise.resolve(createNewToken(currentString));
        currentString += newToken.token;
        setDisplayText(currentString);
        setCurrentProbs(newToken.probs);
        await new Promise(resolve => setTimeout(resolve, 500));
      } catch (error) {
        console.error('Error in createNewToken:', error);
        break;
      }
    }*/

    setIsProcessing(false);
  };

  return (
    <div style={stylesDarkTextDisplay.container}>
      <div style={stylesDarkTextDisplay.header}>
        <div style={stylesDarkTextDisplay.inputRow}>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Введите текст..."
            style={stylesDarkTextDisplay.textField}
          />
          <button
            onClick={handleStart}
            disabled={isProcessing}
            style={{
              ...stylesDarkTextDisplay.button,
              ...(isProcessing ? { opacity: 0.6 } : {})
            }}
          >
            {isProcessing ? 'Обработка...' : 'Старт'}
          </button>
        </div>
      </div>

      {displayText && (
        <div style={stylesDarkTextDisplay.displayArea}>
          {displayText}
          {isProcessing && <div>Длина: {displayText.length}/64</div>}
        </div>
      )}
    </div>
  );
};

export default DarkTextDisplay;