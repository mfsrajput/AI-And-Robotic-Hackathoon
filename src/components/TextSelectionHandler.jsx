import React, { useState, useEffect, useRef } from 'react';

const TextSelectionHandler = ({ children }) => {
  const [showButton, setShowButton] = useState(false);
  const [buttonPosition, setButtonPosition] = useState({ x: 0, y: 0 });
  const selectionTimeout = useRef(null);

  const handleSelection = () => {
    // Clear any existing timeout
    if (selectionTimeout.current) {
      clearTimeout(selectionTimeout.current);
    }

    const selection = window.getSelection();
    const text = selection.toString().trim();

    if (text && selection.anchorOffset !== selection.focusOffset) {
      // Get the bounding rectangle of the selection
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();

      // Position the button near the selection (above it)
      setButtonPosition({
        x: rect.left + window.scrollX,
        y: rect.top + window.scrollY - 40 // 40px above the selection
      });

      setShowButton(true);
    } else {
      // Hide button after a short delay to prevent flickering
      selectionTimeout.current = setTimeout(() => {
        setShowButton(false);
      }, 150);
    }
  };

  const handleAskAboutText = () => {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    if (selectedText) {
      setShowButton(false);

      // Create a custom event to communicate with the chatbot
      const event = new CustomEvent('selectedTextEvent', {
        detail: { selectedText }
      });
      window.dispatchEvent(event);
    }
  };

  const handleMouseDown = (e) => {
    // Hide the button if clicking elsewhere
    if (!e.target.closest('.ask-about-this-button') &&
        !e.target.closest('.chatbot-toggle') &&
        !e.target.closest('.chatbot-modal')) {
      setShowButton(false);
    }
  };

  useEffect(() => {
    // Add event listeners
    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('mousedown', handleMouseDown);

    // Cleanup
    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('mousedown', handleMouseDown);
      if (selectionTimeout.current) {
        clearTimeout(selectionTimeout.current);
      }
    };
  }, []);

  return (
    <>
      {children}
      {showButton && (
        <button
          className="ask-about-this-button"
          style={{
            position: 'fixed',
            left: `${buttonPosition.x}px`,
            top: `${buttonPosition.y}px`,
            zIndex: 10000,
            background: '#3b82f6',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            padding: '8px 12px',
            fontSize: '14px',
            cursor: 'pointer',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            fontWeight: '500',
          }}
          onClick={handleAskAboutText}
        >
          Ask about this
        </button>
      )}
    </>
  );
};

export default TextSelectionHandler;