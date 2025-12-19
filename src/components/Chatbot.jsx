import React, { useState, useRef, useEffect } from 'react';
import '../css/chatbot.css';

const Chatbot = ({ backendUrl = "http://localhost:5000/chat" }) => {
  // Define the exact chat endpoint to avoid path issues
  const CHAT_ENDPOINT = "http://localhost:5000/chat";  // no trailing slash
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, type: 'assistant', content: 'Hello! I\'m your AI assistant for the Physical AI & Humanoid Robotics textbook. How can I help you today?' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Listen for selected text events from TextSelectionHandler
  useEffect(() => {
    const handleSelectedTextEvent = (event) => {
      const selectedText = event.detail.selectedText;
      if (selectedText) {
        // Pre-fill the input with a query about the selected text
        setInputValue(`Explain this: "${selectedText}"`);
        // Open the chat if it's closed
        if (!isOpen) {
          setIsOpen(true);
        }
        // Focus the input after a short delay to ensure the chat is open
        setTimeout(() => {
          inputRef.current?.focus();
        }, 300);
      }
    };

    window.addEventListener('selectedTextEvent', handleSelectedTextEvent);

    return () => {
      window.removeEventListener('selectedTextEvent', handleSelectedTextEvent);
    };
  }, [isOpen]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Check if the current input is based on selected text
    const isBasedOnSelectedText = inputValue.startsWith('Explain this: "');
    let query = inputValue;
    let selectedText = null;

    if (isBasedOnSelectedText) {
      // Extract the selected text from the pre-filled query
      const match = inputValue.match(/Explain this: "(.*)"/);
      if (match && match[1]) {
        selectedText = match[1];
        query = inputValue; // Keep the full query for context
      }
    }

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Increase timeout to 60s to handle Render cold starts
      // Use exact chat endpoint to avoid path issues
      console.log("Sending request to:", CHAT_ENDPOINT); // Force POST method to resolve 405
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 seconds

      const response = await fetch(CHAT_ENDPOINT, {
        method: 'POST',  // Explicitly set POST method to prevent 405 errors
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          selected_text: selectedText
        }),
        signal: controller.signal
      }); // Hardcoded /chat path — fixes POST to root 405

      // Check if response is ok before attempting to stream
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      clearTimeout(timeoutId);

      // Create initial assistant message with "Thinking..." state
      let assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: '',
        sources: []
      };

      setMessages(prev => [...prev, assistantMessage]);

      if (response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            // Keep the last partial line in the buffer
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const chunk = line.slice(5); // Remove 'data: ' prefix
                // Backend now sends plain text — append directly
                assistantMessage.content += chunk;

                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1] = { ...assistantMessage };
                  return newMessages;
                });
              }
            }
          }

          // Process any remaining buffer
          if (buffer.trim() && buffer.startsWith('data: ')) {
            const chunk = buffer.slice(5); // Remove 'data: ' prefix
            // Backend now sends plain text — append directly
            assistantMessage.content += chunk;

            setMessages(prev => {
              const newMessages = [...prev];
              newMessages[newMessages.length - 1] = { ...assistantMessage };
              return newMessages;
            });
          }
        } finally {
          reader.releaseLock();
        }
      } else {
        // Fallback for non-streaming responses
        const data = await response.json();
        const finalMessage = {
          id: Date.now() + 1,
          type: 'assistant',
          content: data.response || 'Sorry, I encountered an error while processing your request.',
          sources: data.sources || []
        };
        setMessages(prev => [...prev.slice(0, -1), finalMessage]);
      }
    } catch (error) {
      console.error("Chat error:", error); // Log error in development

      let errorMessage = 'Sorry, I encountered an error while processing your request. Please try again.';

      // Check if it's a timeout or network error
      if (error.name === 'AbortError') {
        errorMessage = 'Request timed out. This might be due to cold start on the server. Please try again.';
      } else if (error.message.includes('HTTP error')) {
        // Use the backend error message if available
        errorMessage = error.message;
      }

      const errorBotMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: errorMessage,
        retryAvailable: true // Flag to show retry button
      };
      setMessages(prev => [...prev, errorBotMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
    if (!isOpen && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const handleRetry = (messageId) => {
    // Find the failed message and retry the last user message
    const messageIndex = messages.findIndex(msg => msg.id === messageId);
    if (messageIndex > 0) {
      // Get the previous user message to retry
      for (let i = messageIndex - 1; i >= 0; i--) {
        if (messages[i].type === 'user') {
          setInputValue(messages[i].content);
          handleSubmit({ preventDefault: () => {} });
          break;
        }
      }
    }
  };

  return (
    <>
      <button
        className="chatbot-toggle"
        onClick={toggleChat}
        aria-label={isOpen ? "Close chat" : "Open chat"}
      >
        <div className="chatbot-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M21 15A6 6 0 1 1 12 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <path d="M15 6C16.5 4.5 18 3.5 18 3.5M15 6C13.5 4.5 12 3.5 12 3.5M15 6H18M18 3.5V6.5M9 12C9 12.5523 8.55228 13 8 13C7.44772 13 7 12.5523 7 12C7 11.4477 7.44772 11 8 11C8.55228 11 9 11.4477 9 12ZM17 12C17 12.5523 16.5523 13 16 13C15.4477 13 15 12.5523 15 12C15 11.4477 15.4477 11 16 11C16.5523 11 17 11.4477 17 12ZM13 12C13 12.5523 12.5523 13 12 13C11.4477 13 11 12.5523 11 12C11 11.4477 11.4477 11 12 11C12.5523 11 13 11.4477 13 12Z" stroke="currentColor" strokeWidth="2"/>
          </svg>
        </div>
      </button>

      {isOpen && (
        <div className="chatbot-modal">
          <div className="chatbot-window">
            <div className="chatbot-header">
              <h3>Textbook Assistant</h3>
              <button
                className="chatbot-close"
                onClick={toggleChat}
                aria-label="Close chat"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </div>

            <div className="chatbot-messages">
              {messages.map((message) => (
                <div key={message.id} className={`message message-${message.type}`}>
                  <div className="message-content">
                    <p>{message.content}</p>
                    {message.sources && message.sources.length > 0 && (
                      <div className="message-sources">
                        <strong>Sources:</strong>
                        <ul>
                          {message.sources.map((source, index) => (
                            <li key={index}>
                              <a
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                              >
                                {source.title}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {message.retryAvailable && (
                      <button
                        className="retry-button"
                        onClick={() => handleRetry(message.id)}
                      >
                        Retry
                      </button>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="message message-assistant">
                  <div className="message-content">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                    <p>Thinking... (This may take longer due to server cold start)</p>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <form className="chatbot-input-form" onSubmit={handleSubmit}>
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask about the textbook..."
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isLoading}
                aria-label="Send message"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M22 2L11 13M22 2L15 22L11 13M11 13L2 9L22 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </form>
          </div>
        </div>
      )}
    </>
  );
};

export default Chatbot;