import React, { useState, useEffect } from 'react';
import OriginalLayout from '@theme-original/Layout';
import Chatbot from '@site/src/components/Chatbot';
import TextSelectionHandler from '@site/src/components/TextSelectionHandler';

// Default backend URL - can be overridden by environment or build process
const BACKEND_URL = typeof window !== 'undefined'
  ? window.CHATBOT_BACKEND_URL || (typeof process !== 'undefined' && process.env ? process.env.CHATBOT_BACKEND_URL : null) || "http://localhost:5000"
  : "http://localhost:5000";

export default function Layout(props) {
  const [selectedText, setSelectedText] = useState(null);

  // Listen for custom events from TextSelectionHandler
  useEffect(() => {
    const handleSelectedText = (event) => {
      setSelectedText(event.detail.selectedText);
    };

    window.addEventListener('selectedTextEvent', handleSelectedText);

    return () => {
      window.removeEventListener('selectedTextEvent', handleSelectedText);
    };
  }, []);

  return (
    <TextSelectionHandler>
      <OriginalLayout {...props}>
        {props.children}
        <Chatbot backendUrl={BACKEND_URL} />
      </OriginalLayout>
    </TextSelectionHandler>
  );
}