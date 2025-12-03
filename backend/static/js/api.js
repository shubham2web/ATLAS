// API Module with detailed logging
const API = {
    baseURL: 'http://127.0.0.1:8000', // Backend server port

    async sendMessage(message, mode = 'analytical', conversationHistory = [], signal = null) {
        console.log('=== API Call (ATLAS v4.1 Verdict Engine) ===');
        console.log('Message:', message);
        console.log('Mode:', mode);
        console.log('Conversation History:', conversationHistory);
        console.log('Abort Signal provided:', !!signal);
        
        // Get or create session_id from localStorage for conversation continuity
        let sessionId = localStorage.getItem('atlas-session-id');
        if (!sessionId) {
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('atlas-session-id', sessionId);
        }
        
        try {
            const endpoint = `${this.baseURL}/analyze_topic`;
            
            const requestBody = {
                topic: message,
                model: 'llama3',
                mode: mode,
                session_id: sessionId,
                conversation_history: conversationHistory
            };
            
            const fetchOptions = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            };
            
            // Add abort signal if provided
            if (signal) {
                fetchOptions.signal = signal;
            }
            
            const response = await fetch(endpoint, fetchOptions);
            
            console.log('Response status:', response.status);
            console.log('Response ok:', response.ok);
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
            }
            
            // Regular JSON response
            const responseText = await response.text();
            console.log('Response text:', responseText);
            const data = JSON.parse(responseText);
            console.log('Parsed data:', data);
            
            return data;
            
        } catch (error) {
            console.error('=== API Error ===');
            console.error('Error details:', error);
            throw error;
        }
    }
};
