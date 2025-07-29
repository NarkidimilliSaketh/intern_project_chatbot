// client/src/components/ChatPage.js
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    sendMessage as apiSendMessage, saveChatHistory, generatePodcast, generateMindMap,
    getUserFiles, deleteUserFile, renameUserFile, performDeepSearch,
    queryHybridRagService, getSessionDetails, summarizeConversation
} from '../services/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { v4 as uuidv4 } from 'uuid';
import { FaBars, FaPaperPlane, FaMicrophone, FaHistory, FaPlus, FaCog, FaFolderOpen, FaSignOutAlt, FaTimes } from 'react-icons/fa';
import { Popover, Typography, Button, Box, IconButton as MuiIconButton } from '@mui/material';

import FileUploadWidget from './FileUploadWidget';
import FileManagerWidget from './FileManagerWidget';
import HistorySidebarWidget from './HistorySidebarWidget';
import SettingsWidget from './SettingsWidget';
import MindMap from './MindMap';
import { getPromptTextById, availablePrompts } from '../utils/prompts';

import './ChatPage.css';

const GeminiIcon = () => (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ transform: 'scale(1.2)' }}>
        <path d="M12 2.75L13.2319 8.26814L15.1914 5.80859L15.7319 8.73186L19.1914 7.80859L17.2319 10.2681L20.1914 11.8086L17.7319 12.2681L18.1914 14.8086L15.2319 13.7681L14.1914 16.1914L12.7319 13.2681L12 15.25L11.2681 13.2681L9.80859 16.1914L8.76814 13.7681L5.80859 14.8086L6.26814 12.2681L3.80859 11.8086L6.76814 10.2681L4.80859 7.80859L8.26814 8.73186L8.80859 5.80859L10.7681 8.26814L12 2.75Z" fill="url(#gemini-gradient)" />
        <defs>
            <linearGradient id="gemini-gradient" x1="4" y1="4" x2="20" y2="20" gradientUnits="userSpaceOnUse">
                <stop stopColor="#8957E9" />
                <stop offset="1" stopColor="#7094E6" />
            </linearGradient>
        </defs>
    </svg>
);

const ChatPage = ({ setIsAuthenticated }) => {
    const [isSidebarExpanded, setIsSidebarExpanded] = useState(window.innerWidth > 1024);
    const [sidebarView, setSidebarView] = useState('files');
    const [profileAnchorEl, setProfileAnchorEl] = useState(null);
    const [loadingStates, setLoadingStates] = useState({
        chat: false, files: false, podcast: false, mindMap: false, deepSearch: false, listening: false
    });
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState('');
    const [error, setError] = useState('');
    const [sessionId, setSessionId] = useState('');
    const [userId, setUserId] = useState('');
    const [username, setUsername] = useState('');
    const [currentSystemPromptId, setCurrentSystemPromptId] = useState('friendly');
    const [editableSystemPromptText, setEditableSystemPromptText] = useState(() => getPromptTextById('friendly'));
    const [files, setFiles] = useState([]);
    const [fileError, setFileError] = useState('');
    const [activeFileForRag, setActiveFileForRag] = useState(null); // Tracks { id, name } of the active file
    const [isDeepSearchEnabled, setIsDeepSearchEnabled] = useState(false);
    const [currentlySpeakingIndex, setCurrentlySpeakingIndex] = useState(null);
    const [conversationSummary, setConversationSummary] = useState('');
    const [ragSourceInfo, setRagSourceInfo] = useState(null); // Feedback for RAG responses
    const summaryTriggerCount = useRef(0);

    const messagesEndRef = useRef(null);
    const recognitionRef = useRef(null);
    const navigate = useNavigate();
    const isProcessing = Object.values(loadingStates).some(Boolean);
    const handleProfileClick = (event) => setProfileAnchorEl(event.currentTarget);
    const handleProfileClose = () => setProfileAnchorEl(null);
    const isProfileOpen = Boolean(profileAnchorEl);

    const handleSidebarAction = () => {
        if (window.innerWidth < 1024) {
            setIsSidebarExpanded(false);
        }
    };

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    useEffect(() => {
        const SUMMARY_THRESHOLD = 6;
        if (messages.length >= SUMMARY_THRESHOLD && messages.length > summaryTriggerCount.current) {
            summaryTriggerCount.current = messages.length;
            const generateSummary = async () => {
                try {
                    const response = await summarizeConversation(messages);
                    if (response.data.summary) {
                        setConversationSummary(response.data.summary);
                        console.log("Conversation summary updated:", response.data.summary);
                    }
                } catch (err) {
                    console.error("Failed to generate conversation summary:", err);
                }
            };
            generateSummary();
        }
    }, [messages]);

    const saveAndReset = useCallback(async (isLoggingOut = false, onCompleteCallback = null) => {
        const currentSessionId = localStorage.getItem('sessionId');
        const currentUserId = localStorage.getItem('userId');
        if (!currentSessionId || !currentUserId || isProcessing || messages.length === 0) {
            if (onCompleteCallback) onCompleteCallback();
            return;
        }
        try {
            const firstUserMessage = messages.find(m => m.role === 'user');
            const chatTitle = firstUserMessage ? firstUserMessage.parts[0].text.substring(0, 50) : 'New Conversation';
            await saveChatHistory({ sessionId: currentSessionId, messages, systemPrompt: editableSystemPromptText, title: chatTitle });
        } catch (saveError) {
            console.error("Failed to save chat history:", saveError);
        } finally {
            if (!isLoggingOut) {
                setMessages([]);
                setConversationSummary('');
                summaryTriggerCount.current = 0;
                setActiveFileForRag(null);
                setRagSourceInfo(null);
                const newSessionId = uuidv4();
                setSessionId(newSessionId);
                localStorage.setItem('sessionId', newSessionId);
            }
            if (onCompleteCallback) onCompleteCallback();
        }
    }, [messages, isProcessing, editableSystemPromptText]);

    const handleLogout = useCallback((skipSave = false) => {
        const performCleanup = () => {
            if (window.speechSynthesis) window.speechSynthesis.cancel();
            localStorage.clear();
            setIsAuthenticated(false);
            navigate('/login', { replace: true });
        };
        if (!skipSave && messages.length > 0) {
            saveAndReset(true, performCleanup);
        } else {
            performCleanup();
        }
    }, [messages.length, setIsAuthenticated, navigate, saveAndReset]);

    useEffect(() => {
        const storedUserId = String(localStorage.getItem('userId'));
        const storedUsername = localStorage.getItem('username');
        if (!storedUserId || !storedUsername) {
            handleLogout(true);
        } else {
            setUserId(storedUserId);
            setUsername(storedUsername);
            const newSessionId = uuidv4();
            setSessionId(newSessionId);
            localStorage.setItem('sessionId', newSessionId);
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            const recognition = new SpeechRecognition();
            recognition.lang = 'en-US';
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;
            recognition.onstart = () => setLoadingStates(prev => ({ ...prev, listening: true }));
            recognition.onresult = (event) => setInputText(event.results[0][0].transcript);
            recognition.onerror = (e) => setError(`STT Error: ${e.error}`);
            recognition.onend = () => setLoadingStates(prev => ({ ...prev, listening: false }));
            recognitionRef.current = recognition;
        } else {
            console.warn('Web Speech API is not supported in this browser.');
        }
        return () => {
            if (recognitionRef.current) recognitionRef.current.stop();
            if (window.speechSynthesis) window.speechSynthesis.cancel();
        };
    }, [handleLogout]);

    const fetchFiles = useCallback(async () => {
        if (!userId) return;
        setLoadingStates(prev => ({ ...prev, files: true }));
        setFileError('');
        try {
            const response = await getUserFiles();
            setFiles(response.data || []);
        } catch (err) {
            setFileError('Could not load files.');
        } finally {
            setLoadingStates(prev => ({ ...prev, files: false }));
        }
    }, [userId]);
    
    useEffect(() => {
        if (userId) fetchFiles();
    }, [userId, fetchFiles]);

    const handleNewChat = useCallback(() => {
        if (!isProcessing) {
            saveAndReset(false, () => {
                setSidebarView('files');
            });
        }
    }, [isProcessing, saveAndReset]);

    const handleSendMessage = useCallback(async (e) => {
        if (e) e.preventDefault();
        const trimmedInput = inputText.trim();
        if (!trimmedInput || isProcessing) return;
        if (loadingStates.listening) recognitionRef.current?.stop();

        const newUserMessage = { role: 'user', parts: [{ text: trimmedInput }], timestamp: new Date() };
        setMessages(prev => [...prev, newUserMessage]);
        setInputText('');
        setError('');
        setRagSourceInfo(null); // Clear previous source info

        const historyToSend = [...messages, newUserMessage];
        
        if (isDeepSearchEnabled) {
            setLoadingStates(prev => ({ ...prev, deepSearch: true }));
            try {
                const response = await performDeepSearch(trimmedInput);
                const deepSearchResult = {
                    role: 'assistant', type: 'deep_search',
                    parts: [{ text: response.data.message }],
                    timestamp: new Date(), metadata: response.data.metadata
                };
                setMessages(prev => [...prev, deepSearchResult]);
            } catch (err) {
                setError(`Deep Search Error: ${err.response?.data?.message || 'Deep search failed.'}`);
                setMessages(prev => prev.slice(0, -1));
            } finally {
                setLoadingStates(prev => ({ ...prev, deepSearch: false }));
            }
        } else if (activeFileForRag) {
            setLoadingStates(prev => ({ ...prev, chat: true }));
            try {
                const ragPayload = {
                    query: trimmedInput,
                    fileId: activeFileForRag.id
                };
                const response = await queryHybridRagService(ragPayload);
                const assistantMessage = {
                    role: 'assistant', type: response.data.metadata.searchType,
                    parts: [{ text: response.data.message }],
                    timestamp: new Date(), metadata: response.data.metadata
                };
                setMessages(prev => [...prev, assistantMessage]);
                
                if (response.data.metadata.searchType === 'rag') {
                    const count = response.data.metadata.source_count || 0;
                    setRagSourceInfo(`Answered from ${count} relevant section(s) in "${activeFileForRag.name}".`);
                } else if (response.data.metadata.searchType === 'summary') {
                    setRagSourceInfo(`Generated a summary from the entire document: "${activeFileForRag.name}".`);
                }

            } catch (err) {
                setError(`RAG Error: ${err.response?.data?.message || 'RAG query failed.'}`);
                setMessages(prev => prev.slice(0, -1));
            } finally {
                setLoadingStates(prev => ({ ...prev, chat: false }));
            }
        } else {
            setLoadingStates(prev => ({ ...prev, chat: true }));
            try {
                const payload = {
                    query: trimmedInput,
                    history: historyToSend,
                    sessionId,
                    systemPrompt: editableSystemPromptText,
                    conversationSummary
                };
                const response = await apiSendMessage(payload);
                const assistantMessage = {
                    role: 'assistant', parts: [{ text: response.data.message }], timestamp: new Date()
                };
                setMessages(prev => [...prev, assistantMessage]);
            } catch (err) {
                setError(err.response?.data?.error || 'Chat error.');
                setMessages(prev => prev.slice(0, -1));
            } finally {
                setLoadingStates(prev => ({ ...prev, chat: false }));
            }
        }
    }, [
        inputText, isProcessing, loadingStates.listening, messages, isDeepSearchEnabled,
        activeFileForRag, sessionId, editableSystemPromptText, conversationSummary
    ]);
    
    const handleEnterKey = useCallback((e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    }, [handleSendMessage]);

    const handleMicButtonClick = useCallback(() => {
        if (!recognitionRef.current) return;
        if (loadingStates.listening) {
            recognitionRef.current.stop();
        } else {
            recognitionRef.current.start();
        }
    }, [loadingStates.listening]);

    const handleLoadSession = useCallback(async (sessionIdToLoad) => {
        try {
            const response = await getSessionDetails(sessionIdToLoad);
            const sessionData = response.data;
            if (sessionData?.messages) {
                setMessages(sessionData.messages);
                setEditableSystemPromptText(sessionData.systemPrompt || getPromptTextById('friendly'));
                setCurrentSystemPromptId(availablePrompts.find(p => p.prompt === sessionData.systemPrompt)?.id || 'custom');
                setSessionId(sessionData.sessionId);
                localStorage.setItem('sessionId', sessionData.sessionId);
                setConversationSummary('');
                summaryTriggerCount.current = 0;
                setActiveFileForRag(null);
                setRagSourceInfo(null);
                setSidebarView('files');
                handleSidebarAction();
            }
        } catch (err) {
            setError(`Failed to load session ${sessionIdToLoad}.`);
        }
    }, []);
    
    const handleTextToSpeech = useCallback((text, index) => {
        if (!('speechSynthesis' in window)) {
            setError('Sorry, your browser does not support text-to-speech.');
            return;
        }
        window.speechSynthesis.cancel();
        if (currentlySpeakingIndex === index) {
            setCurrentlySpeakingIndex(null);
            return;
        }
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.onend = () => setCurrentlySpeakingIndex(null);
        utterance.onerror = () => {
            setError('An error occurred during speech synthesis.');
            setCurrentlySpeakingIndex(null);
        };
        setCurrentlySpeakingIndex(index);
        window.speechSynthesis.speak(utterance);
    }, [currentlySpeakingIndex]);

    const handleDeleteFile = async (fileId, fileName) => {
        if (window.confirm(`Are you sure you want to delete "${fileName}"?`)) {
            try {
                await deleteUserFile(fileId);
                fetchFiles();
                if (activeFileForRag?.id === fileId) {
                    setActiveFileForRag(null);
                    setRagSourceInfo(null);
                }
            } catch (err) {
                setFileError(`Could not delete ${fileName}.`);
            }
        }
    };

    const handleRenameFile = async (fileId, currentName) => {
        const newName = prompt("Enter new file name:", currentName);
        if (newName && newName !== currentName) {
            try {
                await renameUserFile(fileId, newName);
                fetchFiles();
                if (activeFileForRag?.id === fileId) {
                    setActiveFileForRag(prev => ({ ...prev, name: newName }));
                }
            } catch (err) {
                setFileError(`Could not rename file.`);
            }
        }
    };
    
    const handleChatWithFile = useCallback((fileId, fileName) => {
        setActiveFileForRag({ id: fileId, name: fileName });
        setIsDeepSearchEnabled(false);
        setMessages(prev => [...prev, {
            role: 'system',
            parts: [{ text: `Switched to RAG mode. Now chatting with "${fileName}".` }],
            timestamp: new Date()
        }]);
    }, []);

    const handleGeneratePodcast = useCallback(async (fileId, fileName) => {
        if (isProcessing) return;
        setLoadingStates(prev => ({ ...prev, podcast: true }));
        setError('');
        setMessages(prev => [...prev, { role: 'user', parts: [{ text: `Requesting a podcast for "${fileName}"...` }], timestamp: new Date() }]);
        try {
            const response = await generatePodcast(fileId);
            const isAudioFile = response.data.podcastUrl?.endsWith('.mp3') || response.data.podcastUrl?.endsWith('.wav');
            if (isAudioFile) {
                const podcastMessage = {
                    role: 'assistant', type: 'audio',
                    parts: [{ text: `🎧 Podcast generated successfully!` }],
                    audioUrl: response.data.podcastUrl, timestamp: new Date()
                };
                setMessages(prev => [...prev, podcastMessage]);
            } else {
                throw new Error('Podcast generation failed. Audio could not be generated.');
            }
        } catch (err) {
            let errorMessageText = err.response?.data?.message || err.message || 'Failed to generate podcast.';
            if (errorMessageText.includes('Not enough content in file')) {
                errorMessageText = 'The selected file does not have enough content to generate a podcast. Please upload a longer or more detailed file.';
            }
            setError(`Podcast Error: ${errorMessageText}`);
            setMessages(prev => [...prev, { role: 'assistant', parts: [{ text: `Error generating podcast: ${errorMessageText}` }], timestamp: new Date() }]);
        }
        setLoadingStates(prev => ({ ...prev, podcast: false }));
    }, [isProcessing]);
    
    const handleGenerateMindMap = useCallback(async (fileId, fileName) => {
        if (isProcessing) return;
        setLoadingStates(prev => ({ ...prev, mindMap: true }));
        setError('');
        setMessages(prev => [...prev, { role: 'user', parts: [{ text: `Generate a mind map for the file: ${fileName}` }], timestamp: new Date() }]);
        try {
            const response = await generateMindMap(fileId);
            const mindMapData = response.data?.mindmap || response.data;
            if (mindMapData?.nodes) {
                const mindMapMessage = {
                    role: 'assistant', type: 'mindmap',
                    parts: [{ text: `Here is the mind map for "${fileName}":` }],
                    mindMapData: mindMapData, timestamp: new Date()
                };
                setMessages(prev => [...prev, mindMapMessage]);
            } else {
                throw new Error('Invalid mind map data received from server');
            }
        } catch (err) {
            const errorMessageText = err.response?.data?.message || err.message || 'Failed to generate mind map.';
            setError(`Mind Map Error: ${errorMessageText}`);
            setMessages(prev => [...prev, { role: 'assistant', parts: [{ text: `❌ Mind Map Error: ${errorMessageText}` }], timestamp: new Date() }]);
        } finally {
            setLoadingStates(prev => ({ ...prev, mindMap: false }));
        }
    }, [isProcessing]);

    const handlePromptSelectChange = useCallback((newId) => {
        setCurrentSystemPromptId(newId);
        setEditableSystemPromptText(getPromptTextById(newId));
    }, []);
    
    const handlePromptTextChange = useCallback((newText) => {
        setEditableSystemPromptText(newText);
        const matchingPreset = availablePrompts.find(p => p.id !== 'custom' && p.prompt === newText);
        setCurrentSystemPromptId(matchingPreset ? matchingPreset.id : 'custom');
    }, []);

    if (!userId) {
        return <div className="loading-indicator"><span>Initializing...</span></div>;
    }
    
    return (
        <div className="chat-page-container">
            {isSidebarExpanded && <div className="drawer-overlay" onClick={() => setIsSidebarExpanded(false)} />}
            
            <div className={`sidebar-area ${isSidebarExpanded ? 'expanded' : 'collapsed'}`}>
                <div className="sidebar-icons">
                    <div className="sidebar-icons-top">
                        <button onClick={() => setIsSidebarExpanded(p => !p)} className="icon-button" title="Toggle Menu"> <FaBars /> </button>
                        <button onClick={handleNewChat} className="icon-button" disabled={isProcessing} title="New Chat"> <FaPlus /> </button>
                        <button onClick={() => setSidebarView('history')} className={`icon-button ${sidebarView === 'history' ? 'active' : ''}`} disabled={isProcessing} title="Chat History"> <FaHistory /> </button>
                        <button onClick={() => setSidebarView('files')} className={`icon-button ${sidebarView === 'files' ? 'active' : ''}`} title="My Files"> <FaFolderOpen /> </button>
                    </div>
                    <div className="sidebar-icons-bottom">
                            <button onClick={() => setSidebarView('settings')} className={`icon-button ${sidebarView === 'settings' ? 'active' : ''}`} title="Settings"> <FaCog /> </button>
                    </div>
                </div>
                <div className="sidebar-content">
                    <div className="mobile-sidebar-nav">
                        <button onClick={() => { handleNewChat(); setIsSidebarExpanded(false); }} className="mobile-nav-button" disabled={isProcessing}>
                            <FaPlus /> New Chat
                        </button>
                        <button onClick={() => setSidebarView('history')} className={`mobile-nav-button ${sidebarView === 'history' ? 'active' : ''}`} disabled={isProcessing}>
                            <FaHistory /> Chat History
                        </button>
                        <button onClick={() => setSidebarView('files')} className={`mobile-nav-button ${sidebarView === 'files' ? 'active' : ''}`}>
                            <FaFolderOpen /> My Files
                        </button>
                        <button onClick={() => setSidebarView('settings')} className={`mobile-nav-button ${sidebarView === 'settings' ? 'active' : ''}`}>
                            <FaCog /> Settings
                        </button>
                    </div>
                    <div className="sidebar-divider" />

                    {sidebarView === 'files' && (
                        <>
                            <FileUploadWidget onUploadSuccess={fetchFiles} />
                            <FileManagerWidget
                                files={files}
                                isLoading={loadingStates.files}
                                error={fileError}
                                onDeleteFile={handleDeleteFile}
                                onRenameFile={handleRenameFile}
                                onGeneratePodcast={handleGeneratePodcast}
                                onGenerateMindMap={handleGenerateMindMap}
                                onChatWithFile={handleChatWithFile}
                                isProcessing={isProcessing}
                                onActionTaken={handleSidebarAction}
                            />
                        </>
                    )}
                    {sidebarView === 'history' && (
                        <HistorySidebarWidget onLoadSession={handleLoadSession} />
                    )}
                    {sidebarView === 'settings' && (
                        <SettingsWidget 
                            selectedPromptId={currentSystemPromptId}
                            promptText={editableSystemPromptText}
                            onSelectChange={handlePromptSelectChange}
                            onTextChange={handlePromptTextChange}
                        />
                    )}
                </div>
            </div>
    
            <div className="chat-container">
                <header className="chat-header">
                    <div className="header-left">
                        <button onClick={() => setIsSidebarExpanded(p => !p)} className="header-button hamburger-button-mobile" title="Toggle Menu">
                            <FaBars />
                        </button>
                        <h1>TutorAI</h1>
                    </div>
                    <div className="header-right">
                        <MuiIconButton onClick={handleProfileClick} title="Profile">
                            <div className="avatar profile-avatar">{username?.[0]?.toUpperCase()}</div>
                        </MuiIconButton>
                        <Popover
                            open={isProfileOpen}
                            anchorEl={profileAnchorEl}
                            onClose={handleProfileClose}
                            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                        >
                            <Box sx={{ p: 2, minWidth: '240px', backgroundColor: '#2d2d2d', color: '#fff', border: '1px solid #444', borderRadius: '8px' }}>
                                <Typography variant="subtitle1" gutterBottom>
                                    Signed in as
                                </Typography>
                                <Typography variant="body1" color="text.primary" sx={{ mb: 2, fontWeight: 'bold' }}>
                                    {username}
                                </Typography>
                                <Button
                                    fullWidth
                                    variant="contained"
                                    color="error"
                                    startIcon={<FaSignOutAlt />}
                                    sx={{ 
                                        textTransform: 'none', 
                                        backgroundColor: '#B00020', 
                                        '&:hover': { backgroundColor: '#D50000' } 
                                    }}
                                    onClick={() => handleLogout(false)}
                                >
                                    Logout
                                </Button>
                            </Box>
                        </Popover>
                    </div>
                </header>
    
                <main className="messages-area">
                    {messages.length === 0 && !isProcessing && (
                        <div className="welcome-message">
                            <GeminiIcon />
                            <h2>How can I help you today?</h2>
                        </div>
                    )}
                    {messages.map((msg, index) => {
                            if (!msg?.role || !msg?.parts?.length) return null;
                            const messageText = msg.parts[0]?.text || '';
                            return (
                                <div key={index} className={`message-row ${msg.role}`}>
                                    <div className="avatar">
                                        {msg.role === 'user' ? (username?.[0]?.toUpperCase() || 'U') : <GeminiIcon />}
                                    </div>
                                    <div className="message-bubble-container">
                                        <div className={`message-bubble ${msg.type || ''}`}>
                                            {msg.type === 'mindmap' && msg.mindMapData ? (
                                                <div className="mindmap-container">
                                                    <MindMap mindMapData={msg.mindMapData} />
                                                </div>
                                            ) : msg.type === 'audio' && msg.audioUrl ? (
                                                <div className="audio-player-container">
                                                    <p>{messageText}</p>
                                                    <audio controls src={msg.audioUrl} />
                                                </div>
                                            ) : (
                                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{messageText}</ReactMarkdown>
                                            )}
                                        </div>
                                        <div className="message-actions">
                                            {msg.role === 'assistant' && (
                                                <button
                                                    onClick={() => handleTextToSpeech(messageText, index)}
                                                    className={`tts-button ${currentlySpeakingIndex === index ? 'speaking' : ''}`}
                                                    title="Read aloud"
                                                    disabled={isProcessing}
                                                >
                                                     <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M11.536 14.01A8.473 8.473 0 0 0 14.026 8a8.473 8.473 0 0 0-2.49-6.01l-.708.707A7.476 7.476 0 0 1 13.025 8c0 2.071-.84 3.946-2.197 5.303l.708.707z"/><path d="M10.121 12.596A6.48 6.48 0 0 0 12.025 8a6.48 6.48 0 0 0-1.904-4.596l-.707.707A5.483 5.483 0 0 1 11.025 8a5.483 5.483 0 0 1-1.61 3.89l.706.706z"/><path d="M8.707 11.182A4.486 4.486 0 0 0 10.025 8a4.486 4.486 0 0 0-1.318-3.182L8 5.525A3.489 3.489 0 0 1 9.025 8 3.49 3.49 0 0 1 8 10.475l.707.707zM6.717 3.55A.5.5 0 0 1 7 4v8a.5.5 0 0 1-.812.39L3.825 10.5H1.5A.5.5 0 0 1 1 10V6a.5.5 0 0 1 .5-.5h2.325l2.363-1.89a.5.5 0 0 1 .529-.06z"/></svg>
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            );
                    })}
                    <div ref={messagesEndRef} />
                </main>
    
                <footer className="chat-footer">
                    <div className="input-area-container">
                        {ragSourceInfo && (
                            <div className="rag-source-info">
                                <span>{ragSourceInfo}</span>
                            </div>
                        )}
                        {activeFileForRag && !ragSourceInfo && (
                            <div className="active-rag-indicator">
                                <span>Chatting with: <strong>{activeFileForRag.name}</strong></span>
                                <button onClick={() => setActiveFileForRag(null)} title="Stop chatting with this file">
                                    <FaTimes />
                                </button>
                            </div>
                        )}
                        <form className="modern-input-bar" onSubmit={handleSendMessage}>
                            <button
                                type="button"
                                className={`input-action-btn ${isDeepSearchEnabled ? 'active' : ''}`}
                                title="Deep Research"
                                onClick={() => { setIsDeepSearchEnabled(v => !v); setActiveFileForRag(null); }}
                                disabled={isProcessing}
                            >
                                DS
                            </button>
                            <button
                                type="button"
                                className={`input-action-btn ${activeFileForRag ? 'active' : ''}`}
                                title={activeFileForRag ? "Deactivate RAG mode" : "Select a file to enable RAG"}
                                onClick={() => setActiveFileForRag(null)}
                                disabled={isProcessing || !files.length}
                            >
                                RAG
                            </button>
                            <textarea
                                value={inputText}
                                onChange={e => setInputText(e.target.value)}
                                onKeyDown={handleEnterKey}
                                placeholder={activeFileForRag ? `Ask about ${activeFileForRag.name}...` : "Enter a prompt here"}
                                className="modern-input"
                                disabled={isProcessing}
                                rows="1"
                            />
                            <button
                                type="button"
                                className="input-action-btn mic-button"
                                title="Use microphone"
                                onClick={handleMicButtonClick}
                                disabled={isProcessing}
                            >
                                <FaMicrophone />
                            </button>
                            <button
                                type="submit"
                                className="input-action-btn send-button"
                                title="Send message"
                                disabled={isProcessing || !inputText.trim()}
                            >
                                <FaPaperPlane />
                            </button>
                        </form>
                        {error && <p className="error-message">{error}</p>}
                    </div>
                </footer>
            </div>
        </div>
    );
};

export default ChatPage;