import { useState, useRef, useCallback, useEffect } from 'react'
import { Send, Bot, User, Loader2, Sparkles, Plus } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-3 py-2">
      <div className="w-1.5 h-1.5 bg-databricks-orange rounded-full animate-pulse-dot" style={{ animationDelay: '0s' }} />
      <div className="w-1.5 h-1.5 bg-databricks-orange rounded-full animate-pulse-dot" style={{ animationDelay: '0.2s' }} />
      <div className="w-1.5 h-1.5 bg-databricks-orange rounded-full animate-pulse-dot" style={{ animationDelay: '0.4s' }} />
    </div>
  )
}

function MessageBubble({ message, isLoading }) {
  const isUser = message.role === 'user'
  const showTyping = !isUser && isLoading && !message.content

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        isUser
          ? 'bg-gradient-to-br from-databricks-red to-databricks-orange'
          : 'bg-databricks-gray border border-white/10'
      }`}>
        {isUser ? <User className="w-4 h-4 text-white" /> : <Bot className="w-4 h-4 text-databricks-orange" />}
      </div>

      <div className={`flex flex-col max-w-[75%] ${isUser ? 'items-end' : 'items-start'}`}>
        {showTyping ? (
          <div className="rounded-2xl px-4 py-3 bg-databricks-gray/80 border border-white/5">
            <TypingIndicator />
          </div>
        ) : (
          <div className={`rounded-2xl px-4 py-3 ${
            isUser
              ? 'bg-gradient-to-br from-databricks-red/90 to-databricks-orange/80 text-white'
              : 'bg-databricks-gray/80 border border-white/5 text-databricks-light'
          }`}>
            {isUser ? (
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
            ) : (
              <div className="prose-chat text-sm">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [threadId, setThreadId] = useState(null)
  const [userId, setUserId] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const msgIdCounter = useRef(0)

  const nextMsgId = () => `msg-${++msgIdCounter.current}`

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(scrollToBottom, [messages])

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || isLoading) return

    const userMsg = { id: nextMsgId(), role: 'user', content: text }
    const assistantMsgId = nextMsgId()
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    const apiMessages = [...messages, userMsg].map(m => ({
      role: m.role,
      content: m.content,
    }))

    try {
      // Add placeholder assistant message (shows typing dots)
      setMessages(prev => [...prev, { id: assistantMsgId, role: 'assistant', content: '' }])

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: apiMessages,
          thread_id: threadId,
          user_id: userId,
        }),
      })

      if (!res.ok) throw new Error(`Server error: ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let assistantContent = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event = JSON.parse(line.slice(6))

            if (event.type === 'delta') {
              assistantContent += event.content
              setMessages(prev => {
                const updated = [...prev]
                const idx = updated.findIndex(m => m.id === assistantMsgId)
                if (idx >= 0) updated[idx] = { ...updated[idx], content: assistantContent }
                return updated
              })
            } else if (event.type === 'done') {
              if (event.thread_id) setThreadId(event.thread_id)
              if (event.user_id) setUserId(event.user_id)
            } else if (event.type === 'error') {
              assistantContent = `Error: ${event.content}`
              setMessages(prev => {
                const updated = [...prev]
                const idx = updated.findIndex(m => m.id === assistantMsgId)
                if (idx >= 0) updated[idx] = { ...updated[idx], content: assistantContent }
                return updated
              })
            }
          } catch { /* skip malformed */ }
        }
      }

      // Sync fallback if streaming returned nothing
      if (!assistantContent) {
        setMessages(prev => prev.filter(m => m.id !== assistantMsgId))
        const syncRes = await fetch('/api/chat/sync', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: apiMessages, thread_id: threadId, user_id: userId }),
        })
        const data = await syncRes.json()
        setMessages(prev => [...prev, {
          id: assistantMsgId,
          role: 'assistant',
          content: data.content || 'No response received.',
        }])
        if (data.thread_id) setThreadId(data.thread_id)
        if (data.user_id) setUserId(data.user_id)
      }
    } catch (err) {
      setMessages(prev => {
        const idx = prev.findIndex(m => m.id === assistantMsgId)
        if (idx >= 0) {
          const updated = [...prev]
          updated[idx] = { ...updated[idx], content: `Connection error: ${err.message}` }
          return updated
        }
        return [...prev, { id: assistantMsgId, role: 'assistant', content: `Connection error: ${err.message}` }]
      })
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }, [input, isLoading, messages, threadId, userId])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleNewChat = useCallback(() => {
    setMessages([])
    setThreadId(null)
    setInput('')
    inputRef.current?.focus()
  }, [])

  const isEmpty = messages.length === 0

  return (
    <div className="h-screen flex flex-col bg-databricks-darker">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-white/5 bg-databricks-dark/50 backdrop-blur-sm">
        <div className="max-w-3xl mx-auto px-4 py-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-databricks-red to-databricks-orange flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div className="flex-1">
            <h1 className="text-sm font-semibold text-white">Workshop Agent</h1>
            <p className="text-xs text-white/40">Powered by Databricks</p>
          </div>
          {!isEmpty && (
            <button
              onClick={handleNewChat}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white/60 hover:text-white bg-databricks-gray/50 hover:bg-databricks-gray border border-white/10 rounded-lg transition-all"
            >
              <Plus className="w-3.5 h-3.5" />
              New Chat
            </button>
          )}
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-6">
          {isEmpty ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-databricks-red/20 to-databricks-orange/20 border border-databricks-red/20 flex items-center justify-center mb-6">
                <Sparkles className="w-8 h-8 text-databricks-orange" />
              </div>
              <h2 className="text-xl font-semibold text-white mb-2">Workshop Agent</h2>
              <p className="text-white/40 text-sm mb-8 max-w-md">
                Ask me about weather, calculations, or search the knowledge base.
                I can also remember things about you across conversations.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
                {[
                  "What's the weather in Tokyo?",
                  "Calculate 42 * 17",
                  "What is Unity Catalog?",
                  "Remember that I prefer Python",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => { setInput(suggestion); inputRef.current?.focus() }}
                    className="text-left text-sm px-4 py-3 rounded-xl bg-databricks-gray/50 border border-white/5 text-white/60 hover:text-white hover:border-databricks-orange/30 hover:bg-databricks-gray transition-all"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} isLoading={isLoading} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </main>

      {/* Input */}
      <footer className="flex-shrink-0 border-t border-white/5 bg-databricks-dark/30 backdrop-blur-sm">
        <div className="max-w-3xl mx-auto px-4 py-4">
          <div className="flex items-end gap-3 bg-databricks-gray/60 border border-white/10 rounded-2xl px-4 py-2 focus-within:border-databricks-orange/40 transition-colors">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask something..."
              rows={1}
              className="flex-1 bg-transparent text-sm text-white placeholder-white/30 resize-none outline-none py-1.5 max-h-32"
              style={{ minHeight: '24px' }}
              onInput={(e) => {
                e.target.style.height = '24px'
                e.target.style.height = Math.min(e.target.scrollHeight, 128) + 'px'
              }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className={`flex-shrink-0 p-2 rounded-xl transition-all ${
                input.trim() && !isLoading
                  ? 'bg-gradient-to-r from-databricks-red to-databricks-orange text-white hover:opacity-90'
                  : 'bg-white/5 text-white/20 cursor-not-allowed'
              }`}
            >
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </button>
          </div>
          <p className="text-center text-[10px] text-white/20 mt-2">
            Responses may be inaccurate. Verify important information.
          </p>
        </div>
      </footer>
    </div>
  )
}
