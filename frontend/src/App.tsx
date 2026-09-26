import { useState } from 'react'
import axios from 'axios'

const API_URL = 'http://localhost:8001'

function App() {
  const [message, setMessage] = useState('')
  const [reply, setReply] = useState('')
  const [sources, setSources] = useState<string[]>([])
  const [urgency, setUrgency] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!message.trim()) return

    setLoading(true)
    setError('')
    setReply('')
    setSources([])
    setUrgency('')

    try {
      const response = await axios.post(`${API_URL}/chat`, { message })
      setReply(response.data.reply)
      setSources(response.data.sources)
      setUrgency(response.data.urgency)
    } catch (err) {
      setError('Something went wrong talking to Baymax. Is the backend running?')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center p-8">
      <h1 className="text-3xl font-bold text-purple-600 mb-6">Baymax</h1>

      <form onSubmit={handleSubmit} className="w-full max-w-xl flex gap-2 mb-6">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Describe what's going on..."
          className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-400"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-purple-600 text-white px-4 py-2 rounded-lg disabled:opacity-50"
        >
          {loading ? 'Thinking...' : 'Send'}
        </button>
      </form>

      {error && <p className="text-red-500">{error}</p>}

      {reply && (
        <div className="w-full max-w-xl bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
          <p className="whitespace-pre-wrap text-gray-800">{reply}</p>

          {sources.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-100">
              <p className="text-sm font-semibold text-gray-500 mb-1">Sources:</p>
              <ul className="text-sm text-gray-500 list-disc list-inside">
                {sources.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </div>
          )}

          <p className="mt-2 text-xs text-gray-400">Urgency: {urgency}</p>
        </div>
      )}
    </div>
  )
}

export default App