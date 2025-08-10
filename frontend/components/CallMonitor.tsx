'use client'

import { useState, useEffect, useRef } from 'react'
import { Mic, MicOff, Play, Square } from 'lucide-react'
import { io, Socket } from 'socket.io-client'

interface CallMonitorProps {
  isConnected: boolean
  onConnectionChange: (connected: boolean) => void
  onRiskChange: (risk: 'safe' | 'suspicious' | 'scam') => void
}

export default function CallMonitor({ isConnected, onConnectionChange, onRiskChange }: CallMonitorProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [status, setStatus] = useState<string>('Disconnected')
  const socketRef = useRef<Socket | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  useEffect(() => {
    // Initialize socket connection
    socketRef.current = io('http://localhost:8000', {
      transports: ['websocket', 'polling'],
      forceNew: true,
      timeout: 20000,
      autoConnect: true
    })
    
    socketRef.current.on('connect', () => {
      setStatus('Connected to server')
      onConnectionChange(true)
    })
    
    socketRef.current.on('disconnect', () => {
      setStatus('Disconnected from server')
      onConnectionChange(false)
    })
    
    socketRef.current.on('transcription', (data: { text: string }) => {
      setTranscript(prev => prev + ' ' + data.text)
    })
    
    socketRef.current.on('risk_assessment', (data: { risk: 'safe' | 'suspicious' | 'scam', reason: string }) => {
      onRiskChange(data.risk)
      setStatus(`Risk: ${data.risk.toUpperCase()} - ${data.reason}`)
    })
    
    socketRef.current.on('alert', (data: { level: string, message: string, audio?: string, audio_format?: string }) => {
      // Display visual alert
      setStatus(`ALERT: ${data.message}`)
      
      // Play audio alert if available
      if (data.audio && data.audio_format) {
        playAudioAlert(data.audio, data.audio_format)
      }
    })

    return () => {
      socketRef.current?.disconnect()
    }
  }, [onConnectionChange, onRiskChange])

  const playAudioAlert = (audioData: string, format: string) => {
    try {
      const audioBlob = new Blob(
        [Uint8Array.from(atob(audioData), c => c.charCodeAt(0))], 
        { type: `audio/${format}` }
      )
      const audioUrl = URL.createObjectURL(audioBlob)
      const audio = new Audio(audioUrl)
      audio.volume = 0.7 // Discrete volume level
      audio.play().catch(e => console.error('Error playing alert:', e))
      
      // Clean up URL after playing
      audio.onended = () => URL.revokeObjectURL(audioUrl)
    } catch (error) {
      console.error('Error playing audio alert:', error)
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      })
      streamRef.current = stream
      
      // Use Web Audio API for direct PCM audio processing
      const audioContext = new AudioContext({ sampleRate: 16000 })
      const source = audioContext.createMediaStreamSource(stream)
      const processor = audioContext.createScriptProcessor(4096, 1, 1)
      
      processor.onaudioprocess = (event) => {
        if (socketRef.current) {
          const inputBuffer = event.inputBuffer
          const inputData = inputBuffer.getChannelData(0) // Get mono channel
          
          // Debug: Check the values we're getting
          const min = Math.min(...inputData)
          const max = Math.max(...inputData)
          console.log(`Frontend audio data - min: ${min.toFixed(3)}, max: ${max.toFixed(3)}`)
          
          // Convert Float32Array to regular array for Socket.IO
          const audioArray = Array.from(inputData)
          socketRef.current?.emit('audio_chunk', { data: audioArray })
        }
      }
      
      source.connect(processor)
      processor.connect(audioContext.destination)
      
      // Store references for cleanup
      mediaRecorderRef.current = { audioContext, source, processor } as any
      setIsRecording(true)
      setStatus('Recording and monitoring...')
    } catch (error) {
      console.error('Error starting recording:', error)
      setStatus('Error: Could not access microphone')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      const audioRefs = mediaRecorderRef.current as any
      if (audioRefs.processor) {
        audioRefs.source.disconnect()
        audioRefs.processor.disconnect()
        audioRefs.audioContext.close()
      }
      setIsRecording(false)
      setStatus('Recording stopped')
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {isRecording ? (
            <Mic className="h-5 w-5 text-red-500" />
          ) : (
            <MicOff className="h-5 w-5 text-gray-400" />
          )}
          <span className={`text-sm ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
            {status}
          </span>
        </div>
        
        <button
          onClick={isRecording ? stopRecording : startRecording}
          className={`px-4 py-2 rounded-lg font-medium flex items-center space-x-2 ${
            isRecording
              ? 'bg-red-500 hover:bg-red-600 text-white'
              : 'bg-indigo-500 hover:bg-indigo-600 text-white'
          }`}
        >
          {isRecording ? (
            <>
              <Square className="h-4 w-4" />
              <span>Stop Monitoring</span>
            </>
          ) : (
            <>
              <Play className="h-4 w-4" />
              <span>Start Monitoring</span>
            </>
          )}
        </button>
      </div>

      <div className="bg-gray-50 rounded-lg p-4 min-h-32">
        <h3 className="font-medium text-gray-700 mb-2">Live Transcript</h3>
        <div className="text-sm text-gray-600 max-h-24 overflow-y-auto">
          {transcript || 'No transcript available. Start monitoring to see real-time transcription.'}
        </div>
      </div>
    </div>
  )
}