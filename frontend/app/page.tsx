'use client'

import { useState, useEffect } from 'react'
import { Shield, Phone, AlertTriangle, CheckCircle } from 'lucide-react'
import CallMonitor from '@/components/CallMonitor'
import RiskDashboard from '@/components/RiskDashboard'

export default function Home() {
  const [isConnected, setIsConnected] = useState(false)
  const [riskLevel, setRiskLevel] = useState<'safe' | 'suspicious' | 'scam'>('safe')

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Shield className="h-12 w-12 text-indigo-600 mr-3" />
            <h1 className="text-4xl font-bold text-gray-800">Voice Scam Shield</h1>
          </div>
          <p className="text-gray-600 text-lg">
            Real-time AI-powered call scam detection in multiple languages
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-4 flex items-center">
              <Phone className="h-6 w-6 mr-2 text-indigo-600" />
              Call Monitor
            </h2>
            <CallMonitor 
              isConnected={isConnected}
              onConnectionChange={setIsConnected}
              onRiskChange={setRiskLevel}
            />
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-4 flex items-center">
              {riskLevel === 'safe' && <CheckCircle className="h-6 w-6 mr-2 text-green-600" />}
              {riskLevel === 'suspicious' && <AlertTriangle className="h-6 w-6 mr-2 text-yellow-600" />}
              {riskLevel === 'scam' && <AlertTriangle className="h-6 w-6 mr-2 text-red-600" />}
              Risk Assessment
            </h2>
            <RiskDashboard riskLevel={riskLevel} />
          </div>
        </div>

        <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Supported Languages</h2>
          <div className="flex flex-wrap gap-2">
            <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm">🇺🇸 English</span>
            <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm">🇪🇸 Spanish</span>
            <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm">🇫🇷 French</span>
          </div>
        </div>
      </div>
    </main>
  )
}