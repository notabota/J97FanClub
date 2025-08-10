'use client'

import { AlertTriangle, CheckCircle, Shield } from 'lucide-react'

interface RiskDashboardProps {
  riskLevel: 'safe' | 'suspicious' | 'scam'
}

export default function RiskDashboard({ riskLevel }: RiskDashboardProps) {
  const getRiskColor = () => {
    switch (riskLevel) {
      case 'safe': return 'text-green-600 bg-green-100'
      case 'suspicious': return 'text-yellow-600 bg-yellow-100'
      case 'scam': return 'text-red-600 bg-red-100'
    }
  }

  const getRiskIcon = () => {
    switch (riskLevel) {
      case 'safe': return <CheckCircle className="h-8 w-8" />
      case 'suspicious': return <AlertTriangle className="h-8 w-8" />
      case 'scam': return <AlertTriangle className="h-8 w-8" />
    }
  }

  const getRiskMessage = () => {
    switch (riskLevel) {
      case 'safe':
        return {
          title: 'Call is Safe',
          description: 'No suspicious activity detected. The conversation appears legitimate.'
        }
      case 'suspicious':
        return {
          title: 'Potentially Suspicious',
          description: 'Some suspicious patterns detected. Exercise caution and avoid sharing sensitive information.'
        }
      case 'scam':
        return {
          title: 'HIGH RISK - Likely Scam',
          description: 'Strong indicators of fraudulent activity. Do not share personal information, codes, or make payments.'
        }
    }
  }

  const risk = getRiskMessage()

  return (
    <div className="space-y-4">
      <div className={`rounded-lg p-4 border-2 ${
        riskLevel === 'safe' ? 'border-green-200 bg-green-50' :
        riskLevel === 'suspicious' ? 'border-yellow-200 bg-yellow-50' :
        'border-red-200 bg-red-50'
      }`}>
        <div className="flex items-center space-x-3 mb-2">
          <div className={getRiskColor()}>
            {getRiskIcon()}
          </div>
          <h3 className={`text-lg font-semibold ${getRiskColor()}`}>
            {risk.title}
          </h3>
        </div>
        <p className="text-gray-700 text-sm">
          {risk.description}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="bg-gray-50 rounded-lg p-3">
          <h4 className="font-medium text-gray-700 mb-1">Voice Analysis</h4>
          <p className="text-gray-600">
            {riskLevel === 'scam' ? '⚠️ Synthetic voice detected' : '✅ Natural voice'}
          </p>
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <h4 className="font-medium text-gray-700 mb-1">Content Analysis</h4>
          <p className="text-gray-600">
            {riskLevel === 'safe' ? '✅ Normal conversation' : 
             riskLevel === 'suspicious' ? '⚠️ Suspicious patterns' :
             '🚨 Scam indicators'}
          </p>
        </div>
      </div>

      {riskLevel !== 'safe' && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <h4 className="font-medium text-blue-800 mb-2 flex items-center">
            <Shield className="h-4 w-4 mr-1" />
            Protection Tips
          </h4>
          <ul className="text-sm text-blue-700 space-y-1">
            <li>• Never share personal information, passwords, or verification codes</li>
            <li>• Don't make immediate payments or transfers</li>
            <li>• Hang up and call back using official numbers</li>
            <li>• Verify the caller's identity through other means</li>
          </ul>
        </div>
      )}
    </div>
  )
}