import React, { useState, useEffect } from 'react';
import { TrendingUp, BarChart3, Shield, Zap, ArrowRight, GitBranch, ChevronDown, ChevronUp } from 'lucide-react';

const Home = ({ onGetStarted }) => {
  const [versionHistory, setVersionHistory] = useState([]);
  const [showVersionHistory, setShowVersionHistory] = useState(false);
  const [loadingVersions, setLoadingVersions] = useState(false);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  useEffect(() => {
    // Fetch version history
    const fetchVersionHistory = async () => {
      setLoadingVersions(true);
      try {
        const response = await fetch(`${apiUrl}/api/version-history`);
        if (response.ok) {
          const data = await response.json();
          setVersionHistory(data.versions || []);
        }
      } catch (error) {
        console.error('Error fetching version history:', error);
      } finally {
        setLoadingVersions(false);
      }
    };

    fetchVersionHistory();
  }, [apiUrl]);

  const features = [
    {
      icon: <TrendingUp className="w-8 h-8" />,
      title: 'AI-Powered Analysis',
      description: 'Advanced sentiment analysis using news, Reddit, and financial data to give you accurate insights.'
    },
    {
      icon: <BarChart3 className="w-8 h-8" />,
      title: 'Visual Charts',
      description: 'Track sentiment trends over time with interactive charts and historical data.'
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: 'Personal Watchlists',
      description: 'Organize your favorite stocks into custom watchlists and track them easily.'
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: 'Real-Time Insights',
      description: 'Get instant sentiment scores and confidence ratings for up to 5 stocks at once.'
    }
  ];

  const steps = [
    {
      number: '1',
      title: 'Sign Up',
      description: 'Create your free account in seconds'
    },
    {
      number: '2',
      title: 'Search Stocks',
      description: 'Enter up to 5 stock symbols to analyze'
    },
    {
      number: '3',
      title: 'Get Insights',
      description: 'Receive comprehensive sentiment analysis with grades and confidence scores'
    },
    {
      number: '4',
      title: 'Track Trends',
      description: 'Monitor sentiment changes over time with interactive charts'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 via-gray-900 to-purple-900/20"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-32">
          <div className="text-center">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Stock Sentiment Analyzer
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
              AI-powered sentiment analysis for smarter stock decisions. Analyze news, Reddit discussions, and financial data in real-time.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={onGetStarted}
                className="px-8 py-4 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold text-lg transition-all transform hover:scale-105 flex items-center justify-center gap-2"
              >
                Get Started
                <ArrowRight className="w-5 h-5" />
              </button>
              <a
                href="#features"
                className="px-8 py-4 bg-gray-800 hover:bg-gray-700 rounded-lg font-semibold text-lg transition-all"
              >
                Learn More
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div id="features" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <h2 className="text-4xl font-bold text-center mb-12">Powerful Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-gray-800 p-6 rounded-lg hover:bg-gray-750 transition-all transform hover:scale-105"
            >
              <div className="text-blue-400 mb-4">{feature.icon}</div>
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-gray-400">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* How It Works Section */}
      <div className="bg-gray-800 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-4xl font-bold text-center mb-12">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {steps.map((step, index) => (
              <div key={index} className="text-center">
                <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-4">
                  {step.number}
                </div>
                <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
                <p className="text-gray-400">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-12 text-center">
          <h2 className="text-4xl font-bold mb-4">Ready to Get Started?</h2>
          <p className="text-xl text-blue-100 mb-8">
            Join thousands of investors making smarter decisions with AI-powered sentiment analysis.
          </p>
          <button
            onClick={onGetStarted}
            className="px-8 py-4 bg-white text-blue-600 rounded-lg font-semibold text-lg hover:bg-gray-100 transition-all transform hover:scale-105 flex items-center justify-center gap-2 mx-auto"
          >
            Start Analyzing Now
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Version History Section */}
      <div className="bg-gray-800 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-gray-700 rounded-lg border border-gray-600">
            <button
              onClick={() => setShowVersionHistory(!showVersionHistory)}
              className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-600 transition-colors rounded-lg"
            >
              <div className="flex items-center gap-3">
                <GitBranch className="w-6 h-6 text-blue-400" />
                <h2 className="text-2xl font-bold">Version History</h2>
              </div>
              {showVersionHistory ? (
                <ChevronUp className="w-5 h-5 text-gray-400" />
              ) : (
                <ChevronDown className="w-5 h-5 text-gray-400" />
              )}
            </button>

            {showVersionHistory && (
              <div className="px-6 pb-6">
                {loadingVersions ? (
                  <div className="text-center py-8 text-gray-400">
                    <p>Loading version history...</p>
                  </div>
                ) : versionHistory.length > 0 ? (
                  <div className="space-y-4 mt-4">
                    {versionHistory.map((version, index) => (
                      <div
                        key={index}
                        className="bg-gray-800 rounded-lg p-4 border border-gray-600"
                      >
                        <h3 className="text-lg font-semibold text-blue-400 mb-2">
                          {version.version}
                        </h3>
                        <div className="text-gray-300 text-sm">
                          {version.description.split('\n').map((line, lineIndex) => {
                            const trimmedLine = line.trim();
                            if (!trimmedLine) return null;
                            return (
                              <div key={lineIndex} className="mb-1 flex items-start">
                                {trimmedLine.startsWith('-') ? (
                                  <>
                                    <span className="text-gray-400 mr-2">•</span>
                                    <span>{trimmedLine.substring(1).trim()}</span>
                                  </>
                                ) : (
                                  <span>{trimmedLine}</span>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-400">
                    <p>No version history available.</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-gray-400">
          <p>&copy; 2025 Stock Sentiment Analyzer. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default Home;

