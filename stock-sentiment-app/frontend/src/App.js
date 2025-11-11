import React, { useState } from 'react';
import { Search, TrendingUp, TrendingDown, Minus, AlertCircle, Clock, BarChart3 } from 'lucide-react';

const StockSentimentApp = () => {
  const [symbols, setSymbols] = useState(['']);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  const addSymbolInput = () => {
    if (symbols.length < 5) {
      setSymbols([...symbols, '']);
    }
  };

  const removeSymbolInput = (index) => {
    if (symbols.length > 1) {
      const newSymbols = symbols.filter((_, i) => i !== index);
      setSymbols(newSymbols);
    }
  };

  const updateSymbol = (index, value) => {
    const newSymbols = [...symbols];
    newSymbols[index] = value.toUpperCase();
    setSymbols(newSymbols);
  };

  const analyzeStocks = async () => {
    const validSymbols = symbols.filter(s => s.trim().length > 0);
    
    if (validSymbols.length === 0) {
      setError('Please enter at least one stock symbol');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      // API call to Render backend
      const apiUrl = process.env.REACT_APP_API_URL || 'https://stocksentimentanalyzer.onrender.com';
      const response = await fetch(`${apiUrl}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symbols: validSymbols })
      });
      
      if (!response.ok) {
        throw new Error('Analysis failed');
      }
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError('Failed to analyze stocks. Please check your backend connection.');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (score) => {
    if (score > 0.3) return 'text-green-500';
    if (score > 0.1) return 'text-green-400';
    if (score > -0.1) return 'text-yellow-500';
    if (score > -0.3) return 'text-red-400';
    return 'text-red-500';
  };

  const getSentimentIcon = (score) => {
    if (score > 0.1) return <TrendingUp className="w-5 h-5 text-green-500" />;
    if (score < -0.1) return <TrendingDown className="w-5 h-5 text-red-500" />;
    return <Minus className="w-5 h-5 text-yellow-500" />;
  };

  const getGradeColor = (grade) => {
    if (['A+', 'A'].includes(grade)) return 'bg-green-500';
    if (['B+', 'B'].includes(grade)) return 'bg-blue-500';
    if (grade === 'C') return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">Stock Sentiment Analyzer</h1>
          <p className="text-gray-400">AI-powered sentiment analysis for up to 5 stocks</p>
        </div>

        <div className="max-w-2xl mx-auto mb-8">
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Enter Stock Symbols</h2>
            
            {symbols.map((symbol, index) => (
              <div key={index} className="flex gap-2 mb-3">
                <input
                  type="text"
                  value={symbol}
                  onChange={(e) => updateSymbol(index, e.target.value)}
                  placeholder="Enter symbol (e.g., AAPL)"
                  className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:border-blue-500 focus:outline-none"
                  maxLength={10}
                />
                {symbols.length > 1 && (
                  <button
                    onClick={() => removeSymbolInput(index)}
                    className="px-3 py-2 bg-red-600 hover:bg-red-700 rounded-lg"
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
            
            <div className="flex gap-3 mt-4">
              {symbols.length < 5 && (
                <button
                  onClick={addSymbolInput}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
                >
                  Add Symbol
                </button>
              )}
              
              <button
                onClick={analyzeStocks}
                disabled={loading}
                className="flex-1 flex items-center justify-center gap-2 px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    Analyze Sentiment
                  </>
                )}
              </button>
            </div>

            {error && (
              <div className="mt-4 p-3 bg-red-900 border border-red-700 rounded-lg flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <span className="text-red-200">{error}</span>
              </div>
            )}
          </div>
        </div>

        {results && (
          <div className="max-w-6xl mx-auto">
            <h2 className="text-2xl font-bold mb-6">Analysis Results</h2>
            
            <div className="grid gap-6">
              {results.results.map((stock, index) => (
                <div key={stock.symbol} className="bg-gray-800 rounded-lg p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-4">
                      <h3 className="text-2xl font-bold">{stock.symbol}</h3>
                      <span className={`px-3 py-1 rounded-full text-white text-sm font-semibold ${getGradeColor(stock.sentiment.grade)}`}>
                        {stock.sentiment.grade}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {getSentimentIcon(stock.sentiment.overall_score)}
                      <span className={`text-lg font-semibold ${getSentimentColor(stock.sentiment.overall_score)}`}>
                        {stock.sentiment.overall_score.toFixed(3)}
                      </span>
                    </div>
                  </div>

                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold text-gray-300 mb-2">Company Info</h4>
                      <div className="space-y-1 text-sm">
                        <p><span className="text-gray-400">Name:</span> {stock.stock_info.longName}</p>
                        <p><span className="text-gray-400">Sector:</span> {stock.stock_info.sector}</p>
                        <p><span className="text-gray-400">Price:</span> ${stock.stock_info.currentPrice.toFixed(2)}</p>
                        <p><span className="text-gray-400">Articles Analyzed:</span> {stock.news_analysis.article_count}</p>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold text-gray-300 mb-2">Analysis</h4>
                      <div className="space-y-1 text-sm">
                        <p><span className="text-gray-400">Confidence:</span> {(stock.sentiment.confidence * 100).toFixed(1)}%</p>
                        <p><span className="text-gray-400">Prediction:</span> 
                          <span className={stock.prediction.direction === 'bullish' ? 'text-green-400' : 
                                         stock.prediction.direction === 'bearish' ? 'text-red-400' : 'text-yellow-400'}>
                            {' ' + stock.prediction.direction}
                          </span>
                        </p>
                        <p><span className="text-gray-400">5-Day Expected:</span> 
                          <span className={stock.prediction.expected_return_5d > 0 ? 'text-green-400' : 'text-red-400'}>
                            {' ' + (stock.prediction.expected_return_5d * 100).toFixed(1)}%
                          </span>
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-4">
                    <h4 className="font-semibold text-gray-300 mb-2">Key Insights</h4>
                    <p className="text-sm text-gray-300 mb-2">{stock.sentiment.description}</p>
                    
                    {stock.keywords && stock.keywords.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {stock.keywords.map((keyword, i) => (
                          <span key={i} className="px-2 py-1 bg-gray-700 rounded text-xs">
                            {keyword}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="mt-4 pt-4 border-t border-gray-700">
                    <div className="flex items-center justify-between text-xs text-gray-400">
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        <span>Analysis completed</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <BarChart3 className="w-3 h-3" />
                        <span>Confidence: {(stock.sentiment.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-8 p-4 bg-gray-800 rounded-lg">
              <p className="text-xs text-gray-400 text-center">
                This analysis is for informational purposes only. Always conduct your own research before making investment decisions.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockSentimentApp;