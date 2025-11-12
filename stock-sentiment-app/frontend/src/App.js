import React, { useState, useEffect } from 'react';
import { Search, TrendingUp, TrendingDown, Minus, AlertCircle, Clock, BarChart3, ExternalLink, LogOut, User, Heart, Star, Eye, Plus, X, LayoutDashboard } from 'lucide-react';
import { useAuth } from './AuthContext';
import Login from './Login';
import Register from './Register';
import SentimentChart from './SentimentChart';
import Home from './Home';
import Dashboard from './Dashboard';

const StockSentimentApp = () => {
  const { isAuthenticated, loading: authLoading, user, logout, getAuthHeaders } = useAuth();
  const [showRegister, setShowRegister] = useState(false);
  const [showFavorites, setShowFavorites] = useState(false);
  const [symbols, setSymbols] = useState(['']);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [favorites, setFavorites] = useState([]);
  const [favoritesLoading, setFavoritesLoading] = useState(false);
  const [watchlists, setWatchlists] = useState([]);
  const [watchlistsLoading, setWatchlistsLoading] = useState(false);
  const [showWatchlists, setShowWatchlists] = useState(false);
  const [selectedWatchlist, setSelectedWatchlist] = useState(null);
  const [watchlistStocks, setWatchlistStocks] = useState([]);
  const [showWatchlistModal, setShowWatchlistModal] = useState(false);
  const [newWatchlistName, setNewWatchlistName] = useState('');
  const [stockToAdd, setStockToAdd] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState({});
  const [showChartFor, setShowChartFor] = useState(null);
  const [showDashboard, setShowDashboard] = useState(true); // Show dashboard by default when authenticated
  const [showHome, setShowHome] = useState(true); // Show home by default when not authenticated

  // Define fetchFavorites and fetchWatchlists before useEffect
  const fetchFavorites = async () => {
    try {
      setFavoritesLoading(true);
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await fetch(`${apiUrl}/api/favorites`, {
        headers: {
          ...getAuthHeaders(),
        },
      });

      if (response.ok) {
        const data = await response.json();
        setFavorites(data.favorites || []);
      }
    } catch (error) {
      console.error('Error fetching favorites:', error);
    } finally {
      setFavoritesLoading(false);
    }
  };

  const fetchWatchlists = async () => {
    try {
      setWatchlistsLoading(true);
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await fetch(`${apiUrl}/api/watchlists`, {
        headers: {
          ...getAuthHeaders(),
        },
      });

      if (response.ok) {
        const data = await response.json();
        setWatchlists(data.watchlists || []);
      }
    } catch (error) {
      console.error('Error fetching watchlists:', error);
    } finally {
      setWatchlistsLoading(false);
    }
  };

  // Load analysis history from localStorage on mount
  useEffect(() => {
    try {
      const savedHistory = localStorage.getItem('analysisHistory');
      if (savedHistory) {
        setAnalysisHistory(JSON.parse(savedHistory));
      }
    } catch (error) {
      console.error('Error loading analysis history:', error);
    }
  }, []);

  // Save analysis history to localStorage whenever it changes
  useEffect(() => {
    try {
      localStorage.setItem('analysisHistory', JSON.stringify(analysisHistory));
    } catch (error) {
      console.error('Error saving analysis history:', error);
    }
  }, [analysisHistory]);

  // Fetch favorites and watchlists when authenticated - MUST be before any early returns
  useEffect(() => {
    if (isAuthenticated) {
      fetchFavorites();
      fetchWatchlists();
      // Ensure dashboard is shown when authenticated
      setShowDashboard(true);
      setShowFavorites(false);
      setShowWatchlists(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated]);

  // Show loading state while checking authentication
  if (authLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  // Show home page or login/register if not authenticated
  if (!isAuthenticated) {
    if (showHome && !showRegister) {
      return <Home onGetStarted={() => setShowHome(false)} />;
    }
    if (showRegister) {
      return <Register onSwitchToLogin={() => setShowRegister(false)} />;
    }
    return <Login onSwitchToRegister={() => setShowRegister(true)} />;
  }

  // Define analyzeStocks before it's used
  const analyzeStocks = async (symbolsToAnalyze = null) => {
    // Ensure we always have an array
    const symbolsToUse = Array.isArray(symbolsToAnalyze) ? symbolsToAnalyze : (Array.isArray(symbols) ? symbols : []);
    const validSymbols = symbolsToUse.filter(s => s && s.trim && s.trim().length > 0);
    
    if (validSymbols.length === 0) {
      setError('Please enter at least one stock symbol');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      // API call - use localhost for local dev, Render URL for production
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      
      // Get auth headers
      const authHeaders = getAuthHeaders();
      
      // Create AbortController for timeout (2 minutes)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minutes
      
      const response = await fetch(`${apiUrl}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders,
        },
        body: JSON.stringify({ symbols: validSymbols }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        // Handle unauthorized (401) - token expired or invalid
        if (response.status === 401) {
          logout();
          setError('Session expired. Please log in again.');
          return;
        }
        // Handle 422 - JWT validation error
        if (response.status === 422) {
          const errorData = await response.json().catch(() => ({}));
          console.error('422 Error details:', errorData);
          
          // Check if it's the "Subject must be a string" error
          if (errorData.msg && errorData.msg.includes('Subject must be a string')) {
            setError('Your session token is outdated. Logging you out...');
            // Immediately logout to get a fresh token
            logout();
            return;
          }
        }
        
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        setError(errorData.error || `Error: ${response.status} ${response.statusText}`);
        return;
      }
      
      const data = await response.json();
      setResults(data);
      
      // Save to analysis history
      if (data.results && Array.isArray(data.results)) {
        const today = new Date().toDateString();
        setAnalysisHistory(prevHistory => {
          const newHistory = { ...prevHistory };
          data.results.forEach(stock => {
            if (!newHistory[stock.symbol]) {
              newHistory[stock.symbol] = [];
            }
            
            // Check if there's already an entry for today with the same sentiment and confidence
            const existingToday = newHistory[stock.symbol].find(entry => {
              const entryDate = new Date(entry.timestamp).toDateString();
              return entryDate === today &&
                     Math.abs(entry.sentiment - stock.sentiment.overall_score) < 0.001 &&
                     Math.abs(entry.confidence - stock.sentiment.confidence) < 0.001;
            });
            
            // Only add if it's not a duplicate
            if (!existingToday) {
              const timestamp = new Date().toISOString();
              newHistory[stock.symbol].push({
                timestamp,
                sentiment: stock.sentiment.overall_score,
                confidence: stock.sentiment.confidence,
                grade: stock.sentiment.grade,
              });
              // Keep only last 30 entries per stock
              if (newHistory[stock.symbol].length > 30) {
                newHistory[stock.symbol] = newHistory[stock.symbol].slice(-30);
              }
            }
          });
          return newHistory;
        });
      }
      
      // Refresh favorites to show correct heart state
      fetchFavorites();
    } catch (error) {
      if (error.name === 'AbortError') {
        setError('Request timed out. Please try again.');
      } else if (error.message.includes('Failed to fetch')) {
        setError('Cannot connect to backend server. Please ensure the backend is running.');
      } else {
        setError(`Error: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  // Show dashboard if authenticated and showDashboard is true
  if (showDashboard) {
    return (
      <Dashboard
        onStartAnalyzing={(symbols) => {
          setShowDashboard(false);
          if (symbols && symbols.length > 0) {
            setSymbols(symbols);
            setTimeout(async () => {
              await analyzeStocks(symbols);
            }, 50);
          }
        }}
        favorites={favorites}
        watchlists={watchlists}
        analysisHistory={analysisHistory}
      />
    );
  }

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

  const toggleFavorite = async (symbol, e) => {
    // Prevent event propagation to avoid triggering parent click handlers
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    
    console.log('Toggling favorite for:', symbol);
    const isFavorited = favorites.some(fav => fav.symbol === symbol);
    const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

    try {
      if (isFavorited) {
        // Remove from favorites
        console.log('Removing from favorites:', symbol);
        const response = await fetch(`${apiUrl}/api/favorites/${symbol}`, {
          method: 'DELETE',
          headers: {
            ...getAuthHeaders(),
          },
        });

        if (response.ok) {
          setFavorites(favorites.filter(fav => fav.symbol !== symbol));
          console.log('Removed from favorites successfully');
        } else {
          const errorData = await response.json().catch(() => ({}));
          console.error('Failed to remove favorite:', errorData);
        }
      } else {
        // Add to favorites
        console.log('Adding to favorites:', symbol);
        const response = await fetch(`${apiUrl}/api/favorites`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...getAuthHeaders(),
          },
          body: JSON.stringify({ symbol }),
        });

        if (response.ok) {
          const data = await response.json();
          setFavorites([...favorites, data.favorite]);
          console.log('Added to favorites successfully:', data.favorite);
        } else {
          const errorData = await response.json().catch(() => ({}));
          console.error('Failed to add favorite:', errorData);
        }
      }
    } catch (error) {
      console.error('Error toggling favorite:', error);
    }
  };

  const isFavorite = (symbol) => {
    return favorites.some(fav => fav.symbol === symbol);
  };

  const analyzeFromFavorite = async (symbol) => {
    setShowFavorites(false);
    setResults(null);
    setError('');
    setSymbols([symbol]);
    // Wait for state to update, then analyze
    setTimeout(async () => {
      await analyzeStocks([symbol]);
    }, 50);
  };

  // Watchlist functions
  const createWatchlist = async (name) => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await fetch(`${apiUrl}/api/watchlists`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders(),
        },
        body: JSON.stringify({ name }),
      });

      if (response.ok) {
        const data = await response.json();
        setWatchlists([...watchlists, data.watchlist]);
        return data.watchlist;
      } else {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to create watchlist');
      }
    } catch (error) {
      console.error('Error creating watchlist:', error);
      throw error;
    }
  };

  const addStockToWatchlist = async (watchlistId, symbol) => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await fetch(`${apiUrl}/api/watchlists/${watchlistId}/stocks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders(),
        },
        body: JSON.stringify({ symbol }),
      });

      if (response.ok) {
        return true;
      } else {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to add stock to watchlist');
      }
    } catch (error) {
      console.error('Error adding stock to watchlist:', error);
      throw error;
    }
  };

  const fetchWatchlistStocks = async (watchlistId) => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await fetch(`${apiUrl}/api/watchlists/${watchlistId}/stocks`, {
        headers: {
          ...getAuthHeaders(),
        },
      });

      if (response.ok) {
        const data = await response.json();
        setWatchlistStocks(data.stocks || []);
        setSelectedWatchlist(data.watchlist);
      }
    } catch (error) {
      console.error('Error fetching watchlist stocks:', error);
    }
  };

  const handleAddToWatchlist = (symbol) => {
    setStockToAdd(symbol);
    setShowWatchlistModal(true);
    fetchWatchlists();
  };

  const handleWatchlistSelect = async (watchlistId, createNew = false) => {
    try {
      let targetWatchlistId = watchlistId;
      
      if (createNew) {
        if (!newWatchlistName.trim()) {
          alert('Please enter a watchlist name');
          return;
        }
        const newWatchlist = await createWatchlist(newWatchlistName.trim());
        targetWatchlistId = newWatchlist.id;
        setNewWatchlistName('');
      }

      await addStockToWatchlist(targetWatchlistId, stockToAdd);
      setShowWatchlistModal(false);
      setStockToAdd(null);
      // Refresh watchlists
      fetchWatchlists();
    } catch (error) {
      alert(error.message || 'Failed to add stock to watchlist');
    }
  };

  const analyzeFromWatchlist = async (symbol) => {
    setShowWatchlists(false);
    setResults(null);
    setError('');
    setSymbols([symbol]);
    setTimeout(async () => {
      await analyzeStocks([symbol]);
    }, 50);
  };

  const clearStockHistory = (symbol) => {
    if (window.confirm(`Are you sure you want to clear all history for ${symbol}?`)) {
      setAnalysisHistory(prevHistory => {
        const newHistory = { ...prevHistory };
        delete newHistory[symbol];
        return newHistory;
      });
    }
  };

  const clearAllHistory = () => {
    if (window.confirm('Are you sure you want to clear all analysis history? This cannot be undone.')) {
      setAnalysisHistory({});
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header with user info and logout */}
        <div className="flex items-center justify-between mb-8">
          <div className="text-center flex-1">
            <h1 className="text-4xl font-bold mb-2">Stock Sentiment Analyzer</h1>
            <p className="text-gray-400">AI-powered sentiment analysis for up to 5 stocks</p>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => {
                setShowDashboard(true);
                setShowFavorites(false);
                setShowWatchlists(false);
              }}
              className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              title="Go to Dashboard"
            >
              <LayoutDashboard className="w-4 h-4" />
              Dashboard
            </button>
            <button
              onClick={() => {
                setShowFavorites(!showFavorites);
                if (!showFavorites) {
                  fetchFavorites();
                }
                setShowWatchlists(false);
                setShowDashboard(false);
              }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                showFavorites 
                  ? 'bg-blue-600 hover:bg-blue-700' 
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              <Star className="w-4 h-4" />
              Favorites
            </button>
            <button
              onClick={() => {
                setShowWatchlists(!showWatchlists);
                if (!showWatchlists) {
                  fetchWatchlists();
                }
                setShowFavorites(false);
                setShowDashboard(false);
              }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                showWatchlists 
                  ? 'bg-blue-600 hover:bg-blue-700' 
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              <Eye className="w-4 h-4" />
              Watchlists
            </button>
            <div className="flex items-center gap-2 text-gray-300">
              <User className="w-5 h-5" />
              <span className="text-sm">{user?.email}</span>
            </div>
            <button
              onClick={clearAllHistory}
              className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              title="Clear all analysis history"
            >
              <X className="w-4 h-4" />
              Clear History
            </button>
            <button
              onClick={logout}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
        </div>

        {/* Watchlist Modal */}
        {showWatchlistModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">Add to Watchlist</h3>
                <button
                  onClick={() => {
                    setShowWatchlistModal(false);
                    setStockToAdd(null);
                    setNewWatchlistName('');
                  }}
                  className="text-gray-400 hover:text-white"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Create New Watchlist</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newWatchlistName}
                    onChange={(e) => setNewWatchlistName(e.target.value)}
                    placeholder="Watchlist name"
                    className="flex-1 px-3 py-2 bg-gray-700 rounded-lg text-white"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && newWatchlistName.trim()) {
                        handleWatchlistSelect(null, true);
                      }
                    }}
                  />
                  <button
                    onClick={() => handleWatchlistSelect(null, true)}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2"
                  >
                    <Plus className="w-4 h-4" />
                    Create
                  </button>
                </div>
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Or Select Existing</label>
                <div className="max-h-60 overflow-y-auto space-y-2">
                  {watchlists.length === 0 ? (
                    <p className="text-gray-400 text-sm text-center py-4">No watchlists yet</p>
                  ) : (
                    watchlists.map((watchlist) => (
                      <button
                        key={watchlist.id}
                        onClick={() => handleWatchlistSelect(watchlist.id)}
                        className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-left flex items-center justify-between"
                      >
                        <span>{watchlist.name}</span>
                        <span className="text-xs text-gray-400">{watchlist.stock_count} stocks</span>
                      </button>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Watchlists View */}
        {showWatchlists && (
          <div className="max-w-6xl mx-auto mb-8">
            <div className="bg-gray-800 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Your Watchlists</h2>
                <button
                  onClick={() => setShowWatchlists(false)}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
                >
                  Back to Analysis
                </button>
              </div>
              
              {watchlistsLoading ? (
                <div className="text-center py-8">
                  <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                  <p className="text-gray-400">Loading watchlists...</p>
                </div>
              ) : selectedWatchlist ? (
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold">{selectedWatchlist.name}</h3>
                      <p className="text-sm text-gray-400">{watchlistStocks.length} stocks</p>
                    </div>
                    <button
                      onClick={() => {
                        setSelectedWatchlist(null);
                        setWatchlistStocks([]);
                      }}
                      className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
                    >
                      Back to Watchlists
                    </button>
                  </div>
                  
                  {watchlistStocks.length === 0 ? (
                    <div className="text-center py-8">
                      <Eye className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                      <p className="text-gray-400">No stocks in this watchlist yet.</p>
                    </div>
                  ) : (
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                      {watchlistStocks.map((stock) => (
                        <button
                          key={stock.id}
                          onClick={() => analyzeFromWatchlist(stock.symbol)}
                          className="bg-gray-700 hover:bg-gray-600 rounded-lg p-4 text-center transition-colors group"
                        >
                          <div className="text-2xl font-bold mb-2 group-hover:text-blue-400 transition-colors">
                            {stock.symbol}
                          </div>
                          <div className="text-xs text-gray-400">Click to analyze</div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ) : watchlists.length === 0 ? (
                <div className="text-center py-8">
                  <Eye className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400">No watchlists yet. Click the eye icon on any stock analysis to create one.</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {watchlists.map((watchlist) => (
                    <button
                      key={watchlist.id}
                      onClick={() => fetchWatchlistStocks(watchlist.id)}
                      className="bg-gray-700 hover:bg-gray-600 rounded-lg p-4 text-left transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold">{watchlist.name}</h3>
                        <Eye className="w-5 h-5 text-gray-400" />
                      </div>
                      <p className="text-sm text-gray-400">{watchlist.stock_count} stocks</p>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Favorites View */}
        {showFavorites && (
          <div className="max-w-4xl mx-auto mb-8">
            <div className="bg-gray-800 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Your Favorite Stocks</h2>
                <button
                  onClick={() => setShowFavorites(false)}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
                >
                  Back to Analysis
                </button>
              </div>
              
              {favoritesLoading ? (
                <div className="text-center py-8">
                  <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                  <p className="text-gray-400">Loading favorites...</p>
                </div>
              ) : favorites.length === 0 ? (
                <div className="text-center py-8">
                  <Star className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400">No favorite stocks yet. Click the star icon on any stock analysis to add it to favorites.</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                  {favorites.map((fav) => (
                    <button
                      key={fav.id}
                      onClick={() => analyzeFromFavorite(fav.symbol)}
                      className="bg-gray-700 hover:bg-gray-600 rounded-lg p-4 text-center transition-colors group"
                    >
                      <div className="text-2xl font-bold mb-2 group-hover:text-blue-400 transition-colors">
                        {fav.symbol}
                      </div>
                      <div className="text-xs text-gray-400">Click to analyze</div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Main Analysis View */}
        {!showFavorites && !showWatchlists && (
          <>
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
                <div key={stock.symbol} className="bg-gray-800 rounded-lg p-6 relative">
                  {/* Star icon for favorites */}
                  <button
                    onClick={(e) => toggleFavorite(stock.symbol, e)}
                    className="absolute top-4 right-4 p-2 hover:bg-gray-700 rounded-full transition-colors z-10"
                    title={isFavorite(stock.symbol) ? 'Remove from favorites' : 'Add to favorites'}
                    type="button"
                  >
                    <Star 
                      className={`w-5 h-5 transition-colors ${
                        isFavorite(stock.symbol) 
                          ? 'fill-yellow-500 text-yellow-500' 
                          : 'text-gray-400 hover:text-yellow-400'
                      }`}
                    />
                  </button>
                  
                  {/* Eye icon for watchlist */}
                  <button
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      handleAddToWatchlist(stock.symbol);
                    }}
                    className="absolute top-4 right-16 p-2 hover:bg-gray-700 rounded-full transition-colors z-10"
                    title="Add to watchlist"
                    type="button"
                  >
                    <Eye className="w-5 h-5 text-gray-400 hover:text-blue-400 transition-colors" />
                  </button>
                  
                  {/* Chart icon for sentiment trend */}
                  <button
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      setShowChartFor(showChartFor === stock.symbol ? null : stock.symbol);
                    }}
                    className="absolute top-4 right-28 p-2 hover:bg-gray-700 rounded-full transition-colors z-10"
                    title="View sentiment trend"
                    type="button"
                  >
                    <BarChart3 className={`w-5 h-5 transition-colors ${
                      showChartFor === stock.symbol 
                        ? 'text-blue-500' 
                        : 'text-gray-400 hover:text-blue-400'
                    }`} />
                  </button>
                  
                  <div className="flex items-center justify-between mb-4 pr-36">
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
                      <div className="mb-3">
                        <p className="text-xs text-gray-400 mb-1">Keywords:</p>
                        <div className="flex flex-wrap gap-2">
                          {stock.keywords.map((keyword, i) => (
                            <span key={i} className="px-2 py-1 bg-gray-700 rounded text-xs">
                              {keyword}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {stock.news_analysis?.sources && stock.news_analysis.sources.length > 0 && (
                      <div>
                        <p className="text-xs text-gray-400 mb-1">Sources:</p>
                        <div className="flex flex-wrap gap-2">
                          {stock.news_analysis.sources.map((source, i) => (
                            <a
                              key={i}
                              href={source.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs flex items-center gap-1 transition-colors"
                            >
                              {source.name}
                              <ExternalLink className="w-3 h-3" />
                            </a>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Sentiment Chart Section */}
                  {showChartFor === stock.symbol && (
                    <div className="mt-6 pt-6 border-t border-gray-700">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="font-semibold text-gray-300">Sentiment History</h4>
                        <div className="flex gap-2">
                          <button
                            onClick={() => clearStockHistory(stock.symbol)}
                            className="px-3 py-1.5 bg-red-600 hover:bg-red-700 rounded-lg text-sm transition-colors flex items-center gap-1"
                            title="Clear history for this stock"
                          >
                            <X className="w-4 h-4" />
                            Clear {stock.symbol} History
                          </button>
                        </div>
                      </div>
                      <SentimentChart 
                        history={analysisHistory[stock.symbol] || []} 
                        symbol={stock.symbol}
                      />
                    </div>
                  )}

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
          </>
        )}
      </div>
    </div>
  );
};

export default StockSentimentApp;