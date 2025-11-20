import React, { useState, useEffect } from 'react';
import { TrendingUp, Star, Eye, BarChart3, ArrowRight, Search, LogOut, ArrowLeft } from 'lucide-react';
import { useAuth } from './AuthContext';

const Dashboard = ({ onStartAnalyzing, favorites, watchlists, analysisHistory, onBackToHome }) => {
  const { user, logout } = useAuth();
  const [recentStocks, setRecentStocks] = useState([]);

  useEffect(() => {
    // Get recent stocks from analysis history
    const historyEntries = Object.entries(analysisHistory || {})
      .map(([symbol, data]) => ({
        symbol,
        lastAnalyzed: data.length > 0 ? new Date(data[data.length - 1].timestamp) : null,
        dataPoints: data.length
      }))
      .filter(entry => entry.lastAnalyzed)
      .sort((a, b) => b.lastAnalyzed - a.lastAnalyzed)
      .slice(0, 5);

    setRecentStocks(historyEntries);
  }, [analysisHistory]);

  const totalAnalyses = Object.values(analysisHistory || {}).reduce((sum, data) => sum + data.length, 0);
  const uniqueStocks = Object.keys(analysisHistory || {}).length;

  const stats = [
    {
      label: 'Total Analyses',
      value: totalAnalyses,
      icon: <BarChart3 className="w-6 h-6" />,
      color: 'text-blue-400'
    },
    {
      label: 'Stocks Tracked',
      value: uniqueStocks,
      icon: <TrendingUp className="w-6 h-6" />,
      color: 'text-green-400'
    },
    {
      label: 'Favorites',
      value: favorites?.length || 0,
      icon: <Star className="w-6 h-6" />,
      color: 'text-yellow-400'
    },
    {
      label: 'Watchlists',
      value: watchlists?.length || 0,
      icon: <Eye className="w-6 h-6" />,
      color: 'text-purple-400'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {onBackToHome && (
                <button
                  onClick={onBackToHome}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                  title="Back to Home"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back
                </button>
              )}
              <div>
                <h1 className="text-3xl font-bold">Welcome back, {user?.email?.split('@')[0] || 'User'}!</h1>
                <p className="text-gray-400 mt-1">Here's your stock sentiment analysis overview</p>
              </div>
            </div>
            <button
              onClick={logout}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
              title="Logout"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => (
            <div
              key={index}
              className="bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-gray-600 transition-all"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`${stat.color}`}>{stat.icon}</div>
              </div>
              <div className="text-3xl font-bold mb-1">{stat.value}</div>
              <div className="text-gray-400 text-sm">{stat.label}</div>
            </div>
          ))}
        </div>

        {/* Quick Actions */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8 border border-gray-700">
          <h2 className="text-2xl font-bold mb-4">Quick Actions</h2>
          <button
            onClick={onStartAnalyzing}
            className="w-full md:w-auto px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-all flex items-center justify-center gap-2"
          >
            <Search className="w-5 h-5" />
            Start Analyzing Stocks
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Recent Analyses */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-2xl font-bold mb-4">Recent Analyses</h2>
            {recentStocks.length > 0 ? (
              <div className="space-y-3">
                {recentStocks.map((stock, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-gray-700 rounded-lg hover:bg-gray-600 transition-all cursor-pointer"
                    onClick={() => onStartAnalyzing([stock.symbol])}
                  >
                    <div>
                      <div className="font-semibold">{stock.symbol}</div>
                      <div className="text-sm text-gray-400">
                        {stock.dataPoints} data point{stock.dataPoints !== 1 ? 's' : ''} •{' '}
                        {stock.lastAnalyzed.toLocaleDateString()}
                      </div>
                    </div>
                    <ArrowRight className="w-5 h-5 text-gray-400" />
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400">
                <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No analyses yet. Start analyzing stocks to see your history here!</p>
              </div>
            )}
          </div>

          {/* Favorites & Watchlists */}
          <div className="space-y-6">
            {/* Favorites */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <Star className="w-6 h-6 text-yellow-400" />
                Favorite Stocks
              </h2>
              {favorites && favorites.length > 0 ? (
                <div className="space-y-2">
                  {favorites.slice(0, 5).map((fav, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 bg-gray-700 rounded-lg hover:bg-gray-600 transition-all cursor-pointer"
                      onClick={() => onStartAnalyzing([fav.symbol])}
                    >
                      <span className="font-semibold">{fav.symbol}</span>
                      <ArrowRight className="w-5 h-5 text-gray-400" />
                    </div>
                  ))}
                  {favorites.length > 5 && (
                    <p className="text-sm text-gray-400 text-center mt-2">
                      +{favorites.length - 5} more favorite{favorites.length - 5 !== 1 ? 's' : ''}
                    </p>
                  )}
                </div>
              ) : (
                <div className="text-center py-6 text-gray-400">
                  <p>No favorites yet. Star stocks to add them here!</p>
                </div>
              )}
            </div>

            {/* Watchlists */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <Eye className="w-6 h-6 text-purple-400" />
                Your Watchlists
              </h2>
              {watchlists && watchlists.length > 0 ? (
                <div className="space-y-2">
                  {watchlists.slice(0, 3).map((watchlist, index) => (
                    <div
                      key={index}
                      className="p-3 bg-gray-700 rounded-lg hover:bg-gray-600 transition-all"
                    >
                      <div className="font-semibold">{watchlist.name}</div>
                      <div className="text-sm text-gray-400">
                        {watchlist.stock_count || 0} stock{watchlist.stock_count !== 1 ? 's' : ''}
                      </div>
                    </div>
                  ))}
                  {watchlists.length > 3 && (
                    <p className="text-sm text-gray-400 text-center mt-2">
                      +{watchlists.length - 3} more watchlist{watchlists.length - 3 !== 1 ? 's' : ''}
                    </p>
                  )}
                </div>
              ) : (
                <div className="text-center py-6 text-gray-400">
                  <p>No watchlists yet. Create one to organize your stocks!</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

