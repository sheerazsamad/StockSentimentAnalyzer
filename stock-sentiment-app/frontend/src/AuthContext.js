import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  const fetchUser = useCallback(async (authToken) => {
    try {
      const response = await fetch(`${apiUrl}/api/me`, {
        headers: {
          'Authorization': `Bearer ${authToken}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setUser(data.user);
      } else {
        // Check if it's the "Subject must be a string" error
        if (response.status === 422) {
          const errorData = await response.json().catch(() => ({}));
          console.log('422 error on /api/me:', errorData);
          if (errorData.msg && errorData.msg.includes('Subject must be a string')) {
            // Old token format - clear it immediately
            console.log('Clearing old token format');
            localStorage.removeItem('auth_token');
            setToken(null);
            setUser(null);
            setLoading(false);
            return;
          }
        }
        // Token is invalid, clear it
        localStorage.removeItem('auth_token');
        setToken(null);
        setUser(null);
      }
    } catch (error) {
      console.error('Error fetching user:', error);
      localStorage.removeItem('auth_token');
      setToken(null);
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  // Check for existing token on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('auth_token');
    if (storedToken) {
      setToken(storedToken);
      // Verify token and get user info
      fetchUser(storedToken);
    } else {
      setLoading(false);
    }
  }, [fetchUser]);

  // Update token state when localStorage changes (for cross-tab sync)
  useEffect(() => {
    const handleStorageChange = (e) => {
      if (e.key === 'auth_token') {
        if (e.newValue) {
          setToken(e.newValue);
        } else {
          setToken(null);
          setUser(null);
        }
      }
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  const login = async (email, password) => {
    try {
      const response = await fetch(`${apiUrl}/api/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      // Check if response is JSON
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        return { 
          success: false, 
          error: `Backend server error. Please ensure the backend is running on ${apiUrl}` 
        };
      }

      const data = await response.json();

      if (response.ok) {
        const token = data.access_token;
        console.log('Login successful, token received:', token ? 'Yes' : 'No');
        setToken(token);
        setUser(data.user);
        localStorage.setItem('auth_token', token);
        return { success: true };
      } else {
        return { success: false, error: data.error || 'Login failed' };
      }
    } catch (error) {
      console.error('Login error:', error);
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        return { 
          success: false, 
          error: `Cannot connect to backend server at ${apiUrl}. Please ensure the backend is running.` 
        };
      }
      return { success: false, error: `Network error: ${error.message}` };
    }
  };

  const register = async (email, password) => {
    try {
      console.log('Attempting to register at:', `${apiUrl}/api/register`);
      const response = await fetch(`${apiUrl}/api/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });
      console.log('Register response status:', response.status);

      // Check if response is JSON
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        return { 
          success: false, 
          error: `Backend server error. Please ensure the backend is running on ${apiUrl}` 
        };
      }

      const data = await response.json();

      if (response.ok) {
        setToken(data.access_token);
        setUser(data.user);
        localStorage.setItem('auth_token', data.access_token);
        return { success: true };
      } else {
        return { success: false, error: data.error || 'Registration failed' };
      }
    } catch (error) {
      console.error('Registration error:', error);
      console.error('Error details:', {
        message: error.message,
        name: error.name,
        stack: error.stack
      });
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError') || error.name === 'TypeError') {
        return { 
          success: false, 
          error: `Cannot connect to backend server at ${apiUrl}. Please ensure the backend is running on port 5000.` 
        };
      }
      return { success: false, error: `Network error: ${error.message}` };
    }
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('auth_token');
  };

  const getAuthHeaders = () => {
    const currentToken = token || localStorage.getItem('auth_token');
    if (!currentToken) {
      console.warn('No token available for auth headers');
      return {};
    }
    return {
      'Authorization': `Bearer ${currentToken}`,
    };
  };

  // Check authentication status - use token from state or localStorage
  const currentToken = token || localStorage.getItem('auth_token');
  const isAuthenticated = !!currentToken && !!user;

  const value = {
    user,
    token: currentToken, // Always return current token (from state or localStorage)
    loading,
    login,
    register,
    logout,
    isAuthenticated,
    getAuthHeaders,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

