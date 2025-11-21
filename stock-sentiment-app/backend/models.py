from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime, timezone

db = SQLAlchemy()
bcrypt = Bcrypt()

# Association table for user favorites
user_favorites = db.Table('user_favorites',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('stock_symbol', db.String(10), primary_key=True),
    db.Column('created_at', db.DateTime, default=datetime.utcnow)
)

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationship to favorites
    favorite_stocks = db.relationship('FavoriteStock', backref='user', lazy=True, cascade='all, delete-orphan')
    # Relationship to analysis history
    analysis_history = db.relationship('AnalysisHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set the user's password"""
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        """Check if the provided password matches the hash"""
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user to dictionary (exclude sensitive data)"""
        return {
            'id': self.id,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class FavoriteStock(db.Model):
    __tablename__ = 'favorite_stocks'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Unique constraint on user_id and symbol
    __table_args__ = (db.UniqueConstraint('user_id', 'symbol', name='unique_user_stock'),)
    
    def to_dict(self):
        """Convert favorite stock to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Watchlist(db.Model):
    __tablename__ = 'watchlists'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to stocks
    stocks = db.relationship('WatchlistStock', backref='watchlist', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert watchlist to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'stock_count': len(self.stocks)
        }

class WatchlistStock(db.Model):
    __tablename__ = 'watchlist_stocks'
    
    id = db.Column(db.Integer, primary_key=True)
    watchlist_id = db.Column(db.Integer, db.ForeignKey('watchlists.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Unique constraint on watchlist_id and symbol
    __table_args__ = (db.UniqueConstraint('watchlist_id', 'symbol', name='unique_watchlist_stock'),)
    
    def to_dict(self):
        """Convert watchlist stock to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class AnalysisHistory(db.Model):
    __tablename__ = 'analysis_history'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    sentiment = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    grade = db.Column(db.String(10), nullable=False)
    
    # Index for faster queries
    __table_args__ = (
        db.Index('idx_user_symbol_timestamp', 'user_id', 'symbol', 'timestamp'),
    )
    
    def to_dict(self):
        """Convert analysis history entry to dictionary"""
        # Ensure timestamp is timezone-aware and includes 'Z' suffix for UTC
        timestamp_str = None
        if self.timestamp:
            if self.timestamp.tzinfo is None:
                # If timezone-naive, assume it's UTC and make it timezone-aware
                timestamp_aware = self.timestamp.replace(tzinfo=timezone.utc)
                timestamp_str = timestamp_aware.isoformat().replace('+00:00', 'Z')
            else:
                # Already timezone-aware, convert to UTC if needed
                timestamp_aware = self.timestamp.astimezone(timezone.utc)
                timestamp_str = timestamp_aware.isoformat().replace('+00:00', 'Z')
        
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': timestamp_str,
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'grade': self.grade
        }

