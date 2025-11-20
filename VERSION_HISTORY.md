# Version History

## v1.7.0 - UI Navigation Improvements
- Moved version history section from dashboard to landing page for better visibility
- Added back button to dashboard to navigate back to landing page
- Improved navigation flow between landing page and dashboard
- Version history now accessible to all users (authenticated and unauthenticated) on landing page

## v1.6.0 - Enhanced Database Persistence & Data Loss Prevention
- Fixed database initialization code to use correct SQLAlchemy API (inspect instead of deprecated has_table)
- Added comprehensive logging to track database connections and detect data loss
- Enhanced error messages to help diagnose database persistence issues on Render
- Added detailed database connection logging (host, database name) for troubleshooting
- Created comprehensive guide (RENDER_DATA_PERSISTENCE.md) for ensuring data persistence on Render
- Improved data loss detection with clear warnings and actionable error messages

## v1.5.0 - Chart Time-Based Spacing & Version History Display
- Fixed chart x-axis to use time-based spacing instead of equal spacing
- Data points now reflect actual time differences (e.g., 1 day apart = close together, 4 days apart = larger gap)
- Added chartjs-adapter-date-fns for proper time scale support
- Chart spacing is now proportional to real time differences between analyses
- Added version history section to dashboard with collapsible display
- Version history automatically fetched from backend API endpoint

## v1.4.0 - Database Persistence Improvements
- Added database initialization logging and warnings
- Added detection for data loss on deployment
- Improved error handling and logging for database operations
- Added documentation for Render database persistence setup

## v1.3.0 - Timezone Fixes
- Fixed timestamp display to show dates in user's local timezone
- Ensured UTC timestamps are properly converted for display
- Fixed issue where dates showed as next day due to timezone conversion

## v1.2.0 - Chart Enhancements
- Expanded sentiment score range from -0.5/0.5 to -1.2/1.2 to show full range
- Increased chart height from 400px to 500px for better visibility
- Added horizontal grid lines every 0.1 increment for both sentiment and confidence axes
- Improved chart readability and visual appeal

## v1.1.0 - Backend Analysis History Storage
- Moved analysis history from localStorage to backend database
- Analysis history now syncs across all devices and browser windows
- Added AnalysisHistory model to store sentiment data with timestamps
- Automatic saving of analysis results to database
- Added GET and DELETE endpoints for analysis history management

## v1.0.0 - Dashboard Implementation
- Added user dashboard with stats (Total Analyses, Stocks Tracked, Favorites, Watchlists)
- Recent analyses section showing last 5 analyzed stocks
- Quick actions to start analyzing stocks
- Dashboard navigation and logout functionality

