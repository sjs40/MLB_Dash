
"""Configuration constants for the baseball dashboard."""

from datetime import datetime

# Current season
CURRENT_SEASON = datetime.now().year

# Stat categories
BATTING_STATS = ['G', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS']
PITCHING_STATS = ['G', 'GS', 'W', 'L', 'SV', 'IP', 'H', 'ER', 'HR', 'BB', 'SO', 'ERA', 'WHIP', 'K/9', 'BB/9']
ADVANCED_BATTING = ['wRC+', 'wOBA', 'ISO', 'BABIP', 'BB%', 'K%', 'WAR']
ADVANCED_PITCHING = ['ERA+', 'FIP', 'xFIP', 'SIERA', 'K%', 'BB%', 'WAR']

# Rolling windows (days)
ROLLING_WINDOWS = [7, 14, 30]

# Min plate appearanced for qualified batters
MIN_PA = 50

# Min innings pitched for qualified pitchers
MIN_IP = 20

# Team Abbreviations
