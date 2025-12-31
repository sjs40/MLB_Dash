import streamlit as st
import pandas as pd
from pybaseball import batting_stats, pitching_stats, playerid_lookup, statcast_batter, statcast_pitcher, team_batting, team_pitching, standings
from datetime import datetime, timedelta
from config import CURRENT_SEASON, MIN_PA, MIN_IP

@st.cache_data(ttl=3600)
def get_batting_stats(season: int = CURRENT_SEASON, qual: int = MIN_PA) -> pd.DataFrame:
  """
  Fetch season batting stats from FanGraphs

  Args:
    season: The season year
    qual: Minimum plate appearances to qualify

  Returns:
    DataFrame with batting statistics
  """
  try:
    df = batting_stats(season, qual=qual)
    return df
  except Exception as e:
    st.error(f"Error fetching batting stats: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_pitching_stats(season: int = CURRENT_SEASON, qual: int = MIN_IP) -> pd.DataFrame:
  """
  Fetch season pitching stats from FanGraphs

  Args:
    season: The season year
    qual: Minimum innings pitched to qualify

  Returns:
    DataFrame with pitching statistics
  """
  try:
    df = pitching_stats(season, qual=qual)
    return df
  except Exception as e:
    st.error(f"Error fetching pitching stats: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_league_averages(season: int = CURRENT_SEASON) -> dict:
  """
  Calculate league average statistics for normalization

  Returns:
    Dictionary with league averages for key stats
  """
  batting = get_batting_stats(season, qual=1)
  pitching = get_pitching_stats(season, qual=1)

  if batting.empty or pitching.empty:
    return {}

  league_avgs = {
    'batting': {
      'AVG': batting['AVG'].mean(),
      'OBP': batting['OBP'].mean(),
      'SLG': batting['SLG'].mean(),
      'OPS': batting['OPS'].mean(),
      'wOBA': batting['wOBA'].mean() if 'wOBA' in batting.columns else None,
      'ISO': batting['ISO'].mean() if 'ISO' in batting.columns else None,
      'BABIP': batting['BABIP'].mean() if 'BABIP' in batting.columns else None,
      'K%': batting['K%'].mean() if 'K%' in batting.columns else None,
      'BB%': batting['BB%'].mean() if 'BB%' in batting.columns else None
    },
    'pitching': {
      'ERA': pitching['ERA'].mean(),
      'WHIP': pitching['WHIP'].mean(),
      'K/9': pitching['K/9'].mean() if 'K/9' in pitching.columns else None,
      'BB/9': pitching['BB/9'].mean() if 'BB/9' in pitching.columns else None,
      'FIP': pitching['FIP'].mean() if 'FIP' in pitching.columns else None
    }
  }
  return league_avgs

@st.cache_data(ttl=3600)
def get_statcast_batter_data(player_id: int, start_date: str, end_date: str) -> pd.DataFrame:
  """
  Fetch Statcast data for a specific batter.

  Args:
    player_id: MLB player ID
    start_date: Start date (YYYY-MM-DD)
    end_date: End date (YYYY-MM-DD)

  Returns:
    DataFrame with Statcast batting data
  """
  try:
    df = statcast_batter(start_date, end_date, player_id)
    return df
  except Exception as e:
    st.error(f"Error fetching Statcast data: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_statcast_pitcher_data(player_id: int, start_date: str, end_date: str) -> pd.DataFrame:
  """
  Fetch Statcast data for a specific pitcher.
  """
  try:
    df = statcast_pitcher(start_date, end_date, player_id)
    return df
  except Exception as e:
    st.error(f"Error fetching Statcast data: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=86400)
def lookup_player(first_name: str, last_name: str) -> pd.DataFrame:
  """
  Look up a player's ID by name
  """
  try:
    return playerid_lookup(first_name, last_name)
  except Exception as e:
    st.error(f"Error looking up player: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_team_stats(season: int = CURRENT_SEASON) -> tuple:
  """
  Fetch team-level batting and pitching statistics.

  Returns:
    Tuple of (team_batting_df, team_pitching_df)
  """
  try:
    batting = team_batting(season)
    pitching = team_pitching(season)
    return batting, pitching
  except Exception as e:
    st.error(f"Error fetching team stats: {e}")
    return pd.DataFrame(), pd.DataFrame()





