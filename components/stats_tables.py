# components/stats_tables.py
"""Table display components for the baseball dashboard."""

import streamlit as st
import pandas as pd


def display_stat_card(label: str, value, delta=None, delta_color="normal"):
  """Display a single stat as a metric card."""
  if isinstance(value, float):
    value = round(value, 3)
  st.metric(label, value, delta=delta, delta_color=delta_color)


def display_stat_table(df: pd.DataFrame, title: str = None, height: int = 400):
  """Display a formatted stats table."""
  if title:
    st.subheader(title)
  st.dataframe(df, use_container_width=True, height=height)


def display_player_header(player_data: pd.Series, stat_type: str = "Batting"):
  """Display player header information."""
  col1, col2, col3, col4 = st.columns(4)

  with col1:
    st.metric("Player", player_data.get('Name', 'Unknown'))
  with col2:
    st.metric("Team", player_data.get('Team', 'N/A'))
  with col3:
    st.metric("Games", int(player_data.get('G', 0)))
  with col4:
    if stat_type == "Batting":
      st.metric("PA", int(player_data.get('PA', 0)))
    else:
      st.metric("IP", round(player_data.get('IP', 0), 1))


def display_key_batting_stats(player_data: pd.Series):
  """Display key batting statistics in a row."""
  cols = st.columns(5)

  stats = [
    ('AVG', 'Batting Avg'),
    ('OBP', 'On-Base %'),
    ('SLG', 'Slugging %'),
    ('OPS', 'OPS'),
    ('HR', 'Home Runs')
  ]

  for col, (stat, label) in zip(cols, stats):
    with col:
      value = player_data.get(stat, 0)
      if isinstance(value, float):
        value = round(value, 3)
      st.metric(label, value)


def display_key_pitching_stats(player_data: pd.Series):
  """Display key pitching statistics in a row."""
  cols = st.columns(5)

  stats = [
    ('ERA', 'ERA'),
    ('WHIP', 'WHIP'),
    ('W', 'Wins'),
    ('SO', 'Strikeouts'),
    ('K/9', 'K/9')
  ]

  for col, (stat, label) in zip(cols, stats):
    with col:
      value = player_data.get(stat, 0)
      if isinstance(value, float):
        value = round(value, 2)
      st.metric(label, value)


def display_detailed_batting_table(player_data: pd.Series):
  """Display detailed batting statistics in a table."""
  detail_stats = [
    'G', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'RBI',
    'SB', 'CS', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS'
  ]

  # Filter to available columns
  available_stats = [s for s in detail_stats if s in player_data.index]

  stat_df = pd.DataFrame({
    'Stat': available_stats,
    'Value': [player_data[s] for s in available_stats]
  }).set_index('Stat').T

  st.dataframe(stat_df, use_container_width=True)


def display_detailed_pitching_table(player_data: pd.Series):
  """Display detailed pitching statistics in a table."""
  detail_stats = [
    'G', 'GS', 'W', 'L', 'SV', 'HLD', 'IP', 'H',
    'ER', 'HR', 'BB', 'SO', 'ERA', 'WHIP', 'K/9', 'BB/9'
  ]

  # Filter to available columns
  available_stats = [s for s in detail_stats if s in player_data.index]

  stat_df = pd.DataFrame({
    'Stat': available_stats,
    'Value': [player_data[s] for s in available_stats]
  }).set_index('Stat').T

  st.dataframe(stat_df, use_container_width=True)


def display_rolling_stats_summary(rolling_stats: dict, stat_type: str = "Batting"):
  """Display rolling stats in columns."""
  if not rolling_stats:
    st.warning("No rolling stats available.")
    return

  cols = st.columns(len(rolling_stats))

  for col, (window, stats) in zip(cols, rolling_stats.items()):
    with col:
      st.markdown(f"**{window}**")

      if stat_type == "Batting":
        st.metric("AVG", f"{stats.get('AVG', 0):.3f}")
        st.metric("HR", stats.get('HR', 0))
        if stats.get('Avg Exit Velo'):
          st.metric("Exit Velo", f"{stats['Avg Exit Velo']:.1f}")
      else:
        st.metric("ERA", f"{stats.get('ERA', 0):.2f}")
        st.metric("K", stats.get('SO', 0))


def format_leaderboard(df: pd.DataFrame, stat: str, ascending: bool = False, top_n: int = 20) -> pd.DataFrame:
  """Format a leaderboard DataFrame."""
  # Sort by the stat
  sorted_df = df.nsmallest(top_n, stat) if ascending else df.nlargest(top_n, stat)

  # Reset index and add rank
  sorted_df = sorted_df.reset_index(drop=True)
  sorted_df.index = sorted_df.index + 1
  sorted_df.index.name = 'Rank'

  return sorted_df


def highlight_above_average(val, threshold=100):
  """Highlight values above a threshold."""
  if isinstance(val, (int, float)) and val >= threshold:
    return 'background-color: lightgreen'
  return ''
