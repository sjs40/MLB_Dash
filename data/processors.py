import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta

from config import ROLLING_WINDOWS

# Team abbreviation mappings between different data sources
# Statcast uses certain abbreviations, FanGraphs may use others
TEAM_ABBREV_MAP = {
  # Statcast abbrev -> Standard abbrev
  'AZ': 'ARI',
  'ARI': 'ARI',
  'ATL': 'ATL',
  'BAL': 'BAL',
  'BOS': 'BOS',
  'CHC': 'CHC',
  'CWS': 'CHW',
  'CHW': 'CHW',
  'CIN': 'CIN',
  'CLE': 'CLE',
  'COL': 'COL',
  'DET': 'DET',
  'HOU': 'HOU',
  'KC': 'KCR',
  'KCR': 'KCR',
  'LAA': 'LAA',
  'LAD': 'LAD',
  'MIA': 'MIA',
  'MIL': 'MIL',
  'MIN': 'MIN',
  'NYM': 'NYM',
  'NYY': 'NYY',
  'OAK': 'OAK',
  'PHI': 'PHI',
  'PIT': 'PIT',
  'SD': 'SDP',
  'SDP': 'SDP',
  'SF': 'SFG',
  'SFG': 'SFG',
  'SEA': 'SEA',
  'STL': 'STL',
  'TB': 'TBR',
  'TBR': 'TBR',
  'TEX': 'TEX',
  'TOR': 'TOR',
  'WSH': 'WSN',
  'WSN': 'WSN',
}

def standardize_team_abbrev(abbrev: str) -> str:
  """Convert team abbreviation to standard format."""
  if pd.isna(abbrev):
    return None
  return TEAM_ABBREV_MAP.get(abbrev.upper(), abbrev.upper())

def calculate_normalized_stats(player_stats: pd.Series, league_avgs: dict, stat_type: str = 'batting') -> dict:
  """
  Calculate league-normalized statistics (like OPS+, ERA+)

  The formula for "plus" stats: (League Average / Player Stat) * 100 for ERA-type stats
                                (Player Stat / League Average) * 100 for offensive stats

  Args:
    player_stats: Series of player stats
    league_avgs: Dictionary of league averages
    stat_type: 'batting' or 'pitching'

  Returns:
    Dictionary of normalized stats
  """
  normalized = {}

  if stat_type == 'batting':
    avgs = league_avgs.get('batting', {})

    # OPS+ calculation (simplified)
    if avgs.get('OPS') and player_stats.get('OPS'):
      normalized['OPS+'] = round((player_stats['OPS'] / avgs['OPS']) * 100, 1)

    # AVG+
    if avgs.get('AVG') and player_stats.get('AVG'):
      normalized['AVG+'] = round((player_stats['AVG'] / avgs['AVG']) * 100, 1)

    # ISO+
    if avgs.get('ISO') and player_stats.get('ISO'):
      normalized['ISO+'] = round((player_stats['ISO'] / avgs['ISO']) * 100, 1)

    # K%+ (lower is better for batters, so invert
    if avgs.get('K%') and player_stats.get('K%'):
      normalized['K%+'] = round((avgs['K%'] / player_stats['K%']) * 100, 1)

    # BB%+
    if avgs.get('BB%') and player_stats.get('BB%'):
      normalized['BB%+'] = round((player_stats['BB%'] / avgs['BB%']) * 100, 1)
  elif stat_type == 'pitching':
    avgs = league_avgs.get('pitching', {})

    # ERA+
    if avgs.get('ERA') and player_stats.get('ERA'):
      normalized['ERA+'] = round((avgs['ERA'] / player_stats['ERA']) * 100, 1)

    # FIP+
    if avgs.get('FIP') and player_stats.get('FIP'):
      normalized['FIP+'] = round((avgs['FIP'] / player_stats['FIP']) * 100, 1)

    # WHIP+
    if avgs.get('WHIP') and player_stats.get('WHIP'):
      normalized['WHIP+'] = round((avgs['WHIP'] / player_stats['WHIP']) * 100, 1)

    # K/9+
    if avgs.get('K/9') and player_stats.get('K/9'):
      normalized['K/9+'] = round((player_stats['K/9'] / avgs['K/9']) * 100, 1)

  return normalized

def calculate_percentile_ranks(player_value: float, all_values: pd.Series) -> int:
  """
  Calculate the percentile rank of a player's stats among all players.

  Args:
    player_value: The player's stat value
    all_values: Series of all player stats

  Returns:
    Percentile rank (0-100)
  """
  return int(round((all_values < player_value).mean() * 100))

def calculate_rolling_stats(statcast_df: pd.DataFrame, windows: list = ROLLING_WINDOWS) -> dict:
  """
  Calculate rolling statistics from Statcast data.

  Args:
    statcast_df: DataFrame with Statcast data
    windows: List of rolling window sizes (in days)

  Returns:
    Dictionary with rolling stats for each window
  """
  if statcast_df.empty:
    return {}

  # Ensure game_date is datetime
  statcast_df = statcast_df.copy()
  statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date'])

  rolling_stats = {}
  today = datetime.now()

  for window in windows:
    start_date = today - timedelta(days=window)
    window_df = statcast_df[statcast_df['game_date'] >= start_date]

    if window_df.empty:
      continue

    # Calculate batting stats for the window
    at_bats = len(window_df[window_df['events'].notna()])
    hits = len(window_df[window_df['events'].isin('single', 'double', 'triple', 'home run')])

    stats = {
      'PA': len(window_df),
      'AB': at_bats,
      'H': hits,
      'HR': len(window_df[window_df['events'] == 'home run']),
      'BB': len(window_df[window_df['events'] == 'walk']),
      'SO': len(window_df[window_df['events'] == 'strikeout']),
      'AVG': round(hits / at_bats, 3) if at_bats > 0 else 0,
    }

    # Statcast-specific metrics
    if 'launch_speed' in window_df.columns:
      batted_balls = window_df[window_df['launch_speed'].notna()]
      stats['Avg Exit Velo'] = round(batted_balls['launch_speed'].mean(), 1) if len(batted_balls) > 0 else None
      stats['Max Exit Velo'] = round(batted_balls['launch_speed'].max(), 1) if len(batted_balls) > 0 else None

    if 'launch_angle' in window_df.columns:
      batted_balls = window_df[window_df['launch_angle'].notna()]
      stats['Avg Launch Angle'] = round(batted_balls['launch_angle'].mean(), 1) if len(batted_balls) > 0 else None

    if 'barrel' in window_df.columns:
      batted_balls = window_df[window_df['barrel'].notna()]
      stats['Barrel%'] = round(batted_balls['barrel'].mean() * 100, 1) if len(batted_balls) > 0 else None

    rolling_stats[f'Last {window} Days'] = stats

  return rolling_stats

def categorize_opponent_quality(team_stats: pd.DataFrame, stat_col: str = 'ERA', reverse: bool = True) -> dict:
  """
  Categorize teams into quality tiers based on a statistic.

  Args:
    team_stats: DataFrame with team stats
    stat_col: Statistic to use for categorization
    reverse: If True, lower values = better (like ERA)

    Returns:
      Dictionary mappting team abbreviations to quality tiers
  """
  if team_stats.empty or stat_col not in team_stats.columns:
    return {}

  df = team_stats.copy()

  # Ensure we have a Team colum
  if 'Team' not in df.columns:
    if df.index.name == 'Team':
      df = df.reset_index()
    else:
      return {}

  # Sort teams by the stat
  df = df.sort_values(stat_col, ascending=reverse)

  # Divide into thirds
  n_teams = len(df)
  tier_size = n_teams // 3

  team_tiers = {}

  for i, (_, row) in enumerate(df.iterrows()):
    team = standardize_team_abbrev(row['Team'])
    if team is None:
      continue

    if i < tier_size:
      team_tiers[team] = 'Elite'
    elif i < 2 * tier_size:
      team_tiers[team] = 'Average'
    else:
      team_tiers[team] = 'Below Average'

  return team_tiers

def get_team_pitching_tiers(season: int) -> dict:
  """
  Fetch team pitching stats and categorize into quality tiers.

  Args:
    season: The season year

  Returns:
    Dictionary mapping team abbreviations to quality tiers
  """
  try:
    from pybaseball import pitching_stats

    # Get all pitchers with minimal qualification
    pitching = pitching_stats(season, qual=1)

    if pitching.empty:
      return {}

    # Aggregate to team level
    team_pitching = pitching.groupby('Team').agg({
      'ERA': 'mean',
      'WHIP': 'mean',
      'IPouts': 'sum'
    }).reset_index()

    # Only include teams with meaningful IP
    team_pitching = team_pitching[team_pitching['IP'] >= 100]

    # Categorize by ERA (lower is better)
    return categorize_opponent_quality(team_pitching, 'ERA', reverse=True)

  except Exception as e:
    print(f"Error fetching pitching tiers: {e}")
    return {}

def get_team_batting_tiers(season: int) -> dict:
  """
  Fetch team batting stats and categorize into quality tiers.

  Args:
    season: The season year

  Returns:
    Dictionary mapping team abbreviations to quality tiers
  """
  try:
    from pybaseball import batting_stats

    #Get all batters with minimal qualification
    batting = batting_stats(season, qual=1)

    if batting.empty:
      return {}

    # Aggregate to team level
    team_batting = batting.groupby('Team').agg({
      'OPS': 'mean',
      'wRC+': 'mean' if 'wRC+' in batting.columns else 'first',
      'PA': 'sum'
    }).reset_index()

    # Only include teams with meaningful PA
    team_batting = team_batting[team_batting['PA'] >= 500]

    # Categorize by OPS (higher is better)
    return categorize_opponent_quality(team_batting, 'OPS', reverse=False)

  except Exception as e:
    print(f"Error fetching batting tiers: {e}")
    return {}

def add_opponent_to_statcast(statcast_df: pd.DataFrame, player_team: str) -> pd.DataFrame:
  """
  Add opponent team column to Statcast data.

  Statcast data has 'home_team' and 'away_team'. We need to figure out which one is the opponent
  based on the player's team.

  Args:
    statcast_df: Player's Statcast data
    player_team: Player's team abbreviation

  Returns:
    DataFrame with 'opponent_team' column added
  """
  if statcast_df.empty:
    return statcast_df

  df = statcast_df.copy()

  # Standardize the player's team
  player_team_std = standardize_team_abbrev(player_team)

  # standardize home and away teams in the Statcast data
  if 'home_team' in df.columns:
    df['home_team_std'] = df['home_team'].apply(standardize_team_abbrev)
  if 'away_team' in df.columns:
    df['away_team_std'] = df['away_team'].apply(standardize_team_abbrev)

  # Determine opponent: if player is on home team, opponent is away team and vice versa
  def get_opponent(row):
    if pd.isna(row.get('home_team_std')) or pd.isna(row.get('away_team_std')):
      return None
    if row['home_team_std'] == player_team_std:
      return row['away_team_std']
    elif row['away_team_std'] == player_team_std:
      return row['home_team_std']
    else:
      # Player may have been traded?
      return None

  df['opponent_team'] = df.apply(get_opponent, axis=1)

  return df

def calculate_splits_by_opponent(statcast_df: pd.DataFrame, player_team: str, season: int, player_type: str = 'batter') -> dict:
  """
  Calculate player stats splits by opponent quality tier

  Args:
    statcast_df: Player's Statcast data
    player_team: The player's team abbreviation
    season: The season year
    player_type: 'batter' or 'pitcher'

  Returns:
    Dictionary of stats by opponent quality tier with this structure:
    {
      'Elite': {'PA': x, 'AVG': x, ...},
      'Average': {...},
      'Below Average': {...}
    }
  """
  if statcast_df.empty:
    return {}

  # Get team quality tiers based on player type
  # Batters face pitchers, so use pitching tiers, vice versa
  if player_type == 'batter':
    team_tiers = get_team_pitching_tiers(season)
  else:
    team_tiers = get_team_batting_tiers(season)

  if not team_tiers:
    return {}

  # Add opponent team to Statcast data
  df = add_opponent_to_statcast(statcast_df, player_team)

  if 'opponent_team' not in df.columns:
    return {}

  # Map opponent teams to tiers
  df['opponent_tier'] = df['opponent_team'].map(team_tiers)

  # Calculate stats for each tier
  splits = {}

  for tier in ['Elite', 'Average', 'Below Average']:
    tier_df = df[df['opponent_tier'] == tier]

    if tier_df.empty:
      splits[tier] = {
        'PA': 0,
        'AB': 0,
        'AVG': None,
        'OBP': None,
        'SLG': None,
        'HR': 0,
        'SO': 0,
        'BB': 0,
        'Avg Exit Velo': None,
        'Barrel%': None
      }
      continue

    # Count PA outcomes
    pa = len(tier_df)

    # Define what counts as an at-bat (excludes walks, HBP, sac flies, etc.)
    ab_events = ['single', 'double', 'triple', 'home_run', 'field_out',
                 'strikeout', 'grounded_into_double_play', 'force_out',
                 'fielders_choice', 'double_play', 'field_error']
    at_bats = len(tier_df[tier_df['events'].isin(ab_events)])

    # Count hits
    hit_events = ['single', 'double', 'triple', 'home_run']
    hits = len(tier_df[tier_df['events'].isin(hit_events)])

    # Count extra base hits
    doubles = len(tier_df[tier_df['events'] == 'double'])
    triples = len(tier_df[tier_df['events'] == 'triple'])
    home_runs = len(tier_df[tier_df['events'] == 'home_run'])

    # Count walks and strikeouts
    walks = len(tier_df[tier_df['events'].isin(['walk', 'hit_by_pitch'])])
    strikeouts = len(tier_df[tier_df['events'] == 'strikeout'])

    # Calculate rate stats
    avg = hits / at_bats if at_bats > 0 else 0
    obp = (hits + walks) / pa if pa > 0 else 0

    # Total bases for SLG
    total_bases = (hits - doubles - triples - home_runs) + (2 * doubles) + (3 * triples) + (4 * home_runs)
    slg = total_bases / at_bats if at_bats > 0 else 0

    # Statcast metrics
    batted_balls = tier_df[tier_df['launch_speed'].notna()]
    avg_exit_velo = batted_balls['launch_speed'].mean() if len(batted_balls) > 0 else None

    barrel_rate = None
    if 'barrel' in tier_df.columns:
      barrel_balls = tier_df[tier_df['barrel'].notna()]
      if len(barrel_balls) > 0:
        barrel_rate = barrel_balls['barrel'].mean() * 100

    splits[tier] = {
      'PA': pa,
      'AB': at_bats,
      'H': hits,
      'AVG': round(avg, 3),
      'OBP': round(obp, 3),
      'SLG': round(slg, 3),
      'OPS': round(obp + slg, 3),
      'HR': home_runs,
      'SO': strikeouts,
      'BB': walks,
      'K%': round(strikeouts / pa * 100, 1) if pa > 0 else 0,
      'BB%': round(walks / pa * 100, 1) if pa > 0 else 0,
      'Avg Exit Velo': round(avg_exit_velo, 1) if avg_exit_velo else None,
      'Barrel%': round(barrel_rate, 1) if barrel_rate else None
    }

  return splits

def format_splits_for_display(splits: dict) -> pd.DataFrame:
  """
  Convert splits dictionary to a display-friendly DataFrame

  Args:
     splits: Dictionary from calculate_splits_by_opponent

    Returns:
      DataFrame with tiers as rows and stats as columns
  """
  if not splits:
    return pd.DataFrame()

  # Order the tiers
  tier_order = ['Elite', 'Average', 'Below Average']

  rows = []
  for tier in tier_order:
    if tier in splits:
      row = {'Opponent Tier': tier}
      row.update(splits[tier])
      rows.append(row)

  df = pd.DataFrame(rows)

  if not df.empty:
    df = df.set_index('Opponent Tier')

  return df



