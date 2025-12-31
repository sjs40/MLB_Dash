# app.py
"""Main Streamlit application for the Baseball Stats Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# Page configuration - MUST be first Streamlit command
st.set_page_config(
  page_title="Baseball Stats Dashboard",
  page_icon="⚾",
  layout="wide",
  initial_sidebar_state="expanded"
)

# Import our modules
from config import CURRENT_SEASON, BATTING_STATS, PITCHING_STATS
from data.fetcher import (
  get_batting_stats,
  get_pitching_stats,
  get_league_averages,
  get_statcast_batter_data,
  get_statcast_pitcher_data,
  lookup_player,
)
from data.processors import (
  calculate_normalized_stats,
  calculate_percentile_ranks,
  calculate_splits_by_opponent,
  format_splits_for_display,
)
from components.charts import (
  create_percentile_chart,
  create_normalized_bar_chart,
)
from components.stats_tables import (
  display_key_batting_stats,
  display_key_pitching_stats,
  display_detailed_batting_table,
  display_detailed_pitching_table,
)


def main():
  """Main application function."""

  # Header
  st.title("⚾ Baseball Stats Dashboard")
  st.markdown("*Analyze player performance with current stats, league comparisons, and trends*")

  # Sidebar for navigation and filters
  with st.sidebar:
    st.header("🎯 Navigation")

    page = st.radio(
      "Select View",
      ["Player Stats", "League Leaders", "Team Analysis", "Compare Players"],
      index=0
    )

    st.divider()

    st.header("⚙️ Filters")

    season = st.selectbox(
      "Season",
      options=list(range(CURRENT_SEASON, 2019, -1)),
      index=0
    )

    stat_type = st.radio(
      "Stat Type",
      ["Batting", "Pitching"],
      index=0
    )

    st.divider()

    st.markdown("### 📊 Data Info")
    st.caption("Data sourced from FanGraphs and Statcast")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    st.divider()
    st.markdown("### 💡 Tips")
    st.caption("• Use the tabs to explore different stat views")
    st.caption("• Normalized stats: 100 = league average")
    st.caption("• Percentiles: 100th = best in league")

  # Main content based on selected page
  if page == "Player Stats":
    show_player_stats(season, stat_type)
  elif page == "League Leaders":
    show_league_leaders(season, stat_type)
  elif page == "Team Analysis":
    show_team_analysis(season)
  elif page == "Compare Players":
    show_player_comparison(season, stat_type)


def show_player_stats(season: int, stat_type: str):
  """Display individual player statistics."""

  st.header("📈 Individual Player Stats")

  # Load data
  with st.spinner("Loading player data..."):
    if stat_type == "Batting":
      df = get_batting_stats(season)
    else:
      df = get_pitching_stats(season)

  if df.empty:
    st.error("Unable to load data. Please try again later.")
    st.info("This might happen if the season hasn't started yet or if there's an issue with the data source.")
    return

  # Player selector
  player_names = sorted(df['Name'].unique())
  selected_player = st.selectbox(
    "Select Player",
    options=player_names,
    index=0
  )

  # Get player data
  player_data = df[df['Name'] == selected_player].iloc[0]

  # Create tabs for different stat views
  tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Current Stats",
    "📏 League Normalized",
    "📈 Rolling Trends",
    "🎯 Percentile Rankings",
    "⚔️ Opponent Splits"
  ])

  with tab1:
    show_current_stats(player_data, df, stat_type)

  with tab2:
    show_normalized_stats(player_data, season, stat_type)

  with tab3:
    show_rolling_stats(player_data, season, stat_type)

  with tab4:
    show_percentile_rankings(player_data, df, stat_type)

  with tab5:
    show_opponent_splits(player_data, season, stat_type)


def show_current_stats(player_data: pd.Series, all_players: pd.DataFrame, stat_type: str):
  """Display current season statistics."""

  st.subheader(f"Season Statistics: {player_data['Name']}")

  # Team and position info
  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric("Team", player_data.get('Team', 'N/A'))
  with col2:
    if stat_type == "Batting":
      st.metric("Games", int(player_data.get('G', 0)))
    else:
      st.metric("Games Started", int(player_data.get('GS', 0)))
  with col3:
    if stat_type == "Batting":
      st.metric("Plate Appearances", int(player_data.get('PA', 0)))
    else:
      st.metric("Innings Pitched", round(player_data.get('IP', 0), 1))

  st.divider()

  # Key statistics in metric cards
  if stat_type == "Batting":
    display_key_batting_stats(player_data)
  else:
    display_key_pitching_stats(player_data)

  # Detailed stats table
  st.divider()
  st.subheader("Detailed Statistics")

  if stat_type == "Batting":
    display_detailed_batting_table(player_data)
  else:
    display_detailed_pitching_table(player_data)

  # Advanced stats if available
  st.divider()
  st.subheader("Advanced Metrics")

  if stat_type == "Batting":
    adv_cols = st.columns(4)
    adv_stats = [
      ('wRC+', 'wRC+ (Weighted Runs Created Plus)'),
      ('wOBA', 'wOBA (Weighted On-Base Average)'),
      ('ISO', 'ISO (Isolated Power)'),
      ('BABIP', 'BABIP')
    ]
  else:
    adv_cols = st.columns(4)
    adv_stats = [
      ('FIP', 'FIP (Fielding Independent Pitching)'),
      ('xFIP', 'xFIP (Expected FIP)'),
      ('K%', 'K% (Strikeout Rate)'),
      ('BB%', 'BB% (Walk Rate)')
    ]

  for col, (stat, label) in zip(adv_cols, adv_stats):
    with col:
      if stat in player_data.index:
        value = player_data[stat]
        if isinstance(value, float):
          value = round(value, 3)
        st.metric(label, value)
      else:
        st.metric(label, "N/A")


def show_normalized_stats(player_data: pd.Series, season: int, stat_type: str):
  """Display league-normalized statistics."""

  st.subheader("League-Normalized Statistics")
  st.markdown("""
    These stats show how the player compares to league average. 
    A value of **100** is exactly league average. Values above 100 are better than average.
    """)

  # Get league averages
  with st.spinner("Calculating league averages..."):
    league_avgs = get_league_averages(season)

  if not league_avgs:
    st.warning("Unable to calculate league averages for this season.")
    return

  # Calculate normalized stats
  normalized = calculate_normalized_stats(
    player_data,
    league_avgs,
    stat_type.lower()
  )

  if not normalized:
    st.warning("Unable to calculate normalized statistics for this player.")
    return

  # Display as metrics
  cols = st.columns(len(normalized))

  for col, (stat, value) in zip(cols, normalized.items()):
    with col:
      delta = value - 100
      st.metric(
        stat,
        f"{value:.0f}",
        delta=f"{delta:+.1f} vs avg",
        delta_color="normal"
      )

  st.divider()

  # Visual comparison
  st.subheader("Visual Comparison to League Average")

  fig = create_normalized_bar_chart(normalized, player_data['Name'])
  st.plotly_chart(fig, use_container_width=True)

  # Explanation
  with st.expander("📖 Understanding Normalized Stats"):
    st.markdown("""
        **How to interpret these stats:**

        - **100** = Exactly league average
        - **>100** = Better than league average
        - **<100** = Below league average

        For example:
        - An **OPS+ of 150** means the player's OPS is 50% better than league average
        - An **ERA+ of 150** means the player allows 50% fewer runs than league average

        **Stats shown:**

        *For Batters:*
        - **OPS+**: Overall offensive value (higher = better)
        - **AVG+**: Batting average compared to league (higher = better)
        - **ISO+**: Power/extra-base hit ability (higher = better)
        - **K%+**: Strikeout rate (higher = LESS strikeouts, which is better)
        - **BB%+**: Walk rate (higher = more walks, which is better)

        *For Pitchers:*
        - **ERA+**: Run prevention (higher = better)
        - **FIP+**: Fielding-independent pitching (higher = better)
        - **WHIP+**: Baserunner prevention (higher = better)
        - **K/9+**: Strikeout ability (higher = better)
        """)


def show_rolling_stats(player_data: pd.Series, season: int, stat_type: str):
  """Display rolling/recent performance trends."""

  st.subheader("Rolling Performance Trends")
  st.markdown("*See how the player has performed recently compared to their season averages.*")

  st.info("""
    **Note:** Rolling stats require game-by-game Statcast data. 

    To implement this with real data, you would:
    1. Look up the player's MLB ID using `playerid_lookup(last_name, first_name)`
    2. Fetch their Statcast data using `statcast_batter(start_date, end_date, player_id)`
    3. Calculate rolling windows from the game-by-game data

    Below is a simulated example of what this would look like.
    """)

  # Simulated rolling data for demonstration
  dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
  season_avg = player_data.get('AVG', 0.250) if stat_type == "Batting" else player_data.get('ERA', 4.00)

  # Create rolling values with some variance
  np.random.seed(hash(player_data['Name']) % 2 ** 32)

  if stat_type == "Batting":
    base_variation = np.random.normal(0, 0.025, len(dates))
    rolling_7 = np.clip(season_avg + base_variation + np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 0.03, 0.150,
                        0.400)
    rolling_30 = pd.Series(rolling_7).rolling(window=15, min_periods=1).mean().values
  else:
    base_variation = np.random.normal(0, 0.4, len(dates))
    rolling_7 = np.clip(season_avg + base_variation + np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 0.5, 2.0, 7.0)
    rolling_30 = pd.Series(rolling_7).rolling(window=15, min_periods=1).mean().values

  # Create the chart
  fig = go.Figure()

  fig.add_trace(go.Scatter(
    x=dates, y=rolling_7,
    mode='lines',
    name='7-Day Rolling',
    line=dict(color='blue', width=2)
  ))

  fig.add_trace(go.Scatter(
    x=dates, y=rolling_30,
    mode='lines',
    name='30-Day Rolling',
    line=dict(color='orange', width=2)
  ))

  fig.add_hline(y=season_avg, line_dash="dash", line_color="green",
                annotation_text=f"Season {'AVG' if stat_type == 'Batting' else 'ERA'}")

  fig.update_layout(
    title=f"Rolling {'Batting Average' if stat_type == 'Batting' else 'ERA'} Trend (Simulated)",
    xaxis_title="Date",
    yaxis_title="AVG" if stat_type == "Batting" else "ERA",
    height=400,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
  )

  st.plotly_chart(fig, use_container_width=True)

  # Rolling stats summary
  st.subheader("Recent Performance Summary")

  col1, col2, col3 = st.columns(3)

  stat_name = "AVG" if stat_type == "Batting" else "ERA"
  delta_color = "normal" if stat_type == "Batting" else "inverse"

  with col1:
    st.markdown("**Last 7 Days**")
    recent_val = rolling_7[-1]
    diff = recent_val - season_avg
    st.metric(
      stat_name,
      f"{recent_val:.3f}",
      delta=f"{diff:+.3f}",
      delta_color=delta_color
    )

  with col2:
    st.markdown("**Last 14 Days**")
    mid_val = rolling_30[-14] if len(rolling_30) > 14 else rolling_30[-1]
    diff = mid_val - season_avg
    st.metric(
      stat_name,
      f"{mid_val:.3f}",
      delta=f"{diff:+.3f}",
      delta_color=delta_color
    )

  with col3:
    st.markdown("**Last 30 Days**")
    month_val = rolling_30[-1]
    diff = month_val - season_avg
    st.metric(
      stat_name,
      f"{month_val:.3f}",
      delta=f"{diff:+.3f}",
      delta_color=delta_color
    )

  # Hot/Cold streak indicator
  st.divider()
  recent_trend = rolling_7[-1] - rolling_7[-7] if len(rolling_7) >= 7 else 0

  if stat_type == "Batting":
    if recent_trend > 0.020:
      st.success("🔥 **HOT STREAK** - Player is trending upward!")
    elif recent_trend < -0.020:
      st.warning("❄️ **COLD STREAK** - Player is trending downward")
    else:
      st.info("➡️ **STEADY** - Player is performing consistently")
  else:
    if recent_trend < -0.3:
      st.success("🔥 **HOT STREAK** - Pitcher is trending better!")
    elif recent_trend > 0.3:
      st.warning("❄️ **COLD STREAK** - Pitcher is struggling")
    else:
      st.info("➡️ **STEADY** - Pitcher is performing consistently")


def show_percentile_rankings(player_data: pd.Series, all_players: pd.DataFrame, stat_type: str):
  """Display percentile rankings among all players."""

  st.subheader("Percentile Rankings")
  st.markdown("*See where this player ranks among all qualified players (100th = best)*")

  # Define stats to show percentiles for
  if stat_type == "Batting":
    percentile_stats = {
      'AVG': False,  # False = higher is better
      'OBP': False,
      'SLG': False,
      'OPS': False,
      'HR': False,
      'RBI': False,
      'SB': False,
    }
    # Add percentage stats if available
    if 'BB%' in all_players.columns:
      percentile_stats['BB%'] = False
    if 'K%' in all_players.columns:
      percentile_stats['K%'] = True  # Lower is better
  else:
    percentile_stats = {
      'ERA': True,  # Lower is better
      'WHIP': True,
      'W': False,
      'SO': False,
    }
    if 'K/9' in all_players.columns:
      percentile_stats['K/9'] = False
    if 'BB/9' in all_players.columns:
      percentile_stats['BB/9'] = True
    if 'FIP' in all_players.columns:
      percentile_stats['FIP'] = True

  percentiles = {}

  for stat, invert in percentile_stats.items():
    if stat in all_players.columns and stat in player_data.index:
      player_val = player_data[stat]
      all_vals = all_players[stat].dropna()

      if len(all_vals) == 0:
        continue

      if invert:
        # For stats where lower is better, invert the percentile
        pct = 100 - calculate_percentile_ranks(player_val, all_vals)
      else:
        pct = calculate_percentile_ranks(player_val, all_vals)

      percentiles[stat] = pct

  if not percentiles:
    st.warning("Unable to calculate percentile rankings.")
    return

  # Create horizontal bar chart
  stats = list(percentiles.keys())
  pcts = list(percentiles.values())

  # Color bars based on percentile
  colors = ['#d62728' if p < 25 else '#ff7f0e' if p < 50 else '#2ca02c' if p < 75 else '#1f77b4' for p in pcts]

  fig = go.Figure()

  fig.add_trace(go.Bar(
    y=stats,
    x=pcts,
    orientation='h',
    marker_color=colors,
    text=[f"{p}th" for p in pcts],
    textposition='outside'
  ))

  # Add 50th percentile line
  fig.add_vline(x=50, line_dash="dash", line_color="gray",
                annotation_text="50th Percentile")

  fig.update_layout(
    title=f"Percentile Rankings - {player_data['Name']}",
    xaxis_title="Percentile",
    xaxis=dict(range=[0, 105]),
    height=max(300, len(stats) * 50),
    showlegend=False
  )

  st.plotly_chart(fig, use_container_width=True)

  # Legend
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.markdown("🔴 **Below 25th**")
  with col2:
    st.markdown("🟠 **25th-50th**")
  with col3:
    st.markdown("🟢 **50th-75th**")
  with col4:
    st.markdown("🔵 **Above 75th**")

  # Detailed percentile table
  with st.expander("📊 Detailed Percentile Data"):
    pct_df = pd.DataFrame({
      'Stat': stats,
      'Value': [player_data[s] for s in stats],
      'Percentile': pcts,
      'Rating': [
        'Elite' if p >= 90 else 'Great' if p >= 75 else 'Above Avg' if p >= 50 else 'Below Avg' if p >= 25 else 'Poor'
        for p in pcts]
    })
    st.dataframe(pct_df, use_container_width=True, hide_index=True)


def show_opponent_splits(player_data: pd.Series, season: int, stat_type: str):
  """Display stats split by opponent quality."""

  st.subheader("Performance by Opponent Quality")
  st.markdown("""
    *See how this player performs against elite, average, and below-average opponents.*

    - **Elite**: Top third of MLB teams (by pitching ERA for batters, by batting OPS for pitchers)
    - **Average**: Middle third of MLB teams
    - **Below Average**: Bottom third of MLB teams
    """)

  player_name = player_data['Name']
  player_team = player_data.get('Team', '')

  if not player_team:
    st.warning("Could not determine player's team. Cannot calculate opponent splits.")
    return

  # Look up the player's MLB ID
  st.info(f"Looking up Statcast data for {player_name}...")

  try:
    # Parse player name for lookup
    name_parts = player_name.split()
    if len(name_parts) >= 2:
      first_name = name_parts[0]
      last_name = ' '.join(name_parts[1:])  # Handle names like "De La Cruz"
    else:
      st.warning("Could not parse player name for lookup.")
      return

    # Look up player ID
    player_lookup = lookup_player(first_name, last_name)

    if player_lookup.empty:
      st.warning(f"Could not find player ID for {player_name}. Try a different player.")
      return

    # Get the most recent player ID (in case of multiple matches)
    # Filter for MLB players
    if 'mlb_played_last' in player_lookup.columns:
      recent_players = player_lookup[player_lookup['mlb_played_last'] >= season - 1]
      if not recent_players.empty:
        player_lookup = recent_players

    player_id = int(player_lookup.iloc[0]['key_mlbam'])

    # Fetch Statcast data for the season
    start_date = f"{season}-03-01"  # Spring training start
    end_date = f"{season}-11-30"  # After World Series

    with st.spinner(f"Fetching Statcast data for {player_name}..."):
      if stat_type == "Batting":
        statcast_data = get_statcast_batter_data(player_id, start_date, end_date)
      else:
        statcast_data = get_statcast_pitcher_data(player_id, start_date, end_date)

    if statcast_data.empty:
      st.warning(f"No Statcast data found for {player_name} in {season}.")
      return

    st.success(f"Found {len(statcast_data)} plate appearances in Statcast data.")

    # Calculate splits by opponent quality
    with st.spinner("Calculating opponent splits..."):
      player_type = 'batter' if stat_type == "Batting" else 'pitcher'
      splits = calculate_splits_by_opponent(
        statcast_data,
        player_team,
        season,
        player_type
      )

    if not splits:
      st.warning("Could not calculate opponent splits. Team tier data may not be available.")
      return

    # Display splits table
    st.subheader("Stats by Opponent Quality")

    splits_df = format_splits_for_display(splits)

    if not splits_df.empty:
      # Highlight the best tier for each stat
      st.dataframe(splits_df, use_container_width=True)

      # Visual comparison
      st.subheader("Visual Comparison")

      # Select stat to visualize
      available_stats = [col for col in splits_df.columns if splits_df[col].notna().any()]
      numeric_stats = [s for s in available_stats if s not in ['PA', 'AB', 'H', 'HR', 'SO', 'BB']]

      if numeric_stats:
        selected_stat = st.selectbox("Select stat to compare", numeric_stats, index=0)

        # Create bar chart
        tiers = splits_df.index.tolist()
        values = splits_df[selected_stat].tolist()

        # Color code: green for best performance
        if selected_stat in ['K%']:  # Lower is better for batters
          colors = ['green' if v == min([x for x in values if x is not None]) else 'steelblue'
                    for v in values]
        else:  # Higher is better
          colors = ['green' if v == max([x for x in values if x is not None]) else 'steelblue'
                    for v in values]

        fig = go.Figure()
        fig.add_trace(go.Bar(
          x=tiers,
          y=values,
          marker_color=colors,
          text=[f"{v:.3f}" if isinstance(v, float) else str(v) for v in values],
          textposition='outside'
        ))

        fig.update_layout(
          title=f"{selected_stat} by Opponent Quality",
          yaxis_title=selected_stat,
          height=400
        )

        st.plotly_chart(fig, use_container_width=True)

      # Key insights
      st.subheader("Key Insights")

      col1, col2 = st.columns(2)

      with col1:
        # Find where player performs best
        if 'AVG' in splits_df.columns:
          avg_values = {tier: splits_df.loc[tier, 'AVG'] for tier in splits_df.index
                        if pd.notna(splits_df.loc[tier, 'AVG'])}
          if avg_values:
            best_tier = max(avg_values, key=avg_values.get)
            worst_tier = min(avg_values, key=avg_values.get)

            st.metric(
              "Best AVG vs",
              best_tier,
              f"{avg_values[best_tier]:.3f}"
            )

      with col2:
        if 'OPS' in splits_df.columns:
          ops_values = {tier: splits_df.loc[tier, 'OPS'] for tier in splits_df.index
                        if pd.notna(splits_df.loc[tier, 'OPS'])}
          if ops_values:
            best_tier = max(ops_values, key=ops_values.get)

            st.metric(
              "Best OPS vs",
              best_tier,
              f"{ops_values[best_tier]:.3f}"
            )

      # Interpretation
      with st.expander("📖 How to interpret opponent splits"):
        st.markdown("""
                **What these splits tell you:**

                - **Elite opponents**: How the player performs against the best teams in the league
                - **Average opponents**: Performance against middle-of-the-pack teams  
                - **Below Average opponents**: Performance against weaker competition

                **What to look for:**

                1. **Consistent performers** show similar numbers across all tiers
                2. **Elite crushers** actually perform better against top competition (rare but valuable)
                3. **Stat padders** have inflated numbers mostly from below-average opponents

                **Sample size matters!** If PA is low for a tier, the stats may not be reliable.
                Generally, you want 50+ PA per tier for meaningful conclusions.
                """)

  except Exception as e:
    st.error(f"Error calculating opponent splits: {e}")
    st.info("This feature requires Statcast data which may not be available for all players.")


def show_league_leaders(season: int, stat_type: str):
  """Display league leaders for various statistics."""

  st.header("🏆 League Leaders")

  with st.spinner("Loading leaderboard data..."):
    if stat_type == "Batting":
      df = get_batting_stats(season)
    else:
      df = get_pitching_stats(season)

  if df.empty:
    st.error("Unable to load data.")
    return

  # Select stat category
  if stat_type == "Batting":
    stat_options = ['AVG', 'HR', 'RBI', 'OPS', 'SB', 'H', 'R']
    if 'wRC+' in df.columns:
      stat_options.append('wRC+')
    if 'WAR' in df.columns:
      stat_options.append('WAR')
    if 'wOBA' in df.columns:
      stat_options.append('wOBA')
  else:
    stat_options = ['ERA', 'W', 'SO', 'SV', 'WHIP']
    if 'K/9' in df.columns:
      stat_options.append('K/9')
    if 'FIP' in df.columns:
      stat_options.append('FIP')
    if 'WAR' in df.columns:
      stat_options.append('WAR')

  # Filter to available columns
  stat_options = [s for s in stat_options if s in df.columns]

  col1, col2 = st.columns([1, 3])

  with col1:
    selected_stat = st.selectbox("Select Statistic", stat_options)
    num_players = st.slider("Number of Players", 10, 50, 20)

  # Determine sort order (ascending for ERA, WHIP, etc.)
  ascending_stats = ['ERA', 'WHIP', 'FIP', 'BB/9', 'xFIP']
  ascending = selected_stat in ascending_stats

  # Get top players
  top_players = df.nsmallest(num_players, selected_stat) if ascending else df.nlargest(num_players, selected_stat)

  # Display columns
  display_cols = ['Name', 'Team', selected_stat]
  if stat_type == "Batting":
    extra_cols = ['G', 'PA', 'AVG', 'OPS', 'HR']
  else:
    extra_cols = ['G', 'IP', 'ERA', 'WHIP', 'SO']

  for col in extra_cols:
    if col in top_players.columns and col not in display_cols:
      display_cols.append(col)

  # Reset index and add rank
  top_players = top_players[display_cols].reset_index(drop=True)
  top_players.index = top_players.index + 1
  top_players.index.name = 'Rank'

  with col2:
    st.dataframe(
      top_players,
      use_container_width=True,
      height=min(600, num_players * 35 + 50)
    )

  # Quick visualization
  st.divider()
  st.subheader(f"Top 10 - {selected_stat}")

  top_10 = top_players.head(10)

  fig = go.Figure()
  fig.add_trace(go.Bar(
    x=top_10['Name'],
    y=top_10[selected_stat],
    marker_color='steelblue',
    text=top_10[selected_stat].round(3),
    textposition='outside'
  ))

  fig.update_layout(
    xaxis_title="Player",
    yaxis_title=selected_stat,
    height=400,
    xaxis_tickangle=-45
  )

  st.plotly_chart(fig, use_container_width=True)


def show_team_analysis(season: int):
  """Display team-level analysis."""

  st.header("🏟️ Team Analysis")
  st.markdown("*Team analysis helps contextualize opponent quality for splits analysis.*")

  with st.spinner("Loading team data..."):
    batting_df = get_batting_stats(season, qual=1)
    pitching_df = get_pitching_stats(season, qual=1)

  if batting_df.empty:
    st.error("Unable to load team data.")
    return

  # Aggregate to team level
  team_batting = batting_df.groupby('Team').agg({
    'G': 'max',
    'PA': 'sum',
    'H': 'sum',
    'HR': 'sum',
    'RBI': 'sum',
    'AVG': 'mean',
    'OPS': 'mean',
  }).round(3)

  team_pitching = pitching_df.groupby('Team').agg({
    'G': 'max',
    'IP': 'sum',
    'SO': 'sum',
    'ERA': 'mean',
    'WHIP': 'mean',
  }).round(3) if not pitching_df.empty else pd.DataFrame()

  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Team Batting (Avg of Qualified Hitters)")
    st.dataframe(
      team_batting.sort_values('OPS', ascending=False),
      use_container_width=True,
      height=400
    )

  with col2:
    if not team_pitching.empty:
      st.subheader("Team Pitching (Avg of Qualified Pitchers)")
      st.dataframe(
        team_pitching.sort_values('ERA', ascending=True),
        use_container_width=True,
        height=400
      )

  st.divider()

  st.subheader("Team Quality Tiers")
  st.markdown("""
    Teams can be categorized based on their pitching/batting quality for splits analysis:
    - **Elite**: Top third of MLB
    - **Average**: Middle third
    - **Below Average**: Bottom third

    Use these tiers to analyze how players perform against different quality opponents.
    """)

  if not team_pitching.empty:
    # Categorize teams by ERA
    team_pitching_sorted = team_pitching.sort_values('ERA')
    n_teams = len(team_pitching_sorted)

    col1, col2, col3 = st.columns(3)

    with col1:
      st.markdown("**🥇 Elite Pitching**")
      elite = team_pitching_sorted.head(n_teams // 3)
      for team in elite.index:
        st.write(f"• {team} ({elite.loc[team, 'ERA']:.2f} ERA)")

    with col2:
      st.markdown("**🥈 Average Pitching**")
      avg = team_pitching_sorted.iloc[n_teams // 3: 2 * n_teams // 3]
      for team in avg.index:
        st.write(f"• {team} ({avg.loc[team, 'ERA']:.2f} ERA)")

    with col3:
      st.markdown("**🥉 Below Average Pitching**")
      below = team_pitching_sorted.tail(n_teams // 3)
      for team in below.index:
        st.write(f"• {team} ({below.loc[team, 'ERA']:.2f} ERA)")


def show_player_comparison(season: int, stat_type: str):
  """Compare multiple players side by side."""

  st.header("⚖️ Player Comparison")

  with st.spinner("Loading player data..."):
    if stat_type == "Batting":
      df = get_batting_stats(season)
    else:
      df = get_pitching_stats(season)

  if df.empty:
    st.error("Unable to load data.")
    return

  player_names = sorted(df['Name'].unique())

  # Select up to 4 players
  selected_players = st.multiselect(
    "Select Players to Compare (max 4)",
    options=player_names,
    max_selections=4,
    default=player_names[:2] if len(player_names) >= 2 else player_names
  )

  if len(selected_players) < 2:
    st.warning("Please select at least 2 players to compare.")
    return

  # Filter data
  compare_df = df[df['Name'].isin(selected_players)]

  # Select stats to compare
  if stat_type == "Batting":
    compare_stats = ['AVG', 'OBP', 'SLG', 'OPS', 'HR', 'RBI', 'SB']
    if 'wRC+' in compare_df.columns:
      compare_stats.append('wRC+')
    if 'WAR' in compare_df.columns:
      compare_stats.append('WAR')
  else:
    compare_stats = ['ERA', 'WHIP', 'W', 'SO']
    if 'K/9' in compare_df.columns:
      compare_stats.append('K/9')
    if 'FIP' in compare_df.columns:
      compare_stats.append('FIP')
    if 'WAR' in compare_df.columns:
      compare_stats.append('WAR')

  # Filter to available
  compare_stats = [s for s in compare_stats if s in compare_df.columns]

  # Display comparison table
  st.subheader("Side-by-Side Comparison")

  comparison_data = compare_df[['Name'] + compare_stats].set_index('Name').T
  st.dataframe(comparison_data, use_container_width=True)

  # Radar chart comparison
  st.subheader("Visual Comparison")

  # Normalize stats for radar chart (0-1 scale)
  normalized_df = compare_df[compare_stats].copy()
  for col in compare_stats:
    min_val = df[col].min()
    max_val = df[col].max()
    if max_val > min_val:
      # For stats where lower is better (ERA, WHIP), invert
      if col in ['ERA', 'WHIP', 'FIP', 'BB/9']:
        normalized_df[col] = 1 - (compare_df[col] - min_val) / (max_val - min_val)
      else:
        normalized_df[col] = (compare_df[col] - min_val) / (max_val - min_val)

  fig = go.Figure()

  colors = ['blue', 'red', 'green', 'orange']

  for i, (_, row) in enumerate(compare_df.iterrows()):
    player_name = row['Name']
    values = normalized_df.iloc[i].values.tolist()
    values.append(values[0])  # Close the polygon

    fig.add_trace(go.Scatterpolar(
      r=values,
      theta=compare_stats + [compare_stats[0]],
      fill='toself',
      name=player_name,
      line_color=colors[i % len(colors)],
      opacity=0.6
    ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )
    ),
    showlegend=True,
    title="Normalized Stat Comparison",
    height=500
  )

  st.plotly_chart(fig, use_container_width=True)

  st.caption(
    "*Stats are normalized to 0-1 scale where 1 is the league leader. For ERA/WHIP/FIP, the scale is inverted so that better performance = higher value.*")

  # Bar chart comparison
  st.divider()
  st.subheader("Direct Stat Comparison")

  stat_to_compare = st.selectbox("Select stat to compare", compare_stats)

  fig = go.Figure()

  fig.add_trace(go.Bar(
    x=compare_df['Name'],
    y=compare_df[stat_to_compare],
    marker_color=colors[:len(compare_df)],
    text=compare_df[stat_to_compare].round(3),
    textposition='outside'
  ))

  fig.update_layout(
    title=f"{stat_to_compare} Comparison",
    yaxis_title=stat_to_compare,
    height=400
  )

  st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
  main()
