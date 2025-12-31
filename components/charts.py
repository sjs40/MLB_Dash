# components/charts.py
"""Chart components for the baseball dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_percentile_chart(percentiles: dict, player_name: str) -> go.Figure:
  """Create a horizontal bar chart showing percentile rankings."""

  stats = list(percentiles.keys())
  pcts = list(percentiles.values())

  colors = [
    '#d62728' if p < 25 else
    '#ff7f0e' if p < 50 else
    '#2ca02c' if p < 75 else
    '#1f77b4'
    for p in pcts
  ]

  fig = go.Figure()

  fig.add_trace(go.Bar(
    y=stats,
    x=pcts,
    orientation='h',
    marker_color=colors,
    text=[f"{p}th" for p in pcts],
    textposition='outside'
  ))

  fig.add_vline(x=50, line_dash="dash", line_color="gray")

  fig.update_layout(
    title=f"Percentile Rankings - {player_name}",
    xaxis_title="Percentile",
    xaxis=dict(range=[0, 105]),
    height=400,
    showlegend=False
  )

  return fig


def create_rolling_chart(
        dates: list,
        rolling_7: list,
        rolling_30: list,
        season_avg: float,
        stat_name: str
) -> go.Figure:
  """Create a line chart showing rolling performance trends."""

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

  fig.add_hline(
    y=season_avg,
    line_dash="dash",
    line_color="green",
    annotation_text=f"Season {stat_name}"
  )

  fig.update_layout(
    title=f"Rolling {stat_name} Trend",
    xaxis_title="Date",
    yaxis_title=stat_name,
    height=400
  )

  return fig


def create_comparison_radar(
        players_data: pd.DataFrame,
        stats: list,
        all_players: pd.DataFrame
) -> go.Figure:
  """Create a radar chart comparing multiple players."""

  # Normalize stats to 0-1 scale
  normalized = players_data[stats].copy()
  for col in stats:
    min_val = all_players[col].min()
    max_val = all_players[col].max()
    if max_val > min_val:
      normalized[col] = (players_data[col] - min_val) / (max_val - min_val)

  fig = go.Figure()
  colors = ['blue', 'red', 'green', 'orange']

  for i, (_, row) in enumerate(players_data.iterrows()):
    values = normalized.iloc[i].values.tolist()
    values.append(values[0])

    fig.add_trace(go.Scatterpolar(
      r=values,
      theta=stats + [stats[0]],
      fill='toself',
      name=row['Name'],
      line_color=colors[i % len(colors)],
      opacity=0.6
    ))

  fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    height=500
  )

  return fig


def create_normalized_bar_chart(normalized: dict, player_name: str) -> go.Figure:
  """Create a bar chart comparing player stats to league average."""

  stats = list(normalized.keys())
  values = list(normalized.values())

  # Color bars based on whether above or below average
  colors = ['green' if v >= 100 else 'red' for v in values]

  fig = go.Figure()

  fig.add_trace(go.Bar(
    x=stats,
    y=values,
    marker_color=colors,
    text=[f"{v:.0f}" for v in values],
    textposition='outside'
  ))

  # League average line
  fig.add_hline(y=100, line_dash="dash", line_color="gray",
                annotation_text="League Average (100)")

  fig.update_layout(
    title=f"Normalized Stats vs. League Average - {player_name}",
    yaxis_title="Normalized Value (100 = League Avg)",
    showlegend=False,
    height=400
  )

  return fig


def create_spray_chart(statcast_df: pd.DataFrame, player_name: str = "") -> go.Figure:
  """Create a spray chart showing batted ball locations."""

  # Filter to batted balls
  batted = statcast_df[statcast_df['hc_x'].notna()].copy()

  if batted.empty:
    return None

  # Map events to colors
  event_colors = {
    'single': 'green',
    'double': 'blue',
    'triple': 'purple',
    'home_run': 'red',
    'field_out': 'gray',
    'force_out': 'gray',
    'grounded_into_double_play': 'darkgray',
    'sac_fly': 'lightblue',
    'double_play': 'darkgray'
  }

  batted['color'] = batted['events'].map(event_colors).fillna('lightgray')

  fig = go.Figure()

  # Add batted ball locations
  fig.add_trace(go.Scatter(
    x=batted['hc_x'],
    y=batted['hc_y'],
    mode='markers',
    marker=dict(
      color=batted['color'],
      size=8,
      opacity=0.7
    ),
    text=batted['events'],
    hovertemplate='%{text}<extra></extra>'
  ))

  fig.update_layout(
    title=f"Spray Chart{' - ' + player_name if player_name else ''}",
    xaxis=dict(visible=False, range=[0, 250]),
    yaxis=dict(visible=False, range=[-50, 250], scaleanchor="x"),
    height=500,
    showlegend=False
  )

  return fig


def create_exit_velo_distribution(statcast_df: pd.DataFrame, player_name: str = "") -> go.Figure:
  """Create a histogram of exit velocities."""

  batted = statcast_df[statcast_df['launch_speed'].notna()]

  if batted.empty:
    return None

  fig = px.histogram(
    batted,
    x='launch_speed',
    nbins=30,
    title=f'Exit Velocity Distribution{" - " + player_name if player_name else ""}',
    labels={'launch_speed': 'Exit Velocity (mph)'}
  )

  # Add mean line
  mean_ev = batted['launch_speed'].mean()
  fig.add_vline(
    x=mean_ev,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Avg: {mean_ev:.1f} mph"
  )

  fig.update_layout(height=400)

  return fig


def create_launch_angle_scatter(statcast_df: pd.DataFrame, player_name: str = "") -> go.Figure:
  """Create a scatter plot of exit velo vs launch angle."""

  batted = statcast_df[
    statcast_df['launch_speed'].notna() &
    statcast_df['launch_angle'].notna()
    ].copy()

  if batted.empty:
    return None

  # Color by outcome
  event_colors = {
    'single': 'green',
    'double': 'blue',
    'triple': 'purple',
    'home_run': 'red',
    'field_out': 'gray'
  }
  batted['color'] = batted['events'].map(event_colors).fillna('lightgray')

  fig = go.Figure()

  fig.add_trace(go.Scatter(
    x=batted['launch_angle'],
    y=batted['launch_speed'],
    mode='markers',
    marker=dict(
      color=batted['color'],
      size=6,
      opacity=0.6
    ),
    text=batted['events'],
    hovertemplate='LA: %{x}°<br>EV: %{y} mph<br>%{text}<extra></extra>'
  ))

  # Add "barrel zone" rectangle (approximate)
  fig.add_shape(
    type="rect",
    x0=8, x1=32,
    y0=95, y1=116,
    line=dict(color="red", width=2, dash="dash"),
    fillcolor="rgba(255,0,0,0.1)"
  )

  fig.add_annotation(
    x=20, y=105,
    text="Barrel Zone",
    showarrow=False,
    font=dict(color="red", size=10)
  )

  fig.update_layout(
    title=f"Exit Velocity vs Launch Angle{' - ' + player_name if player_name else ''}",
    xaxis_title="Launch Angle (°)",
    yaxis_title="Exit Velocity (mph)",
    height=500
  )

  return fig


def create_trend_sparkline(values: list, title: str = "") -> go.Figure:
  """Create a small sparkline chart for trends."""

  fig = go.Figure()

  fig.add_trace(go.Scatter(
    y=values,
    mode='lines',
    line=dict(color='blue', width=2),
    fill='tozeroy',
    fillcolor='rgba(0, 100, 255, 0.2)'
  ))

  fig.update_layout(
    title=title,
    height=100,
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    showlegend=False
  )

  return fig
