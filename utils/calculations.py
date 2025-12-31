# utils/calculations.py
"""Statistical calculation utilities for baseball metrics."""

import numpy as np
import pandas as pd
from typing import Union, List, Optional


def calculate_ops(obp: float, slg: float) -> float:
  """Calculate OPS (On-base Plus Slugging)."""
  return obp + slg


def calculate_iso(slg: float, avg: float) -> float:
  """Calculate ISO (Isolated Power) = SLG - AVG."""
  return slg - avg


def calculate_babip(h: int, hr: int, ab: int, so: int, sf: int = 0) -> float:
  """
  Calculate BABIP (Batting Average on Balls In Play).
  BABIP = (H - HR) / (AB - K - HR + SF)
  """
  denominator = ab - so - hr + sf
  if denominator <= 0:
    return 0.0
  return (h - hr) / denominator


def calculate_whip(bb: int, h: int, ip: float) -> float:
  """Calculate WHIP (Walks + Hits per Inning Pitched)."""
  if ip <= 0:
    return 0.0
  return (bb + h) / ip


def calculate_era(er: int, ip: float) -> float:
  """Calculate ERA (Earned Run Average)."""
  if ip <= 0:
    return 0.0
  return (er * 9) / ip


def calculate_k_rate(so: int, pa: int) -> float:
  """Calculate strikeout rate (K%)."""
  if pa <= 0:
    return 0.0
  return so / pa


def calculate_bb_rate(bb: int, pa: int) -> float:
  """Calculate walk rate (BB%)."""
  if pa <= 0:
    return 0.0
  return bb / pa


def calculate_k_per_9(so: int, ip: float) -> float:
  """Calculate strikeouts per 9 innings (K/9)."""
  if ip <= 0:
    return 0.0
  return (so * 9) / ip


def calculate_bb_per_9(bb: int, ip: float) -> float:
  """Calculate walks per 9 innings (BB/9)."""
  if ip <= 0:
    return 0.0
  return (bb * 9) / ip


def calculate_fip(hr: int, bb: int, hbp: int, so: int, ip: float,
                  fip_constant: float = 3.10) -> float:
  """
  Calculate FIP (Fielding Independent Pitching).
  FIP = ((13*HR)+(3*(BB+HBP))-(2*K))/IP + FIP_constant

  Note: FIP constant varies by season, 3.10 is approximate.
  """
  if ip <= 0:
    return 0.0
  return ((13 * hr) + (3 * (bb + hbp)) - (2 * so)) / ip + fip_constant


def calculate_woba(bb: int, hbp: int, singles: int, doubles: int,
                   triples: int, hr: int, ab: int, sf: int = 0,
                   weights: Optional[dict] = None) -> float:
  """
  Calculate wOBA (Weighted On-Base Average).

  Default weights are approximate and vary by season.
  """
  if weights is None:
    weights = {
      'bb': 0.69,
      'hbp': 0.72,
      '1b': 0.88,
      '2b': 1.27,
      '3b': 1.62,
      'hr': 2.10
    }

  denominator = ab + bb - hbp + sf
  if denominator <= 0:
    return 0.0

  numerator = (
          weights['bb'] * bb +
          weights['hbp'] * hbp +
          weights['1b'] * singles +
          weights['2b'] * doubles +
          weights['3b'] * triples +
          weights['hr'] * hr
  )

  return numerator / denominator


def calculate_wrc_plus(woba: float, league_woba: float,
                       woba_scale: float = 1.157,
                       league_r_pa: float = 0.12,
                       park_factor: float = 100) -> float:
  """
  Calculate wRC+ (Weighted Runs Created Plus).

  Simplified formula. Actual calculation requires more league factors.
  """
  if league_woba <= 0:
    return 0.0

  wrc = ((woba - league_woba) / woba_scale + league_r_pa) * 100
  return wrc * (100 / park_factor)


def calculate_rolling_average(values: pd.Series, window: int) -> pd.Series:
  """Calculate rolling average for a series."""
  return values.rolling(window=window, min_periods=1).mean()


def calculate_percentile(value: float, distribution: pd.Series) -> int:
  """Calculate percentile rank of a value within a distribution."""
  return int(round((distribution < value).mean() * 100))


def normalize_stat(player_value: float, league_avg: float,
                   invert: bool = False) -> float:
  """
  Normalize a stat to league average (100 = league average).

  Args:
      player_value: The player's stat value
      league_avg: League average for the stat
      invert: If True, lower values are better (like ERA)
  """
  if league_avg <= 0 or player_value <= 0:
    return 100.0

  if invert:
    return (league_avg / player_value) * 100
  else:
    return (player_value / league_avg) * 100


def calculate_expected_stats(launch_speed: float, launch_angle: float) -> dict:
  """
  Estimate expected batting average and slugging based on exit velo and launch angle.

  This is a simplified estimation. Actual xBA/xSLG calculations are more complex.
  """
  # Simplified model based on typical Statcast correlations

  # Barrel zone: 95+ mph, 10-30 degrees
  is_barrel = launch_speed >= 95 and 10 <= launch_angle <= 30

  # Sweet spot: 8-32 degrees
  is_sweet_spot = 8 <= launch_angle <= 32

  # Base xBA calculation (very simplified)
  if is_barrel:
    xba = 0.750 + (launch_speed - 95) * 0.02
  elif is_sweet_spot and launch_speed >= 90:
    xba = 0.400 + (launch_speed - 90) * 0.03
  elif launch_speed >= 90:
    xba = 0.200 + (launch_speed - 90) * 0.02
  else:
    xba = 0.100 + launch_speed * 0.001

  xba = min(max(xba, 0), 1.0)

  # xSLG estimation
  if is_barrel:
    xslg = 1.500 + (launch_speed - 95) * 0.05
  elif is_sweet_spot:
    xslg = 0.600 + (launch_speed - 90) * 0.04
  else:
    xslg = xba * 1.2

  xslg = min(max(xslg, 0), 4.0)

  return {
    'xBA': round(xba, 3),
    'xSLG': round(xslg, 3),
    'is_barrel': is_barrel,
    'is_sweet_spot': is_sweet_spot
  }


def grade_stat(value: float, percentile: int) -> str:
  """Convert a percentile to a letter grade."""
  if percentile >= 90:
    return 'A+'
  elif percentile >= 80:
    return 'A'
  elif percentile >= 70:
    return 'B+'
  elif percentile >= 60:
    return 'B'
  elif percentile >= 50:
    return 'C+'
  elif percentile >= 40:
    return 'C'
  elif percentile >= 30:
    return 'D'
  else:
    return 'F'
