# Data Exploration Report: Hack the Track 2025

## Overview

This document provides a comprehensive analysis of the Toyota GR Cup racing telemetry and race data from the 2025 season. The dataset includes real racing data from **6 professional motorsports tracks** across North America, covering multiple races with detailed telemetry, lap timing, and race results.

## Dataset Structure

### Tracks Included
1. **Barber Motorsports Park** (Alabama)
2. **Circuit of the Americas (COTA)** (Texas)
3. **Road America** (Wisconsin)
4. **Sebring International Raceway** (Florida)
5. **Sonoma Raceway** (California)
6. **Virginia International Raceway (VIR)** (Virginia)

### Data Volume
- **Total tracks**: 6
- **Total CSV files**: ~94 files
- **Telemetry data size**: 
  - Barber Race 1: ~11.6M rows
  - Barber Race 2: ~11.7M rows
  - Similar volumes for other tracks
- **Total compressed data**: ~885 MB
- **Uncompressed data**: ~2+ GB

## Data File Types

Each track contains the following data categories:

### 1. Telemetry Data Files
**Format**: `R{X}_{track}_telemetry_data.csv` or `{track}_telemetry_R{X}.csv`

**Columns**:
- `expire_at`: Expiration timestamp
- `lap`: Lap number
- `meta_event`: Event identifier (e.g., "I_R06_2025-09-07")
- `meta_session`: Session type (R1, R2, etc.)
- `meta_source`: Data source (kafka:gr-raw)
- `meta_time`: Message reception time
- `original_vehicle_id`: Original vehicle identifier
- `outing`: Outing number
- `telemetry_name`: Name of the telemetry parameter
- `telemetry_value`: Value of the telemetry parameter
- `timestamp`: ECU timestamp
- `vehicle_id`: Vehicle identifier (format: GR86-XXX-XXX)
- `vehicle_number`: Car number

**Available Telemetry Parameters** (12 total):
1. **Speed** - Actual vehicle speed (km/h)
2. **Gear** - Current gear selection
3. **nmot** - Engine RPM
4. **ath** - Throttle blade position (0-100%)
5. **aps** - Accelerator pedal position (0-100%)
6. **pbrake_f** - Front brake pressure (bar)
7. **pbrake_r** - Rear brake pressure (bar)
8. **accx_can** - Longitudinal acceleration (G's, positive = accelerating, negative = braking)
9. **accy_can** - Lateral acceleration (G's, positive = left turn, negative = right turn)
10. **Steering_Angle** - Steering wheel angle (degrees, 0 = straight)
11. **VBOX_Long_Minutes** - GPS longitude (degrees)
12. **VBOX_Lat_Min** - GPS latitude (degrees)
13. **Laptrigger_lapdist_dls** - Distance from start/finish line (meters)

**Data Structure**: 
- Long format (one row per telemetry point per parameter)
- High-frequency sampling (~10-20 Hz for most parameters)
- Approximately 1M+ telemetry points per race per vehicle

**Availability**:
- ✅ Barber: R1, R2
- ✅ COTA: R1, R2
- ✅ Road America: R1, R2
- ⚠️ Sebring: R2 only (R1 missing)
- ✅ Sonoma: R1, R2
- ✅ VIR: R1, R2

### 2. Lap Timing Files

#### Lap Start Times
**Format**: `R{X}_{track}_lap_start.csv` or `{track}_lap_start_time_R{X}.csv`

**Columns**:
- `expire_at`
- `lap`: Lap number
- `meta_event`, `meta_session`, `meta_source`, `meta_time`
- `original_vehicle_id`, `outing`
- `timestamp`: When lap started
- `vehicle_id`, `vehicle_number`

#### Lap End Times
**Format**: `R{X}_{track}_lap_end.csv` or `{track}_lap_end_time_R{X}.csv`

**Columns**: Same as lap start

#### Lap Times
**Format**: `R{X}_{track}_lap_time.csv` or `{track}_lap_time_R{X}.csv`

**Columns**: Same structure with calculated lap times

### 3. Race Results Files

#### Provisional & Official Results
**Format**: `03_Provisional Results_Race {X}_Anonymized.CSV` or `03_Results GR Cup Race {X} Official_Anonymized.CSV`

**Columns** (semicolon-delimited):
- `POSITION`: Finishing position
- `NUMBER`: Car number
- `STATUS`: Race status (Classified, DNF, etc.)
- `LAPS`: Total laps completed
- `TOTAL_TIME`: Total race time (mm:ss.sss)
- `GAP_FIRST`: Time behind leader
- `GAP_PREVIOUS`: Time behind car ahead
- `FL_LAPNUM`: Fastest lap number
- `FL_TIME`: Fastest lap time
- `FL_KPH`: Fastest lap speed (km/h)
- `CLASS`: Racing class (Am)
- `GROUP`, `DIVISION`: GR Cup
- `VEHICLE`: Toyota GR86
- Additional metadata fields (ECM IDs, mostly empty)

#### Results by Class
**Format**: `05_Provisional Results by Class_Race {X}_Anonymized.CSV` or `05_Results by Class GR Cup Race {X} Official_Anonymized.CSV`

Similar structure to overall results, organized by class.

### 4. Best Laps Analysis
**Format**: `99_Best 10 Laps By Driver_Race {X}_Anonymized.CSV`

**Columns** (semicolon-delimited):
- `NUMBER`, `VEHICLE`, `CLASS`
- `TOTAL_DRIVER_LAPS`: Total laps completed by driver
- `BESTLAP_1` through `BESTLAP_10`: Best 10 lap times
- `BESTLAP_X_LAPNUM`: Lap numbers for each best lap
- `AVERAGE`: Average of best 10 laps

**Example data**:
```
Car #2: Best lap 1:38.326 (Lap 5), Average top 10: 1:38.421
Car #13: Best lap 1:37.428 (Lap 8)
```

### 5. Endurance Analysis with Sections
**Format**: `23_AnalysisEnduranceWithSections_Race {X}_Anonymized.CSV`

Contains detailed sector/section timing for endurance analysis.

### 6. Weather Data
**Format**: `26_Weather_Race {X}_Anonymized.CSV`

**Columns** (semicolon-delimited):
- `TIME_UTC_SECONDS`: Unix timestamp
- `TIME_UTC_STR`: Human-readable time
- `AIR_TEMP`: Air temperature (°C)
- `TRACK_TEMP`: Track temperature (°C)
- `HUMIDITY`: Relative humidity (%)
- `PRESSURE`: Atmospheric pressure (mbar)
- `WIND_SPEED`: Wind speed (m/s)
- `WIND_DIRECTION`: Wind direction (degrees)
- `RAIN`: Rain indicator (0 = no rain)

**Sample rate**: ~60 seconds

**Example conditions**:
- Barber Race 1: Air temp 29-30°C, Humidity 56-57%, Wind 2-5 m/s

## Data Quality & Characteristics

### Strengths
✅ **High-frequency telemetry**: Multiple data points per second
✅ **Multi-dimensional**: Speed, acceleration, braking, GPS, steering
✅ **Real professional racing data**: Actual competition data from Toyota GR Cup
✅ **Multiple tracks**: Diverse track layouts and characteristics
✅ **Multiple races**: Practice and race sessions
✅ **Weather context**: Environmental conditions for each race
✅ **Ground truth**: Official race results and timing for validation

### Limitations & Notes
⚠️ **Anonymized**: Driver names removed, vehicles identified by numbers
⚠️ **Missing data**: Sebring Race 1 telemetry not available
⚠️ **Data format**: Mixed delimiters (comma vs semicolon) across files
⚠️ **Time synchronization**: meta_time vs timestamp differences
⚠️ **Track temperature**: Often 0 or missing in weather data
⚠️ **Long format telemetry**: Requires pivoting for time-series analysis

## Key Insights & Patterns

### Race Characteristics

#### Lap Times
- **Fastest laps**: ~1:37-1:38 range at Barber
- **Lap count**: 27 laps typical for Barber races
- **Consistency**: Top drivers maintain ~1-2 second lap time range
- **Performance gaps**: 10-15 seconds between leaders and mid-pack over full race

#### Vehicle Performance
- **Speed**: Toyota GR86 spec series (similar vehicles)
- **Competition level**: Amateur class
- **Field size**: ~20-30 cars per race
- **Classification**: Most drivers complete full race distance

#### Environmental Conditions
- **Temperature**: 25-35°C typical
- **Humidity**: 40-65% range
- **Wind**: Variable 2-8 m/s
- **Track conditions**: Predominantly dry (rain = 0)

### Telemetry Patterns

#### Speed & Acceleration
- High-frequency data capture (~10-20 Hz)
- Clear braking zones (negative accx_can)
- Corner entry/exit visible in accy_can
- Gear changes correlated with speed

#### GPS & Track Position
- Precise GPS coordinates for track mapping
- Lap distance tracking from start/finish
- Can reconstruct racing line and track layout

#### Driver Inputs
- Throttle position (aps) vs blade position (ath) for analysis
- Brake pressure distribution (front vs rear)
- Steering angle for line analysis

## Data Usage Recommendations

### For Analysis Projects
1. **Pivot telemetry data**: Convert long format to wide for time-series
2. **Synchronize timestamps**: Use meta_time for consistency
3. **Join datasets**: Link telemetry → lap times → race results
4. **Handle missing data**: Check for gaps in telemetry streams
5. **Track-specific analysis**: Each track has unique characteristics

### For Machine Learning
1. **Feature engineering**: Derive features from raw telemetry
   - Brake points, apex speed, corner exit acceleration
   - Throttle application timing
   - Steering smoothness metrics
2. **Sequence modeling**: Time-series prediction for lap times
3. **Classification**: Optimal vs suboptimal driving patterns
4. **Clustering**: Identify driving styles

### For Visualization
1. **Track maps**: Use GPS data to create circuit layouts
2. **Speed traces**: Overlay speed by track position
3. **Race position charts**: Position changes over time
4. **Weather impact**: Correlate conditions with performance
5. **Comparative analysis**: Driver vs driver, lap vs lap

## File Naming Conventions

### Pattern Examples
- Telemetry: `R{race}_track_telemetry_data.csv`
- Lap timing: `R{race}_track_lap_{start|end|time}.csv`
- Results: `03_Provisional Results_Race {X}_Anonymized.CSV`
- Weather: `26_Weather_Race {X}_Anonymized.CSV`
- Best laps: `99_Best 10 Laps By Driver_Race {X}_Anonymized.CSV`

### Track Name Variations
- `barber`, `cota`, `road_america`, `sebring`, `sonoma`, `vir`

## Technical Specifications

### File Formats
- **Telemetry & Timing**: CSV with comma delimiters
- **Results & Analysis**: CSV with semicolon delimiters
- **Encoding**: UTF-8
- **Line endings**: Mixed (Unix/Windows)

### Data Types
- **Timestamps**: ISO 8601 format (YYYY-MM-DDTHH:MM:SS.sssZ)
- **Numbers**: Decimal with period separator
- **Times**: MM:SS.sss format for lap times
- **IDs**: String format (e.g., "GR86-002-000")

## Additional Data Sources

According to the hackathon documentation, additional official timing results are available at:
- **Series**: SRO
- **2025 Season**: TGRNA GR CUP NORTH AMERICA
- **2024 Season**: Toyota GR Cup

This can be used for cross-validation and additional context.

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total tracks | 6 |
| Total races | ~12 (2 per track) |
| Telemetry parameters | 13 unique |
| Telemetry rows per race | ~11M |
| Cars per race | ~20-30 |
| Laps per race | 20-30 |
| Data points per car | ~1M+ |
| CSV files | ~94 |
| Total data size | ~2+ GB uncompressed |

## Recommended Next Steps

1. ✅ **Data validation**: Check for completeness across all tracks
2. ✅ **Schema standardization**: Handle mixed delimiters and formats
3. ✅ **Data cleaning**: Remove duplicates, handle nulls
4. ✅ **Feature extraction**: Derive racing metrics from raw telemetry
5. ✅ **Track characterization**: Analyze each circuit's unique properties
6. ✅ **Performance baseline**: Establish benchmarks for good lap times
7. ✅ **Integration**: Combine telemetry with results for full picture

---

**Dataset Source**: [https://trddev.com/hackathon-2025/](https://trddev.com/hackathon-2025/)

**Event**: Hack the Track 2025 - Toyota GR Racing

