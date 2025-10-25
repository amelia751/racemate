import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import settings

class TelemetryParser:
    """Parse raw telemetry CSVs into structured format"""
    
    TELEMETRY_SIGNALS = [
        'speed', 'gear', 'nmot', 'aps', 'ath',
        'pbrake_f', 'pbrake_r', 'accx_can', 'accy_can',
        'Steering_Angle', 'VBOX_Lat_Min', 'VBOX_Long_Minutes',
        'Laptrigger_lapdist_dls'
    ]
    
    def __init__(self, data_root: str = None):
        self.data_root = Path(data_root) if data_root else Path(settings.local_data_path)
        
    def load_track_session(self, track: str, race: str) -> pd.DataFrame:
        """Load and parse telemetry for a track/race"""
        
        # Find telemetry file
        track_path = self.data_root / track
        if not track_path.exists():
            print(f"Track path not found: {track_path}")
            return pd.DataFrame()
        
        # Try multiple search patterns to handle different naming conventions
        search_patterns = [
            f"*{race}*telemetry*.csv",  # Standard: R1_track_telemetry_data.csv
            f"*telemetry*{race}*.csv",  # Alternate: track_telemetry_R1.csv
        ]
        
        telemetry_files = []
        for pattern in search_patterns:
            files = list(track_path.rglob(pattern))
            # Filter out MacOS metadata files
            files = [f for f in files if not f.name.startswith('._')]
            telemetry_files.extend(files)
        
        # Remove duplicates
        telemetry_files = list(set(telemetry_files))
        
        if not telemetry_files:
            print(f"No telemetry for {track} {race}")
            return pd.DataFrame()
        
        file_path = telemetry_files[0]
        print(f"Loading: {file_path.name}")
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Parse timestamps
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if 'meta_time' in df.columns:
                df['meta_time'] = pd.to_datetime(df['meta_time'], errors='coerce')
            
            print(f"  Loaded {len(df)} rows")
            return df
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            return pd.DataFrame()
    
    def pivot_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert long format (one row per signal) to wide format"""
        
        if df.empty:
            return pd.DataFrame()
        
        try:
            # Pivot: each timestamp gets all signals as columns
            wide_df = df.pivot_table(
                index=['timestamp', 'vehicle_id', 'vehicle_number', 'lap'],
                columns='telemetry_name',
                values='telemetry_value',
                aggfunc='first'
            ).reset_index()
            
            # Sort by vehicle and time
            wide_df = wide_df.sort_values(['vehicle_id', 'timestamp'])
            
            print(f"  Pivoted: {len(wide_df)} rows, {len(wide_df.columns)} columns")
            
            return wide_df
        except Exception as e:
            print(f"  Error pivoting: {e}")
            return pd.DataFrame()
    
    def validate_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check data quality"""
        stats = {}
        
        for signal in self.TELEMETRY_SIGNALS:
            if signal in df.columns:
                coverage = df[signal].notna().mean()
                stats[signal] = coverage
                
                if coverage < 0.5:
                    print(f"  ⚠️  Low coverage for {signal}: {coverage:.1%}")
        
        return stats
    
    def process_all_tracks(self) -> pd.DataFrame:
        """Process all available tracks and races"""
        
        tracks = [
            'barber-motorsports-park',
            'circuit-of-the-americas',
            'road-america',
            'sebring',
            'sonoma',
            'virginia-international-raceway'
        ]
        
        all_data = []
        
        for track in tracks:
            for race in ['R1', 'R2']:
                print(f"\nProcessing {track} - {race}")
                
                # Load raw
                df_raw = self.load_track_session(track, race)
                
                if df_raw.empty:
                    continue
                
                # Pivot to wide
                df_wide = self.pivot_to_wide(df_raw)
                
                if df_wide.empty:
                    continue
                
                # Add metadata
                df_wide['track'] = track
                df_wide['race'] = race
                df_wide['session_id'] = f"{track}_{race}"
                
                # Validate
                stats = self.validate_signals(df_wide)
                
                all_data.append(df_wide)
        
        # Combine all
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"\n✓ Total processed: {len(combined)} rows")
            print(f"✓ Vehicles: {combined['vehicle_id'].nunique()}")
            print(f"✓ Sessions: {combined['session_id'].nunique()}")
            return combined
        
        return pd.DataFrame()

if __name__ == "__main__":
    print("Testing Telemetry Parser...")
    parser = TelemetryParser()
    
    # Test with one track
    df = parser.load_track_session('barber-motorsports-park', 'R1')
    if not df.empty:
        df_wide = parser.pivot_to_wide(df)
        print(f"\nSample data shape: {df_wide.shape}")
        print(f"Columns: {list(df_wide.columns[:10])}")

