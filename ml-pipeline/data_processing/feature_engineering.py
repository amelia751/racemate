import pandas as pd
import numpy as np
from typing import Dict, List
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Generate derived features per IDEA.md spec"""
    
    def __init__(self):
        self.features = []
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling windows and time-based features"""
        
        print("  Adding temporal features...")
        df = df.sort_values(['vehicle_id', 'timestamp'])
        
        # Group by vehicle
        for signal in ['speed', 'nmot', 'aps', 'pbrake_f', 'accx_can', 'accy_can']:
            if signal in df.columns:
                # 5-second rolling mean (assuming 20Hz → 100 samples)
                df[f'{signal}_rolling_mean_5s'] = (
                    df.groupby('vehicle_id')[signal]
                    .transform(lambda x: x.rolling(window=100, min_periods=10).mean())
                )
                
                # EWMA slope
                df[f'{signal}_ewma_slope'] = (
                    df.groupby('vehicle_id')[signal]
                    .transform(lambda x: x.ewm(span=20).mean().diff())
                )
        
        return df
    
    def add_energy_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Braking energy and lateral load per spec"""
        
        print("  Adding energy metrics...")
        
        # Braking energy per sector
        df['brake_energy'] = (
            (df['pbrake_f'].fillna(0) + df['pbrake_r'].fillna(0)) * 
            df['speed'].fillna(0)
        )
        
        # Lateral load proxy
        df['lateral_load'] = (
            df['accy_can'].abs().fillna(0) * df['speed'].fillna(0)
        )
        
        # Cumulative per lap
        df['cum_brake_energy'] = (
            df.groupby(['vehicle_id', 'lap'])['brake_energy']
            .cumsum()
        )
        
        df['cum_lateral_load'] = (
            df.groupby(['vehicle_id', 'lap'])['lateral_load']
            .cumsum()
        )
        
        return df
    
    def add_throttle_discipline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Throttle smoothness and discipline metrics"""
        
        print("  Adding throttle discipline metrics...")
        
        if 'aps' in df.columns:
            # Throttle variance (smoothness)
            df['throttle_variance'] = (
                df.groupby(['vehicle_id', 'lap'])['aps']
                .transform(lambda x: x.rolling(window=50, min_periods=10).var())
            )
            
            # Time to full throttle (simplified)
            df['on_full_throttle'] = (df['aps'] > 95).astype(int)
        
        return df
    
    def add_steering_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Steering smoothness and jerk"""
        
        print("  Adding steering metrics...")
        
        if 'Steering_Angle' in df.columns:
            # Steering rate of change
            df['steer_rate'] = (
                df.groupby('vehicle_id')['Steering_Angle']
                .diff()
            )
            
            # Steering jerk (2nd derivative)
            df['steer_jerk'] = (
                df.groupby('vehicle_id')['steer_rate']
                .diff()
            )
            
            # Smoothness score (inverse of jerk variance)
            df['steer_smoothness'] = (
                df.groupby(['vehicle_id', 'lap'])['steer_jerk']
                .transform(lambda x: 1.0 / (x.var() + 0.001))
            )
        
        return df
    
    def add_track_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """Micro-sector IDs (10m bins) from lap distance"""
        
        print("  Adding track position features...")
        
        if 'Laptrigger_lapdist_dls' in df.columns:
            # 10m micro-sectors (handle NaN values)
            df['micro_sector_id'] = (
                (df['Laptrigger_lapdist_dls'].fillna(0) // 10).astype(int)
            )
        
        return df
    
    def add_tire_stress_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combined tire stress indicator"""
        
        print("  Adding tire stress proxy...")
        
        # Acceleration magnitude
        df['acc_magnitude'] = np.sqrt(
            df['accx_can'].fillna(0)**2 + 
            df['accy_can'].fillna(0)**2
        )
        
        # Tire stress = acc peaks × dwell time
        df['tire_stress_proxy'] = (
            df['acc_magnitude'] * (df['acc_magnitude'] > 0.8).astype(int)
        )
        
        # Cumulative per lap
        df['cum_tire_stress'] = (
            df.groupby(['vehicle_id', 'lap'])['tire_stress_proxy']
            .cumsum()
        )
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering"""
        
        print("\nEngineering features...")
        original_cols = len(df.columns)
        
        df = self.add_temporal_features(df)
        df = self.add_energy_metrics(df)
        df = self.add_throttle_discipline(df)
        df = self.add_steering_metrics(df)
        df = self.add_track_position(df)
        df = self.add_tire_stress_proxy(df)
        
        new_cols = len(df.columns)
        print(f"\n✓ Feature engineering complete!")
        print(f"  Original features: {original_cols}")
        print(f"  New features: {new_cols}")
        print(f"  Added: {new_cols - original_cols}")
        
        return df

if __name__ == "__main__":
    print("Testing Feature Engineering...")
    
    from csv_parser import TelemetryParser
    
    parser = TelemetryParser()
    engineer = FeatureEngineer()
    
    # Test with sample data
    df = parser.load_track_session('barber-motorsports-park', 'R1')
    if not df.empty:
        df_wide = parser.pivot_to_wide(df)
        
        # Sample small subset for testing
        sample_df = df_wide.head(10000).copy()
        
        result_df = engineer.engineer_all_features(sample_df)
        
        print(f"\nFinal shape: {result_df.shape}")
        print(f"Sample features: {list(result_df.columns[-10:])}")

