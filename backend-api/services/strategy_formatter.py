"""
Professional Race Strategy Formatter
Generates actionable recommendations from ML predictions without LLM
"""

from typing import List, Dict, Any
from datetime import datetime


class StrategyFormatter:
    """Formats ML predictions into professional race strategy recommendations"""
    
    @staticmethod
    def format_recommendations(
        events: List[Dict[str, Any]],
        predictions: Dict[str, Any],
        telemetry: Dict[str, Any]
    ) -> str:
        """Generate professional recommendation text from ML data"""
        
        if not events:
            return "✅ All systems nominal. Continue current strategy."
        
        # Categorize events
        critical_events = [e for e in events if e.get('severity') == 'critical']
        high_events = [e for e in events if e.get('severity') == 'high']
        
        # Build recommendation
        lines = []
        
        # Header based on severity
        if critical_events:
            lines.append("🚨 IMMEDIATE ACTION REQUIRED")
        elif high_events:
            lines.append("⚠️ STRATEGY ADJUSTMENT NEEDED")
        else:
            lines.append("ℹ️ MINOR OPTIMIZATION AVAILABLE")
        
        lines.append("")
        
        # Process critical events first
        for event in critical_events:
            event_type = event.get('type', '').upper()
            message = event.get('message', '')
            
            if 'LOW_FUEL' in event_type or 'FUEL_CRISIS' in event_type:
                fuel_level = telemetry.get('fuel', 0)
                fuel_per_lap = predictions.get('fuel_per_lap', 0.08)
                laps_remaining = int(fuel_level / fuel_per_lap) if fuel_per_lap > 0 else 0
                
                lines.append(f"🔴 FUEL CRITICAL: {fuel_level:.1f}L remaining")
                lines.append(f"   → {laps_remaining} laps of fuel left")
                lines.append(f"   → Consumption: {fuel_per_lap:.3f}L/lap")
                lines.append(f"   📍 ACTION: Box THIS LAP for fuel")
                lines.append("")
                
            elif 'PIT_WINDOW' in event_type:
                lines.append(f"🔴 {message}")
                lines.append(f"   📍 ACTION: Pit within 2 laps for optimal strategy")
                lines.append("")
                
            elif 'ANOMALY' in event_type:
                lines.append(f"🔴 ANOMALY DETECTED")
                lines.append(f"   → {message}")
                lines.append(f"   📍 ACTION: Check telemetry - possible mechanical issue")
                lines.append("")
        
        # Process high-severity events
        for event in high_events:
            event_type = event.get('type', '').upper()
            message = event.get('message', '')
            
            if 'FUEL_CONSUMPTION' in event_type:
                lines.append(f"🟡 FUEL CONSUMPTION SPIKE")
                lines.append(f"   → {message}")
                lines.append(f"   💡 TIP: Lift and coast in high-speed sections")
                lines.append("")
                
            elif 'TIRE' in event_type or 'TYRE' in event_type:
                lines.append(f"🟡 TIRE MANAGEMENT")
                lines.append(f"   → {message}")
                lines.append(f"   💡 TIP: Consider tire change on next pit stop")
                lines.append("")
                
            elif 'HIGH_SPEED' in event_type:
                speed = telemetry.get('speed', 0)
                lines.append(f"🟡 HIGH SPEED ALERT: {speed:.0f} km/h")
                lines.append(f"   → Monitor fuel consumption at this pace")
                lines.append("")
        
        # Add strategic summary
        if critical_events:
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("📊 RACE ENGINEER SUMMARY:")
            lines.append(f"   • Current Lap: {telemetry.get('lap', 0)}")
            lines.append(f"   • Fuel Strategy: {predictions.get('fuel_per_lap', 0.08):.3f}L/lap consumption")
            lines.append(f"   • Next Action: Immediate pit stop recommended")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_optimal_status(telemetry: Dict[str, Any], predictions: Dict[str, Any]) -> str:
        """Format message for optimal conditions"""
        return (
            "✅ OPTIMAL PERFORMANCE\n\n"
            f"All systems operating within parameters:\n"
            f"   • Speed: {telemetry.get('speed', 0):.0f} km/h\n"
            f"   • Fuel: {telemetry.get('fuel', 0):.1f}L\n"
            f"   • Consumption: {predictions.get('fuel_per_lap', 0.08):.3f}L/lap\n\n"
            f"💡 Continue current pace and strategy"
        )

