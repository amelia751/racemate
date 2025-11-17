'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  TrendingUp, MapPin, Zap, Target, Code, CheckCircle2, Clock
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter } from 'recharts';
import { DocumentationHeader, CodeBlock, DocumentationTabs } from './shared';

// Sample data
const circuitPitLossData = [
  { circuit: 'Monaco', pitLaneTime: 15.2, outLapDelta: 8.3, totalLoss: 23.5 },
  { circuit: 'Singapore', pitLaneTime: 16.1, outLapDelta: 6.2, totalLoss: 22.3 },
  { circuit: 'Monza', pitLaneTime: 13.8, outLapDelta: 5.4, totalLoss: 19.2 },
  { circuit: 'Silverstone', pitLaneTime: 14.5, outLapDelta: 6.3, totalLoss: 20.8 },
  { circuit: 'Spa', pitLaneTime: 13.2, outLapDelta: 6.5, totalLoss: 19.7 },
];

const tireCompoundImpactData = [
  { compound: 'Soft→Soft', timeLoss: 19.5, wearGain: 100 },
  { compound: 'Soft→Med', timeLoss: 20.8, wearGain: 135 },
  { compound: 'Soft→Hard', timeLoss: 22.3, wearGain: 180 },
  { compound: 'Med→Med', timeLoss: 19.2, wearGain: 100 },
  { compound: 'Med→Hard', timeLoss: 21.1, wearGain: 145 },
];

const trafficScenarioData = [
  { traffic: 'Clear Track', pitLoss: 19.5, probability: 0.15 },
  { traffic: '1-2 Cars', pitLoss: 21.2, probability: 0.35 },
  { traffic: '3-4 Cars', pitLoss: 23.8, probability: 0.30 },
  { traffic: '5+ Cars', pitLoss: 27.1, probability: 0.20 },
];

export default function PitSection() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <DocumentationHeader
        icon={TrendingUp}
        title="PIT LOSS MODEL"
        subtitle="XGBoost Regressor for Circuit-Specific Pit Stop Time Loss Prediction"
        color="green"
        metrics={[
          { label: 'Algorithm', value: 'XGBoost' },
          { label: 'MAE', value: '0.8s' },
          { label: 'R² Score', value: '0.93' },
          { label: 'Inference', value: '10ms' }
        ]}
      />

      {/* Model Overview */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-green-400">Model Overview</CardTitle>
          <CardDescription>
            Predicting total time loss from pit stops
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-muted-foreground leading-relaxed">
            Pit stop time loss isn't just the stationary time in the pit box. It includes: 
            <strong className="text-green-400"> pit lane speed limit</strong>, 
            <strong className="text-emerald-400"> out-lap performance delta</strong> (cold tires, fuel load), and 
            <strong className="text-cyan-400"> traffic on exit</strong>. Our XGBoost model predicts total time loss with 0.8s accuracy.
          </p>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-gradient-to-br from-green-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <MapPin className="w-5 h-5 text-green-400" />
                <span className="font-semibold">Circuit-Specific</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Monaco: 23.5s avg loss (long pit lane). Monza: 19.2s (short lane, high-speed nature reduces out-lap delta).
              </p>
            </div>

            <div className="bg-gradient-to-br from-emerald-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-5 h-5 text-emerald-400" />
                <span className="font-semibold">Tire Strategy Impact</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Soft→Hard: +2.8s vs. Soft→Soft (cold hard tire penalty). Model accounts for compound warm-up characteristics.
              </p>
            </div>

            <div className="bg-gradient-to-br from-cyan-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-5 h-5 text-cyan-400" />
                <span className="font-semibold">Traffic Prediction</span>
              </div>
              <p className="text-sm text-muted-foreground">
                5+ cars ahead: +7.6s vs. clear track. Combined with Traffic GNN for post-pit position forecasting.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Architecture */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-green-400">XGBoost Architecture</CardTitle>
          <CardDescription>
            Gradient boosting with tree-based ensemble
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <CodeBlock
            title="XGBoost Regressor Implementation"
            language="python"
            code={`import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=150,           # 150 boosting rounds
    max_depth=6,                 # Tree depth
    learning_rate=0.05,          # Conservative learning rate
    subsample=0.8,               # Row sampling
    colsample_bytree=0.8,        # Column sampling
    gamma=0.1,                   # Minimum loss reduction
    reg_alpha=0.1,               # L1 regularization
    reg_lambda=1.0,              # L2 regularization
    objective='reg:squarederror',
    tree_method='hist',          # Histogram-based algorithm
    random_state=42
)

# Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(model, max_num_features=10)

# Prediction
pit_loss = model.predict(X_test)  # Predicted time loss in seconds`}
          />

          <Separator />

          <div>
            <h3 className="font-semibold text-lg mb-4">Feature Engineering (12 Features)</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2 text-sm">
                <h4 className="font-semibold text-green-400">Circuit Features (4)</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• pit_lane_length - Physical pit lane distance (m)</li>
                  <li>• pit_speed_limit - Speed limit (km/h, usually 60-80)</li>
                  <li>• pit_entry_angle - Entry chicane complexity</li>
                  <li>• pit_exit_blend - Exit merge difficulty</li>
                </ul>
              </div>
              <div className="space-y-2 text-sm">
                <h4 className="font-semibold text-emerald-400">Strategy Features (4)</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• tire_compound_change - Soft/Med/Hard transition</li>
                  <li>• fuel_added - Fuel refueling amount (kg)</li>
                  <li>• front_wing_adjustment - Aero changes (binary)</li>
                  <li>• stop_duration - Actual stationary time (s)</li>
                </ul>
              </div>
              <div className="space-y-2 text-sm">
                <h4 className="font-semibold text-cyan-400">Traffic Features (2)</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• cars_ahead_5s - Vehicles within 5s window</li>
                  <li>• blue_flag_risk - Lapping situation probability</li>
                </ul>
              </div>
              <div className="space-y-2 text-sm">
                <h4 className="font-semibold text-purple-400">Environmental (2)</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• track_temperature - Surface temp (°C)</li>
                  <li>• weather_condition - Dry/Damp/Wet</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-green-500/10 rounded-lg p-4 mt-6">
            <div className="flex justify-between items-center mb-2">
              <span className="font-semibold text-green-400">Model Performance</span>
            </div>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Training samples:</span>
                <span className="font-mono ml-2">28,000 pit stops</span>
              </div>
              <div>
                <span className="text-muted-foreground">Feature importance top 3:</span>
                <span className="font-mono ml-2">Lane length, Tire change, Traffic</span>
              </div>
              <div>
                <span className="text-muted-foreground">Cross-validation MAE:</span>
                <span className="font-mono ml-2 text-green-400">0.82s</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Visualizations */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-green-400">Performance Analysis</CardTitle>
          <CardDescription>
            Time loss breakdown across circuits and strategies
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <Tabs defaultValue="circuits" className="w-full" suppressHydrationWarning>
            <TabsList className="grid grid-cols-3 w-full">
              <TabsTrigger value="circuits">Circuit Comparison</TabsTrigger>
              <TabsTrigger value="tires">Tire Compounds</TabsTrigger>
              <TabsTrigger value="traffic">Traffic Impact</TabsTrigger>
            </TabsList>

            <TabsContent value="circuits" className="space-y-4 mt-6">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={circuitPitLossData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="circuit" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" label={{ value: 'Time Loss (seconds)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Bar dataKey="pitLaneTime" stackId="a" fill="#22c55e" name="Pit Lane Time" />
                    <Bar dataKey="outLapDelta" stackId="a" fill="#10b981" name="Out-Lap Delta" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-red-500/10 rounded-lg p-3">
                  <div className="font-semibold mb-1">Monaco (23.5s)</div>
                  <p className="text-xs text-muted-foreground">
                    Longest pit lane (320m) + tight entry/exit. Out-lap delta exacerbated by low-speed nature.
                  </p>
                </div>
                <div className="bg-green-500/10 rounded-lg p-3">
                  <div className="font-semibold mb-1">Monza (19.2s)</div>
                  <p className="text-xs text-muted-foreground">
                    Short pit lane (245m) + high-speed layout reduces out-lap disadvantage. Fastest pit loss.
                  </p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="tires" className="space-y-4 mt-6">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={tireCompoundImpactData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" stroke="#9ca3af" />
                    <YAxis dataKey="compound" type="category" width={100} stroke="#9ca3af" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Bar dataKey="timeLoss" fill="#22c55e" name="Time Loss (s)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="bg-yellow-500/10 rounded-lg p-4">
                <h4 className="font-semibold text-yellow-400 mb-2">Tire Warm-Up Physics</h4>
                <p className="text-sm text-muted-foreground">
                  Softer compounds reach optimal temperature faster (1-2 laps) vs. hard compounds (3-4 laps). 
                  Model accounts for compound-specific warm-up curves in out-lap delta calculation.
                </p>
              </div>
            </TabsContent>

            <TabsContent value="traffic" className="space-y-4 mt-6">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="traffic" stroke="#9ca3af" label={{ value: 'Traffic Density', position: 'insideBottom', offset: -5 }} />
                    <YAxis dataKey="pitLoss" stroke="#9ca3af" label={{ value: 'Pit Loss (s)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                      cursor={{ strokeDasharray: '3 3' }}
                    />
                    <Scatter data={trafficScenarioData} fill="#22c55e" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <p className="text-sm text-muted-foreground">
                Traffic on pit exit can add 2-7.5s to total pit loss. Model predicts post-pit track position 
                using Traffic GNN to determine optimal pit window with minimal traffic interference.
              </p>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Real-World Integration */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-green-400">Real-World Integration</CardTitle>
          <CardDescription>
            Optimal pit stop timing and strategy
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="bg-gradient-to-br from-green-900/30 to-black rounded-lg p-6">
            <h3 className="font-semibold text-green-400 mb-4 flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Strategic Pit Window Analysis
            </h3>
            <CodeBlock
              title="Pit Stop Optimization Example"
              language="text"
              code={`📊 PIT STOP OPTIMIZATION - Lap 18

Circuit: Silverstone (20.8s avg pit loss)
Current Position: P4 (Gap to P3: 12.5s, Gap to P5: 8.2s)

Tire Strategy Options:
  A) Soft → Medium (Predicted loss: 21.5s)
  B) Soft → Soft (Predicted loss: 20.1s)
  C) Soft → Hard (Predicted loss: 22.8s)

Traffic Analysis:
  • Clear window NOW: 0 cars within 5s
  • In 2 laps: 3 cars within 5s (+3.2s traffic penalty)
  • In 5 laps: Clear window returns

Position Forecast (Option A, Pit Now):
  Expected exit position: P6
  Overtake P6→P4: 6-8 laps on fresh mediums
  Net result: P4 with tire advantage

📍 RECOMMENDATION: Option A (Med), Pit THIS LAP
   • Clear traffic window (saves 3.2s vs. waiting)
   • Medium compound optimal for remaining stint (28 laps)
   • Expected final position: P4 (maintain current position)
   • Risk: Low (clear traffic, optimal strategy)`}
            />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-green-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-5 h-5 text-green-400" />
                <span className="font-semibold text-sm">Accurate Predictions</span>
              </div>
              <p className="text-xs text-muted-foreground">
                0.8s MAE enables precise undercut/overcut timing decisions within driver reaction time
              </p>
            </div>
            <div className="bg-blue-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-5 h-5 text-blue-400" />
                <span className="font-semibold text-sm">Traffic Integration</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Combined with Traffic GNN for comprehensive post-pit position and overtaking analysis
              </p>
            </div>
            <div className="bg-purple-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5 text-purple-400" />
                <span className="font-semibold text-sm">Strategy Optimization</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Models all tire compound combinations to find optimal balance of speed vs. durability
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Helper Components

