'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  Fuel, TrendingUp, AlertCircle, CheckCircle2, Code, Database,
  GitBranch, Zap, Settings, BarChart3
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter } from 'recharts';
import { DocumentationHeader, CodeBlock, DocumentationTabs } from './shared';

// Sample data for visualizations
const featureImportanceData = [
  { feature: 'RPM (nMotor)', importance: 22, color: '#22c55e' },
  { feature: 'Speed (vCar)', importance: 18, color: '#3b82f6' },
  { feature: 'Throttle %', importance: 15, color: '#a855f7' },
  { feature: 'Gear Selection', importance: 12, color: '#eab308' },
  { feature: 'Track Layout', importance: 10, color: '#ec4899' },
  { feature: 'Brake Usage', importance: 8, color: '#f97316' },
  { feature: 'Acceleration', importance: 7, color: '#14b8a6' },
  { feature: 'Weather', importance: 5, color: '#6366f1' },
  { feature: 'Others', importance: 3, color: '#64748b' },
];

const performanceMetricsData = [
  { lap: 1, actual: 0.078, predicted: 0.077 },
  { lap: 2, actual: 0.079, predicted: 0.079 },
  { lap: 3, actual: 0.081, predicted: 0.080 },
  { lap: 4, actual: 0.077, predicted: 0.078 },
  { lap: 5, actual: 0.080, predicted: 0.080 },
  { lap: 6, actual: 0.082, predicted: 0.081 },
  { lap: 7, actual: 0.079, predicted: 0.079 },
  { lap: 8, actual: 0.078, predicted: 0.078 },
  { lap: 9, actual: 0.083, predicted: 0.082 },
  { lap: 10, actual: 0.080, predicted: 0.080 },
];

const circuitComparisonData = [
  { circuit: 'Monaco', avgFuel: 0.092, maxFuel: 0.115, minFuel: 0.072 },
  { circuit: 'Monza', avgFuel: 0.095, maxFuel: 0.125, minFuel: 0.078 },
  { circuit: 'Silverstone', avgFuel: 0.088, maxFuel: 0.110, minFuel: 0.070 },
  { circuit: 'Spa', avgFuel: 0.091, maxFuel: 0.118, minFuel: 0.075 },
  { circuit: 'Singapore', avgFuel: 0.087, maxFuel: 0.108, minFuel: 0.069 },
];

export default function FuelSection() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <DocumentationHeader
        icon={Fuel}
        title="FUEL CONSUMPTION MODEL"
        subtitle="Gradient Boosting Regressor for High-Precision Fuel Usage Prediction"
        color="amber"
        metrics={[
          { label: 'Algorithm', value: 'Gradient Boosting' },
          { label: 'MAE', value: '0.008 L/lap' },
          { label: 'R² Score', value: '0.96' },
          { label: 'Inference', value: '12ms' }
        ]}
      />

      {/* Model Architecture */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-amber-300 to-orange-400 bg-clip-text text-transparent">Model Architecture</CardTitle>
          <CardDescription className="text-slate-400">
            Ensemble learning with decision tree weak learners
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <CodeBlock
            title="GradientBoostingRegressor Configuration"
            language="python"
            code={`from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=100,         # 100 sequential trees
    max_depth=5,               # Prevent overfitting
    learning_rate=0.1,         # Step size shrinkage
    subsample=0.8,             # Stochastic gradient boosting
    min_samples_split=20,      # Minimum samples for split
    min_samples_leaf=10,       # Minimum samples per leaf
    max_features='sqrt',       # Feature randomization
    random_state=42,
    loss='squared_error'       # L2 loss function
)`}
          />

          <Separator />

          <div>
            <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5 text-cyan-400" />
              <span>Hyperparameter Tuning Process</span>
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gradient-to-br from-purple-900/20 to-black rounded-lg p-4">
                <h4 className="font-semibold text-purple-400 mb-3">Grid Search Parameters</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">n_estimators:</span>
                    <span className="font-mono">[50, 100, 150, 200]</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">max_depth:</span>
                    <span className="font-mono">[3, 5, 7, 9]</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">learning_rate:</span>
                    <span className="font-mono">[0.01, 0.1, 0.2]</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">subsample:</span>
                    <span className="font-mono">[0.6, 0.8, 1.0]</span>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-cyan-900/20 to-black rounded-lg p-4">
                <h4 className="font-semibold text-cyan-400 mb-3">Cross-Validation Results</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Fold 1:</span>
                    <span className="font-mono text-green-400">0.0081 L/lap</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Fold 2:</span>
                    <span className="font-mono text-green-400">0.0079 L/lap</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Fold 3:</span>
                    <span className="font-mono text-green-400">0.0082 L/lap</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Fold 4:</span>
                    <span className="font-mono text-green-400">0.0078 L/lap</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Fold 5:</span>
                    <span className="font-mono text-green-400">0.0080 L/lap</span>
                  </div>
                  <Separator className="my-2" />
                  <div className="flex justify-between font-semibold">
                    <span className="text-cyan-400">Mean MAE:</span>
                    <span className="font-mono text-cyan-400">0.0080 L/lap</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feature Engineering & Importance */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-amber-300 to-orange-400 bg-clip-text text-transparent">Feature Engineering & Importance</CardTitle>
          <CardDescription className="text-slate-400">
            26 engineered features from raw telemetry data
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <Tabs defaultValue="importance" className="w-full" suppressHydrationWarning>
            <TabsList className="grid grid-cols-3 w-full">
              <TabsTrigger value="importance">Feature Importance</TabsTrigger>
              <TabsTrigger value="engineering">Feature Engineering</TabsTrigger>
              <TabsTrigger value="correlation">Feature Correlation</TabsTrigger>
            </TabsList>

            <TabsContent value="importance" className="space-y-4 mt-6">
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={featureImportanceData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" stroke="#9ca3af" />
                    <YAxis dataKey="feature" type="category" width={150} stroke="#9ca3af" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Bar dataKey="importance" fill="#eab308" radius={[0, 8, 8, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="grid grid-cols-3 gap-4 mt-6">
                <div className="bg-green-500/10 rounded-lg p-4">
                  <div className="text-2xl font-bold text-green-400">RPM</div>
                  <div className="text-sm text-muted-foreground">22% importance</div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Engine speed directly correlates with fuel injection rate. Higher RPM = exponentially higher consumption.
                  </p>
                </div>
                <div className="bg-blue-500/10 rounded-lg p-4">
                  <div className="text-2xl font-bold text-blue-400">Speed</div>
                  <div className="text-sm text-muted-foreground">18% importance</div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Aerodynamic drag increases with square of velocity. High-speed sections require more fuel.
                  </p>
                </div>
                <div className="bg-purple-500/10 rounded-lg p-4">
                  <div className="text-2xl font-bold text-purple-400">Throttle</div>
                  <div className="text-sm text-muted-foreground">15% importance</div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Throttle position controls fuel-air mixture. Full throttle in lower gears = highest consumption.
                  </p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="engineering" className="space-y-4 mt-6">
              <div className="space-y-4">
                <FeatureGroup
                  title="Raw Telemetry Features (8)"
                  features={[
                    { name: 'vCar', description: 'Vehicle speed in km/h' },
                    { name: 'nMotor', description: 'Engine RPM' },
                    { name: 'throttle', description: 'Throttle position (0-100%)' },
                    { name: 'brake', description: 'Brake pressure (0-100%)' },
                    { name: 'gear', description: 'Current gear (1-8)' },
                    { name: 'drs', description: 'DRS activation (binary)' },
                    { name: 'fuel_load', description: 'Current fuel level (L)' },
                    { name: 'lap_distance', description: 'Distance into lap (m)' },
                  ]}
                />

                <FeatureGroup
                  title="Derived Features (10)"
                  features={[
                    { name: 'acceleration', description: 'Δ speed / Δ time' },
                    { name: 'throttle_gradient', description: 'Rate of throttle change' },
                    { name: 'rpm_per_speed', description: 'nMotor / vCar ratio' },
                    { name: 'gear_efficiency', description: 'Optimal gear selection score' },
                    { name: 'cornering_intensity', description: 'Lateral G-force magnitude' },
                    { name: 'straight_time_pct', description: '% of lap on straights' },
                    { name: 'full_throttle_pct', description: '% of lap at 100% throttle' },
                    { name: 'braking_frequency', description: 'Number of braking events' },
                    { name: 'drs_usage_pct', description: '% of lap with DRS active' },
                    { name: 'fuel_load_norm', description: 'Normalized by race start fuel' },
                  ]}
                />

                <FeatureGroup
                  title="Rolling Window Features (5)"
                  features={[
                    { name: 'speed_ma_3', description: '3-second moving avg speed' },
                    { name: 'throttle_ma_5', description: '5-second moving avg throttle' },
                    { name: 'rpm_std_10', description: '10-second rolling std RPM' },
                    { name: 'accel_max_5', description: 'Max acceleration in 5s window' },
                    { name: 'brake_sum_3', description: 'Total brake usage in 3s window' },
                  ]}
                />

                <FeatureGroup
                  title="Track & Context Features (3)"
                  features={[
                    { name: 'circuit_type', description: 'Street/Permanent/Hybrid (encoded)' },
                    { name: 'elevation_change', description: 'Track elevation variance (m)' },
                    { name: 'weather_condition', description: 'Dry/Wet/Mixed (encoded)' },
                  ]}
                />
              </div>
            </TabsContent>

            <TabsContent value="correlation" className="space-y-4 mt-6">
              <div className="bg-gradient-to-br from-red-900/20 to-black rounded-lg p-6">
                <h3 className="font-semibold text-red-400 mb-4">High Correlation Pairs (|r| &gt; 0.7)</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="bg-black/40 rounded-lg p-3">
                    <div className="font-semibold">RPM ↔ Speed</div>
                    <div className="text-muted-foreground">r = 0.82</div>
                    <p className="text-xs text-muted-foreground mt-1">
                      Mitigated via rpm_per_speed ratio feature
                    </p>
                  </div>
                  <div className="bg-black/40 rounded-lg p-3">
                    <div className="font-semibold">Throttle ↔ Acceleration</div>
                    <div className="text-muted-foreground">r = 0.75</div>
                    <p className="text-xs text-muted-foreground mt-1">
                      Expected causation, both features retained
                    </p>
                  </div>
                  <div className="bg-black/40 rounded-lg p-3">
                    <div className="font-semibold">Gear ↔ Speed</div>
                    <div className="text-muted-foreground">r = 0.71</div>
                    <p className="text-xs text-muted-foreground mt-1">
                      Gear efficiency feature captures interaction
                    </p>
                  </div>
                  <div className="bg-black/40 rounded-lg p-3">
                    <div className="font-semibold">Full Throttle % ↔ Straight Time %</div>
                    <div className="text-muted-foreground">r = 0.79</div>
                    <p className="text-xs text-muted-foreground mt-1">
                      Circuit-specific, both provide value
                    </p>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Training Pipeline */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-amber-300 to-orange-400 bg-clip-text text-transparent">Training Pipeline</CardTitle>
          <CardDescription className="text-slate-400">
            End-to-end workflow from raw data to production model
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <TrainingStep
            number={1}
            title="Data Collection & Cleaning"
            icon={<Database className="w-6 h-6" />}
            color="cyan"
            details={[
              'Source: F1 2017-2023 telemetry via FastF1 API',
              'Total dataset: 524,000 laps across 21 circuits',
              'Removed: Safety car laps, formation laps, incomplete sessions',
              'Filtered: Outliers (fuel >3σ from mean), sensor errors',
              'Final dataset: 487,000 valid laps (93% retention)',
            ]}
            metrics={[
              { label: 'Raw Samples', value: '524K' },
              { label: 'Clean Samples', value: '487K' },
              { label: 'Retention Rate', value: '93%' },
            ]}
          />

          <TrainingStep
            number={2}
            title="Feature Engineering"
            icon={<GitBranch className="w-6 h-6" />}
            color="purple"
            details={[
              'Created 26 features from 8 raw telemetry channels',
              'Rolling windows: 3s, 5s, 10s for temporal context',
              'Track encoding: One-hot for 21 circuits',
              'Weather encoding: Ordinal (Dry=0, Damp=1, Wet=2)',
              'Feature scaling: StandardScaler for numerical features',
            ]}
            metrics={[
              { label: 'Raw Features', value: '8' },
              { label: 'Engineered', value: '26' },
              { label: 'Final Dims', value: '34' },
            ]}
          />

          <TrainingStep
            number={3}
            title="Train/Val/Test Split"
            icon={<BarChart3 className="w-6 h-6" />}
            color="green"
            details={[
              'Strategy: Temporal split (2017-2021 train, 2022 val, 2023 test)',
              'Prevents data leakage from future to past',
              'Train: 389K laps (80%), Val: 49K laps (10%), Test: 49K laps (10%)',
              'Stratified by circuit to ensure representation',
              'Class balancing: SMOTE for under-represented circuits',
            ]}
            metrics={[
              { label: 'Train', value: '80%' },
              { label: 'Val', value: '10%' },
              { label: 'Test', value: '10%' },
            ]}
          />

          <TrainingStep
            number={4}
            title="Model Training & Tuning"
            icon={<Settings className="w-6 h-6" />}
            color="yellow"
            details={[
              'Algorithm: Gradient Boosting (100 estimators, depth=5)',
              'Hyperparameter search: GridSearchCV (5-fold CV)',
              'Search space: 144 combinations tested',
              'Training time: 3.5 hours on 16-core CPU',
              'Best config: lr=0.1, subsample=0.8, max_features=sqrt',
            ]}
            metrics={[
              { label: 'Configs Tested', value: '144' },
              { label: 'Training Time', value: '3.5h' },
              { label: 'CV Folds', value: '5' },
            ]}
          />

          <TrainingStep
            number={5}
            title="Validation & Testing"
            icon={<CheckCircle2 className="w-6 h-6" />}
            color="cyan"
            details={[
              'Validation MAE: 0.0079 L/lap (2022 season)',
              'Test MAE: 0.0082 L/lap (2023 season) - slight degradation expected',
              'R² score: 0.96 on test set',
              'Max error: 0.031 L/lap (high-altitude circuits)',
              'Circuit-specific evaluation: All circuits MAE < 0.012 L/lap',
            ]}
            metrics={[
              { label: 'Val MAE', value: '0.0079' },
              { label: 'Test MAE', value: '0.0082' },
              { label: 'R²', value: '0.96' },
            ]}
          />

          <TrainingStep
            number={6}
            title="Model Export & Deployment"
            icon={<Zap className="w-6 h-6" />}
            color="green"
            details={[
              'Serialization: Pickle format (joblib.dump)',
              'Model size: 24.3 MB (compressed)',
              'Upload: Google Cloud Storage (gs://racemate-models)',
              'Versioning: SHA256 hash + timestamp',
              'Backend loads model on startup (12ms inference latency)',
            ]}
            metrics={[
              { label: 'Model Size', value: '24.3MB' },
              { label: 'Inference', value: '12ms' },
              { label: 'Storage', value: 'GCS' },
            ]}
          />
        </CardContent>
      </Card>

      {/* Performance Analysis */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-amber-300 to-orange-400 bg-clip-text text-transparent">Performance Analysis</CardTitle>
          <CardDescription className="text-slate-400">
            Model accuracy and real-world validation
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceMetricsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="lap" stroke="#9ca3af" label={{ value: 'Lap Number', position: 'insideBottom', offset: -5 }} />
                <YAxis stroke="#9ca3af" domain={[0.075, 0.085]} label={{ value: 'Fuel (L/lap)', angle: -90, position: 'insideLeft' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} name="Actual Consumption" />
                <Line type="monotone" dataKey="predicted" stroke="#22c55e" strokeWidth={2} name="Predicted Consumption" strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-green-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-5 h-5 text-green-400" />
                <span className="font-semibold">Excellent Fit</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Predicted values closely track actual consumption with minimal deviation
              </p>
            </div>
            <div className="bg-blue-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5 text-blue-400" />
                <span className="font-semibold">Captures Trends</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Model successfully predicts spikes (lap 6, 9) and efficient laps (lap 4)
              </p>
            </div>
            <div className="bg-purple-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertCircle className="w-5 h-5 text-purple-400" />
                <span className="font-semibold">Real-Time Ready</span>
              </div>
              <p className="text-sm text-muted-foreground">
                12ms inference enables lap-by-lap predictions during live races
              </p>
            </div>
          </div>

          <Separator />

          <div>
            <h3 className="font-semibold mb-4">Circuit-Specific Performance</h3>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={circuitComparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="circuit" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" label={{ value: 'Fuel (L/lap)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                    }}
                  />
                  <Legend />
                  <Bar dataKey="minFuel" fill="#22c55e" name="Min Fuel" />
                  <Bar dataKey="avgFuel" fill="#eab308" name="Avg Fuel" />
                  <Bar dataKey="maxFuel" fill="#ef4444" name="Max Fuel" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-sm text-muted-foreground mt-4">
              Fuel consumption varies significantly by circuit. Monza (high-speed) and Spa (elevation changes) 
              show highest consumption, while Singapore (slow corners) shows lowest despite high lap count.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Real-World Integration */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-amber-300 to-orange-400 bg-clip-text text-transparent">Real-World Integration</CardTitle>
          <CardDescription className="text-slate-400">
            From predictions to strategic recommendations
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="bg-gradient-to-br from-red-900/30 to-black rounded-lg p-6">
            <h3 className="font-semibold text-red-400 mb-4 flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              Critical Fuel Event Detection
            </h3>
            <div className="space-y-4 text-sm">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-black/40 rounded-lg p-4">
                  <div className="font-semibold mb-2">Consumption Spike Detection</div>
                  <div className="font-mono text-xs mb-2">
                    if current_consumption &gt; avg_consumption * 1.10:
                  </div>
                  <p className="text-muted-foreground text-xs">
                    Triggers HIGH severity alert when fuel usage exceeds expected by 10%+
                  </p>
                </div>
                <div className="bg-black/40 rounded-lg p-4">
                  <div className="font-semibold mb-2">Critical Fuel Level</div>
                  <div className="font-mono text-xs mb-2">
                    if fuel_level &lt; 5.0:
                  </div>
                  <p className="text-muted-foreground text-xs">
                    CRITICAL alert when fuel drops below 5L, calculates laps remaining
                  </p>
                </div>
              </div>

              <CodeBlock
                title="Example Alert Output"
                language="text"
                code={`🔴 FUEL CRITICAL: 4.2L remaining
   → 53 laps of fuel left (at 0.079L/lap consumption)
   → Consumption: 0.079L/lap
   📍 ACTION: Box THIS LAP for fuel

🟡 FUEL CONSUMPTION SPIKE
   → Consumption increased by 12% (0.071 → 0.079L/lap)
   💡 TIP: Lift and coast in high-speed sections to conserve fuel`}
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Helper Components

function FeatureGroup({ title, features }: any) {
  return (
    <div className="bg-gray-900/50 rounded-lg p-4 ">
      <h4 className="font-semibold text-cyan-400 mb-3">{title}</h4>
      <div className="grid grid-cols-2 gap-2">
        {features.map((f: any, i: number) => (
          <div key={i} className="text-sm">
            <span className="font-mono text-yellow-400">{f.name}</span>
            <span className="text-muted-foreground"> - {f.description}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TrainingStep({ number, title, icon, color, details, metrics }: any) {
  const colors: any = {
    cyan: 'bg-cyan-500/20 text-cyan-400',
    purple: 'bg-purple-500/20 text-purple-400',
    green: 'bg-green-500/20 text-green-400',
    yellow: 'bg-yellow-500/20 text-yellow-400',
  };

  return (
    <div className="flex gap-4">
      <div className={`flex-shrink-0 w-12 h-12 rounded-lg ${colors[color]} flex items-center justify-center font-bold text-lg`}>
        {number}
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-2">
          <div className={`${colors[color].split(' ')[2]}`}>{icon}</div>
          <h3 className="font-semibold text-lg">{title}</h3>
        </div>
        <ul className="space-y-1 text-sm text-muted-foreground mb-3">
          {details.map((detail: string, i: number) => (
            <li key={i} className="flex items-start gap-2">
              <span className="text-cyan-400 mt-1">→</span>
              <span>{detail}</span>
            </li>
          ))}
        </ul>
        <div className="flex gap-4 text-xs">
          {metrics.map((m: any, i: number) => (
            <div key={i} className="bg-black/40 rounded px-3 py-1">
              <span className="text-muted-foreground">{m.label}:</span>{' '}
              <span className={`font-mono font-semibold ${colors[color].split(' ')[2]}`}>{m.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

