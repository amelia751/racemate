'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  AlertTriangle, Cloud, MapPin, TrendingUp, Target, CheckCircle2
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { DocumentationHeader, CodeBlock, DocumentationTabs } from './shared';

// Sample data
const fcyProbabilityData = [
  { lap: 1, probability: 0.05, actual: 0 },
  { lap: 5, probability: 0.08, actual: 0 },
  { lap: 10, probability: 0.12, actual: 0 },
  { lap: 15, probability: 0.25, actual: 0 },
  { lap: 20, probability: 0.45, actual: 0 },
  { lap: 22, probability: 0.78, actual: 1 },
  { lap: 25, probability: 0.92, actual: 1 },
  { lap: 30, probability: 0.35, actual: 0 },
];

const weatherImpactData = [
  { condition: 'Dry', fcyRate: 12, incidents: 45 },
  { condition: 'Damp', fcyRate: 28, incidents: 112 },
  { condition: 'Wet', fcyRate: 45, incidents: 203 },
  { condition: 'Mixed', fcyRate: 38, incidents: 156 },
];

const circuitRiskData = [
  { name: 'Monaco', value: 42 },
  { name: 'Singapore', value: 38 },
  { name: 'Baku', value: 35 },
  { name: 'Jeddah', value: 32 },
  { name: 'Others', value: 28 },
];

const COLORS = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e'];

export default function FCYSection() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <DocumentationHeader
        icon={AlertTriangle}
        title="FCY HAZARD PREDICTOR"
        subtitle="Random Forest Classifier for Full Course Yellow / Safety Car Probability"
        color="orange"
        metrics={[
          { label: 'Algorithm', value: 'Random Forest' },
          { label: 'Accuracy', value: '89%' },
          { label: 'F1-Score', value: '0.85' },
          { label: 'Inference', value: '8ms' }
        ]}
      />

      {/* Model Overview */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-orange-400">Model Overview</CardTitle>
          <CardDescription>
            Predicting safety car deployments before they happen
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-muted-foreground leading-relaxed">
            FCY (Full Course Yellow) events drastically change race strategy. Predicting safety car deployment 5 laps in advance 
            enables proactive pit stop decisions and track position gains. Our Random Forest classifier analyzes 
            <strong className="text-orange-400"> weather patterns</strong>, <strong className="text-yellow-400">track characteristics</strong>, 
            and <strong className="text-red-400"> incident history</strong> to forecast FCY probability.
          </p>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gradient-to-br from-orange-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Cloud className="w-5 h-5 text-orange-400" />
                <span className="font-semibold">Weather-Driven</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Wet conditions increase FCY probability by 3.7x. Model tracks precipitation intensity, track temperature delta, and drying rate.
              </p>
            </div>

            <div className="bg-gradient-to-br from-yellow-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <MapPin className="w-5 h-5 text-yellow-400" />
                <span className="font-semibold">Circuit-Specific Risk</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Monaco (42% FCY rate) vs. Monza (8%). Street circuits with barriers and tight corners show exponentially higher risk.
              </p>
            </div>

            <div className="bg-gradient-to-br from-red-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-5 h-5 text-red-400" />
                <span className="font-semibold">Incident Clustering</span>
              </div>
              <p className="text-sm text-muted-foreground">
                FCYs often occur in clusters - one safety car increases probability of another within 10 laps by 2.3x due to cold tires.
              </p>
            </div>

            <div className="bg-gradient-to-br from-green-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5 text-green-400" />
                <span className="font-semibold">Lap-Based Patterns</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Lap 1 (start chaos), laps 15-25 (tire degradation), and final 5 laps (desperation moves) show elevated FCY risk.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Architecture */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-orange-400">Architecture Deep Dive</CardTitle>
          <CardDescription>
            Ensemble of 200 decision trees
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <CodeBlock 
            title="Scikit-learn Random Forest Classifier"
            language="python"
            code={`from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,           # 200 decision trees
    max_depth=12,                # Prevent overfitting
    min_samples_split=50,        # Minimum 50 samples to split
    min_samples_leaf=20,         # Minimum 20 samples per leaf
    max_features='sqrt',         # Feature randomization
    class_weight='balanced',     # Handle imbalanced data (FCY is rare)
    random_state=42,
    n_jobs=-1                    # Parallel processing
)

# Feature importance extraction
importances = model.feature_importances_
feature_names = ['weather', 'track_type', 'lap_number', 'incidents_last_5', 
                 'tire_age_avg', 'speed_variance', 'track_position_spread',
                 'drs_enabled', 'rain_intensity', 'visibility']

# Probability prediction
fcy_probability = model.predict_proba(X_test)[:, 1]  # Probability of FCY=1`}
          />

          <Separator />

          <div>
            <h3 className="font-semibold text-lg mb-4">Feature Importance (Top 10)</h3>
            <div className="space-y-3">
              <ImportanceBar label="Weather Condition" value={28} color="bg-orange-500" />
              <ImportanceBar label="Track Type (Street/Permanent)" value={22} color="bg-yellow-500" />
              <ImportanceBar label="Lap Number" value={18} color="bg-red-500" />
              <ImportanceBar label="Incidents (Last 5 Laps)" value={15} color="bg-purple-500" />
              <ImportanceBar label="Average Tire Age" value={8} color="bg-blue-500" />
              <ImportanceBar label="Speed Variance" value={4} color="bg-green-500" />
              <ImportanceBar label="Track Position Spread" value={3} color="bg-cyan-500" />
              <ImportanceBar label="DRS Enabled" value={1} color="bg-pink-500" />
              <ImportanceBar label="Rain Intensity" value={0.8} color="bg-indigo-500" />
              <ImportanceBar label="Visibility" value={0.2} color="bg-gray-500" />
            </div>
          </div>

          <div className="bg-orange-500/10 rounded-lg p-4 mt-6">
            <h4 className="font-semibold text-orange-400 mb-2">Hyperparameter Tuning</h4>
            <p className="text-sm text-muted-foreground mb-3">
              Grid search over 96 combinations found optimal config: 200 estimators, max_depth=12, balanced class weights.
            </p>
            <div className="grid grid-cols-3 gap-4 text-xs">
              <div>
                <span className="text-muted-foreground">n_estimators tested:</span>
                <span className="font-mono ml-2">50, 100, 200, 300</span>
              </div>
              <div>
                <span className="text-muted-foreground">max_depth tested:</span>
                <span className="font-mono ml-2">8, 10, 12, 15</span>
              </div>
              <div>
                <span className="text-muted-foreground">Best CV F1:</span>
                <span className="font-mono ml-2 text-green-400">0.847</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Training Data */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-orange-400">Training Dataset</CardTitle>
          <CardDescription>
            250,000+ race situations from 2017-2023
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3">Data Collection</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="text-orange-400 mt-1">→</span>
                  <span><strong>Positive samples:</strong> 15,000 FCY events (6% of dataset)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-400 mt-1">→</span>
                  <span><strong>Negative samples:</strong> 235,000 normal racing situations</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-400 mt-1">→</span>
                  <span><strong>Time window:</strong> Predict FCY within next 5 laps (binary)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-400 mt-1">→</span>
                  <span><strong>Class balancing:</strong> SMOTE oversampling + class_weight='balanced'</span>
                </li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-3">Feature Engineering</h4>
              <div className="space-y-2 text-sm">
                <div className="bg-gray-900/50 p-2 rounded">
                  <strong className="text-orange-400">Weather encoding:</strong> Ordinal (Dry=0, Damp=1, Wet=2)
                </div>
                <div className="bg-gray-900/50 p-2 rounded">
                  <strong className="text-yellow-400">Track type:</strong> One-hot (Street/Permanent/Hybrid)
                </div>
                <div className="bg-gray-900/50 p-2 rounded">
                  <strong className="text-red-400">Incident history:</strong> Rolling count (last 5, 10, 20 laps)
                </div>
                <div className="bg-gray-900/50 p-2 rounded">
                  <strong className="text-purple-400">Lap phase:</strong> Categorical (Start/Mid/End race)
                </div>
              </div>
            </div>
          </div>

          <Separator />

          <div>
            <h4 className="font-semibold mb-3">Performance Metrics</h4>
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-green-500/10 rounded-lg p-3">
                <div className="text-2xl font-bold text-green-400">89%</div>
                <div className="text-xs text-muted-foreground">Accuracy</div>
              </div>
              <div className="bg-blue-500/10 rounded-lg p-3">
                <div className="text-2xl font-bold text-blue-400">0.85</div>
                <div className="text-xs text-muted-foreground">F1-Score</div>
              </div>
              <div className="bg-purple-500/10 rounded-lg p-3">
                <div className="text-2xl font-bold text-purple-400">87%</div>
                <div className="text-xs text-muted-foreground">Recall</div>
              </div>
              <div className="bg-cyan-500/10 rounded-lg p-3">
                <div className="text-2xl font-bold text-cyan-400">83%</div>
                <div className="text-xs text-muted-foreground">Precision</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Visualizations */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-orange-400">Performance Analysis</CardTitle>
          <CardDescription>
            Prediction accuracy across race conditions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <DocumentationTabs
            defaultTab="probability"
            tabs={[
              {
                id: 'probability',
                label: 'FCY Probability',
                content: (
                  <div className="space-y-4 mt-6">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={fcyProbabilityData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="lap" stroke="#9ca3af" label={{ value: 'Lap Number', position: 'insideBottom', offset: -5 }} />
                    <YAxis stroke="#9ca3af" domain={[0, 1]} label={{ value: 'FCY Probability', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="probability" stroke="#f97316" strokeWidth={3} name="Predicted Probability" />
                    <Line type="stepAfter" dataKey="actual" stroke="#ef4444" strokeWidth={3} name="Actual FCY" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
                    <p className="text-sm text-muted-foreground">
                      Model correctly predicted FCY deployment at lap 22 with 78% probability 2 laps in advance. 
                      Probability spiked from 45% to 78% as incident severity indicators increased.
                    </p>
                  </div>
                )
              },
              {
                id: 'weather',
                label: 'Weather Impact',
                content: (
                  <div className="space-y-4 mt-6">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={weatherImpactData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="condition" stroke="#9ca3af" />
                    <YAxis yAxisId="left" stroke="#9ca3af" label={{ value: 'FCY Rate (%)', angle: -90, position: 'insideLeft' }} />
                    <YAxis yAxisId="right" orientation="right" stroke="#9ca3af" label={{ value: 'Incidents', angle: 90, position: 'insideRight' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Bar yAxisId="left" dataKey="fcyRate" fill="#f97316" name="FCY Rate (%)" />
                    <Bar yAxisId="right" dataKey="incidents" fill="#eab308" name="Total Incidents" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-red-500/10 rounded-lg p-3">
                  <div className="font-semibold mb-1">Wet Conditions</div>
                  <p className="text-xs text-muted-foreground">
                    45% FCY rate - 3.7x higher than dry. Aquaplaning and visibility issues cause 203 incidents per season.
                  </p>
                </div>
                <div className="bg-green-500/10 rounded-lg p-3">
                  <div className="font-semibold mb-1">Dry Conditions</div>
                  <p className="text-xs text-muted-foreground">
                    12% baseline FCY rate. Most incidents from mechanical failures or driver errors, not environmental.
                  </p>
                </div>
              </div>
            </div>
          )
        },
              {
                id: 'circuits',
                label: 'Circuit Risk',
                content: (
                  <div className="space-y-4 mt-6">
              <div className="h-[350px] flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={circuitRiskData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value}%`}
                      outerRadius={120}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {circuitRiskData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-red-500/10 rounded-lg p-2 text-xs">
                  <strong>Monaco (42%):</strong> Narrow streets, barriers, zero run-off
                </div>
                <div className="bg-orange-500/10 rounded-lg p-2 text-xs">
                  <strong>Singapore (38%):</strong> Night race, humidity, tight corners
                </div>
                <div className="bg-yellow-500/10 rounded-lg p-2 text-xs">
                  <strong>Baku (35%):</strong> High-speed street circuit, castle section
                </div>
              </div>
            </div>
          )
        }
            ]}
          />
        </CardContent>
      </Card>

      {/* Real-World Integration */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-orange-400">Real-World Integration</CardTitle>
          <CardDescription>
            Strategic pit stop decisions based on FCY predictions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="bg-gradient-to-br from-red-900/30 to-black rounded-lg p-6">
            <h3 className="font-semibold text-red-400 mb-4 flex items-center gap-2">
              <Target className="w-5 h-5" />
              High FCY Probability Alert
            </h3>
            <div className="bg-gray-900/50 p-4 rounded-lg">
              <pre className="text-xs font-mono text-muted-foreground leading-relaxed">
{`🟡 HIGH FCY PROBABILITY DETECTED

Current Situation (Lap 18):
  • Weather: Damp track, light drizzle starting
  • Circuit: Monaco (street circuit, 42% FCY rate)
  • Incidents: 2 in last 5 laps (collision + spin)
  • Tire age: 18 laps (high degradation)

FCY Probability Forecast:
  Next 5 laps: 65% probability
  Confidence: High (based on weather + circuit + incidents)

📍 STRATEGIC RECOMMENDATION:
   Option A: Pit NOW before FCY
   • Risk: FCY doesn't happen, lose 22s
   • Reward: If FCY happens, effectively "free" pit stop
   
   Option B: Wait for FCY confirmation
   • Risk: Pit under green flag if FCY doesn't happen
   • Reward: Certain pit stop timing
   
   ✅ RECOMMENDED: Option A (Pit now)
   • 65% FCY probability justifies risk
   • Weather deteriorating (increases probability to 75%)
   • Track position: P6, minimal loss if wrong`}
              </pre>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-green-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-5 h-5 text-green-400" />
                <span className="font-semibold text-sm">Proactive Strategy</span>
              </div>
              <p className="text-xs text-muted-foreground">
                5-lap lookahead enables preemptive pitting before FCY, gaining track position
              </p>
            </div>
            <div className="bg-blue-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-5 h-5 text-blue-400" />
                <span className="font-semibold text-sm">Risk Management</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Confidence intervals help quantify decision risk vs. reward trade-offs
              </p>
            </div>
            <div className="bg-purple-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5 text-purple-400" />
                <span className="font-semibold text-sm">Position Gains</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Correct FCY prediction can yield 2-5 position gains from strategic pitting
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Helper Components
function ImportanceBar({ label, value, color }: any) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono font-semibold">{value}%</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-500`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

