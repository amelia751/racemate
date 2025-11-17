'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  AlertTriangle, Activity, Zap, Target, Code, CheckCircle2, XCircle
} from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend, LineChart, Line } from 'recharts';
import { DocumentationHeader, CodeBlock, DocumentationTabs } from './shared';

// Sample data
const anomalyScoreData = [
  { lap: 1, score: 0.15, threshold: 0.5, anomaly: false },
  { lap: 5, score: 0.22, threshold: 0.5, anomaly: false },
  { lap: 10, score: 0.18, threshold: 0.5, anomaly: false },
  { lap: 12, score: 0.78, threshold: 0.5, anomaly: true },
  { lap: 15, score: 0.31, threshold: 0.5, anomaly: false },
  { lap: 18, score: 0.25, threshold: 0.5, anomaly: false },
  { lap: 22, score: 0.85, threshold: 0.5, anomaly: true },
  { lap: 25, score: 0.28, threshold: 0.5, anomaly: false },
];

const anomalyTypeData = [
  { type: 'Gearbox Issue', count: 45, severity: 'critical' },
  { type: 'Wheel Spin', count: 128, severity: 'high' },
  { type: 'Sensor Malfunction', count: 67, severity: 'medium' },
  { type: 'Driver Error', count: 156, severity: 'high' },
  { type: 'Lockup', count: 203, severity: 'medium' },
];

const featureSpace2D = [
  { feature1: 8500, feature2: 65, anomaly: false, label: 'Normal' },
  { feature1: 8200, feature2: 72, anomaly: false, label: 'Normal' },
  { feature1: 8800, feature2: 68, anomaly: false, label: 'Normal' },
  { feature1: 9200, feature2: 35, anomaly: true, label: 'RPM Spike' },
  { feature1: 5200, feature2: 88, anomaly: true, label: 'Low RPM High Speed' },
  { feature1: 8600, feature2: 70, anomaly: false, label: 'Normal' },
  { feature1: 11500, feature2: 42, anomaly: true, label: 'Critical RPM' },
];

export default function AnomalySection() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <DocumentationHeader
        icon={AlertTriangle}
        title="ANOMALY DETECTOR"
        subtitle="Isolation Forest for Unsupervised Detection of Unusual Telemetry Patterns"
        color="rose"
        metrics={[
          { label: 'Algorithm', value: 'Isolation Forest' },
          { label: 'Precision', value: '94%' },
          { label: 'Recall', value: '91%' },
          { label: 'Inference', value: '15ms' }
        ]}
      />

      {/* Model Overview */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-rose-300 to-pink-400 bg-clip-text text-transparent">Model Overview</CardTitle>
          <CardDescription className="text-slate-400">
            Unsupervised detection of mechanical failures and driver errors
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-muted-foreground leading-relaxed">
            Anomaly detection doesn't require labeled "failure" data. <strong className="text-red-400">Isolation Forest</strong> learns 
            normal telemetry patterns and flags deviations. Critical for detecting: 
            <strong className="text-pink-400"> mechanical failures</strong> (gearbox, engine), 
            <strong className="text-orange-400"> sensor malfunctions</strong>, and 
            <strong className="text-yellow-400"> driver errors</strong> (spins, lockups) in real-time.
          </p>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gradient-to-br from-red-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-5 h-5 text-red-400" />
                <span className="font-semibold">24D Feature Space</span>
              </div>
              <p className="text-sm text-muted-foreground">
                RPM, speed, throttle, brake, G-forces, tire temps, fuel consumption - comprehensive telemetry monitoring across 24 dimensions.
              </p>
            </div>

            <div className="bg-gradient-to-br from-pink-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-5 h-5 text-pink-400" />
                <span className="font-semibold">Isolation Principle</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Anomalies are "easier to isolate" - they require fewer splits in decision trees. Normal points cluster densely, anomalies stand alone.
              </p>
            </div>

            <div className="bg-gradient-to-br from-orange-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-5 h-5 text-orange-400" />
                <span className="font-semibold">Unsupervised Learning</span>
              </div>
              <p className="text-sm text-muted-foreground">
                No labeled failure data needed. Model learns normal distribution from 250K clean laps, flags outliers automatically.
              </p>
            </div>

            <div className="bg-gradient-to-br from-yellow-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-5 h-5 text-yellow-400" />
                <span className="font-semibold">Real-Time Detection</span>
              </div>
              <p className="text-sm text-muted-foreground">
                15ms inference enables frame-by-frame (10Hz) anomaly scoring. Alerts trigger immediately when score exceeds threshold (0.5).
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Architecture */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-rose-300 to-pink-400 bg-clip-text text-transparent">Isolation Forest Architecture</CardTitle>
          <CardDescription className="text-slate-400">
            Ensemble of isolation trees
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <CodeBlock
            title="Scikit-learn Isolation Forest"
            language="python"
            code={`from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=100,           # 100 isolation trees
    max_samples='auto',          # Use 256 samples per tree
    contamination=0.05,          # Expected anomaly rate (5%)
    max_features=1.0,            # Use all features
    bootstrap=False,             # No bootstrap sampling
    random_state=42,
    n_jobs=-1                    # Parallel processing
)

# Train on normal data
model.fit(X_normal)

# Anomaly scoring
anomaly_scores = model.decision_function(X_test)  # Negative = anomaly
is_anomaly = model.predict(X_test)  # -1 = anomaly, 1 = normal

# Path length analysis (for interpretability)
path_lengths = model.score_samples(X_test)  # Shorter path = anomaly`}
          />

          <Separator />

          <div>
            <h3 className="font-semibold text-lg mb-4">How Isolation Forest Works</h3>
            <div className="space-y-4">
              <StepCard
                number={1}
                title="Random Feature Selection"
                description="For each tree, randomly select a feature from the 24D space (e.g., RPM, speed, throttle)"
              />
              <StepCard
                number={2}
                title="Random Split Value"
                description="Choose a random split value between min and max of selected feature"
              />
              <StepCard
                number={3}
                title="Recursive Partitioning"
                description="Continue splitting until each point is isolated or max depth reached"
              />
              <StepCard
                number={4}
                title="Path Length Calculation"
                description="Anomalies require fewer splits to isolate (shorter path length)"
              />
              <StepCard
                number={5}
                title="Anomaly Score"
                description="Average path length across all trees. Shorter = more anomalous. Threshold at 0.5."
              />
            </div>
          </div>

          <div className="bg-red-500/10 rounded-lg p-4 mt-6">
            <h4 className="font-semibold text-red-400 mb-2">Why Isolation Forest vs. Other Methods?</h4>
            <div className="grid grid-cols-2 gap-3 text-sm mt-3">
              <div className="bg-black/40 rounded p-2">
                <strong className="text-green-400">✓ Fast:</strong> O(n log n) complexity, 15ms inference
              </div>
              <div className="bg-black/40 rounded p-2">
                <strong className="text-green-400">✓ High-Dimensional:</strong> Handles 24D space effectively
              </div>
              <div className="bg-black/40 rounded p-2">
                <strong className="text-green-400">✓ Unsupervised:</strong> No labeled failures needed
              </div>
              <div className="bg-black/40 rounded p-2">
                <strong className="text-green-400">✓ Interpretable:</strong> Path length reveals which features caused anomaly
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Visualizations */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-rose-300 to-pink-400 bg-clip-text text-transparent">Performance Analysis</CardTitle>
          <CardDescription className="text-slate-400">
            Anomaly detection across race scenarios
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <Tabs defaultValue="scores" className="w-full">
            <TabsList className="grid grid-cols-3 w-full">
              <TabsTrigger value="scores">Anomaly Scores</TabsTrigger>
              <TabsTrigger value="types">Anomaly Types</TabsTrigger>
              <TabsTrigger value="space">Feature Space</TabsTrigger>
            </TabsList>

            <TabsContent value="scores" className="space-y-4 mt-6">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={anomalyScoreData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="lap" stroke="#9ca3af" label={{ value: 'Lap Number', position: 'insideBottom', offset: -5 }} />
                    <YAxis stroke="#9ca3af" domain={[0, 1]} label={{ value: 'Anomaly Score', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="score" stroke="#ef4444" strokeWidth={3} name="Anomaly Score" />
                    <Line type="monotone" dataKey="threshold" stroke="#22c55e" strokeWidth={2} strokeDasharray="5 5" name="Threshold" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-red-500/10 rounded-lg p-3">
                  <div className="font-semibold mb-1 flex items-center gap-2">
                    <XCircle className="w-4 h-4" />
                    Lap 12: Score 0.78 (Anomaly)
                  </div>
                  <p className="text-xs text-muted-foreground">
                    RPM spike + low speed combination detected. Potential gearbox issue or wheel spin event.
                  </p>
                </div>
                <div className="bg-red-500/10 rounded-lg p-3">
                  <div className="font-semibold mb-1 flex items-center gap-2">
                    <XCircle className="w-4 h-4" />
                    Lap 22: Score 0.85 (Anomaly)
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Sudden fuel consumption drop. Sensor malfunction or fuel system issue flagged.
                  </p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="types" className="space-y-4 mt-6">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={anomalyTypeData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" stroke="#9ca3af" />
                    <YAxis dataKey="type" type="category" width={150} stroke="#9ca3af" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Bar dataKey="count" fill="#ef4444" name="Occurrences" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div className="bg-red-500/10 rounded p-2">
                  <strong>Gearbox (45):</strong> RPM mismatch with speed, critical severity
                </div>
                <div className="bg-orange-500/10 rounded p-2">
                  <strong>Wheel Spin (128):</strong> High RPM, low speed, traction loss
                </div>
                <div className="bg-yellow-500/10 rounded p-2">
                  <strong>Driver Error (156):</strong> Lockups, spins, off-track excursions
                </div>
              </div>
            </TabsContent>

            <TabsContent value="space" className="space-y-4 mt-6">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="feature1" stroke="#9ca3af" label={{ value: 'RPM', position: 'insideBottom', offset: -5 }} />
                    <YAxis dataKey="feature2" stroke="#9ca3af" label={{ value: 'Speed (km/h)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                      cursor={{ strokeDasharray: '3 3' }}
                    />
                    <Scatter 
                      data={featureSpace2D.filter(d => !d.anomaly)} 
                      fill="#22c55e" 
                      name="Normal" 
                    />
                    <Scatter 
                      data={featureSpace2D.filter(d => d.anomaly)} 
                      fill="#ef4444" 
                      name="Anomaly" 
                    />
                    <Legend />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <p className="text-sm text-muted-foreground">
                2D projection (RPM vs. Speed) shows normal points clustering densely. Anomalies (red) are isolated: 
                high RPM + low speed, low RPM + high speed, or extreme RPM values.
              </p>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Real-World Integration */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-rose-300 to-pink-400 bg-clip-text text-transparent">Real-World Integration</CardTitle>
          <CardDescription className="text-slate-400">
            Immediate alerts for critical anomalies
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="bg-gradient-to-br from-red-900/30 to-black rounded-lg p-6">
            <h3 className="font-semibold text-red-400 mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              Critical Anomaly Alert Example
            </h3>
            <CodeBlock
              title="Critical Anomaly Alert Example"
              language="text"
              code={`🔴 CRITICAL ANOMALY DETECTED - Lap 18

Anomaly Score: 0.87 (Threshold: 0.5)
Severity: CRITICAL

Affected Features:
  • RPM: 11,200 (Avg: 8,500, +32% deviation)
  • Speed: 45 km/h (Avg: 180, -75% deviation)
  • Throttle: 92% (inconsistent with speed)
  • G-Force: 0.8g (expected 3.2g at this throttle)

Pattern Analysis:
  ⚠️ High RPM + Low Speed = Gearbox issue or wheel spin
  ⚠️ High throttle with low speed = Traction loss
  ⚠️ Low G-force = Not accelerating despite throttle input

📍 RECOMMENDED ACTION:
   1. Check telemetry for mechanical failure
   2. Driver instruction: Reduce throttle, recover traction
   3. Monitor for escalation (pit stop may be required)
   4. Alert team: Potential gearbox or drivetrain issue

Historical Context:
   • Similar pattern at Lap 12 (resolved)
   • Frequency increasing (2 events in 6 laps)
   • Consider precautionary pit stop if persists`}
            />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-green-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-5 h-5 text-green-400" />
                <span className="font-semibold text-sm">High Precision</span>
              </div>
              <p className="text-xs text-muted-foreground">
                94% precision minimizes false positives, ensuring teams only respond to real issues
              </p>
            </div>
            <div className="bg-blue-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-5 h-5 text-blue-400" />
                <span className="font-semibold text-sm">Real-Time Monitoring</span>
              </div>
              <p className="text-xs text-muted-foreground">
                15ms inference at 10Hz telemetry rate enables immediate anomaly flagging
              </p>
            </div>
            <div className="bg-purple-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-5 h-5 text-purple-400" />
                <span className="font-semibold text-sm">Interpretable</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Path length analysis reveals which features contributed to anomaly detection
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Helper Components

function StepCard({ number, title, description }: any) {
  return (
    <div className="flex gap-3 items-start">
      <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-red-500/20 flex items-center justify-center font-bold text-red-400">
        {number}
      </div>
      <div className="flex-1">
        <h4 className="font-semibold text-sm mb-1">{title}</h4>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
    </div>
  );
}

