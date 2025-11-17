'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  Fuel, Zap, Gauge, AlertTriangle, TrendingUp, Network, User, Cpu,
  CheckCircle2, ArrowRight, Database, Cloud, Sparkles
} from 'lucide-react';
import { DocumentationHeader } from './shared';

export default function OverviewSection() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <DocumentationHeader
        icon={Cpu}
        title="COGNIRACE ML PIPELINE"
        subtitle="Next-Generation Race Strategy Intelligence Platform"
        color="cyan"
        metrics={[
          { label: 'Specialized ML Models', value: '8' },
          { label: 'Real-Time Inference', value: '<100ms' },
          { label: 'Training Data Points', value: '1M+' }
        ]}
      />

      {/* Executive Summary */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-cyan-300 to-cyan-400 bg-clip-text text-transparent">Executive Summary</CardTitle>
          <CardDescription className="text-slate-400">
            Real-time race strategy optimization through event-driven machine learning
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-slate-300 leading-relaxed">
            COGNIRACE represents a paradigm shift in motorsport strategy analysis. Unlike traditional telemetry
            systems that simply display data, our platform uses <strong className="text-cyan-300">8 specialized machine learning models</strong> working
            in concert to detect critical race events and generate actionable strategic recommendations in real-time.
          </p>

          <div className="bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-xl p-6">
            <h3 className="font-semibold text-lg mb-4 flex items-center gap-2 text-slate-200">
              <Sparkles className="w-5 h-5 text-amber-400" />
              <span>Core Innovation: Event-Driven Architecture</span>
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-4">
              Traditional systems overwhelm race engineers with continuous data streams. COGNIRACE implements
              an <strong className="text-purple-300">intelligent filtering system</strong> that only surfaces recommendations when:
            </p>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                <span className="text-slate-300"><strong className="text-rose-400">1+ CRITICAL events</strong> are detected (low fuel, mechanical anomalies, pit window closing)</span>
              </li>
              <li className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                <span className="text-slate-300"><strong className="text-orange-400">2+ HIGH severity events</strong> occur simultaneously (fuel spike + high speed, tire stress + traffic)</span>
              </li>
              <li className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                <span className="text-slate-300">This reduces alert fatigue from <strong>100+ alerts/minute</strong> to <strong className="text-cyan-300">5-10 actionable recommendations per race</strong></span>
              </li>
            </ul>
          </div>

          <Separator />

          <div>
            <h3 className="font-semibold text-lg mb-4 text-purple-400">System Architecture Overview</h3>
            <div className="grid grid-cols-1 gap-4">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-yellow-500/20 flex items-center justify-center">
                  <Database className="w-5 h-5 text-yellow-400" />
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-yellow-400 mb-1">Data Layer</div>
                  <p className="text-sm text-muted-foreground">
                    F1 2017-2023 telemetry dataset with 1M+ laps across 20+ circuits. Features include speed, throttle, 
                    RPM, fuel consumption, tire degradation, weather conditions, and driver-specific characteristics.
                  </p>
                </div>
              </div>

              <ArrowRight className="w-5 h-5 text-cyan-400 mx-auto" />

              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                  <Cpu className="w-5 h-5 text-purple-400" />
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-purple-400 mb-1">ML Training Pipeline</div>
                  <p className="text-sm text-muted-foreground">
                    8 models trained independently with specialized architectures: Gradient Boosting for fuel prediction, 
                    Transformers for lap time forecasting, CNN-LSTM for tire degradation, GNN for traffic analysis, and more.
                    Hyperparameter tuning via grid search, 5-fold cross-validation, 80/20 train-test splits.
                  </p>
                </div>
              </div>

              <ArrowRight className="w-5 h-5 text-cyan-400 mx-auto" />

              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
                  <Cloud className="w-5 h-5 text-green-400" />
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-green-400 mb-1">Cloud Deployment</div>
                  <p className="text-sm text-muted-foreground">
                    Models exported as PyTorch (.pth) or Pickle (.pkl) files and stored in Google Cloud Storage 
                    (gs://cognirace-models). FastAPI backend loads models on startup for sub-100ms inference latency.
                  </p>
                </div>
              </div>

              <ArrowRight className="w-5 h-5 text-cyan-400 mx-auto" />

              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-cyan-500/20 flex items-center justify-center">
                  <Zap className="w-5 h-5 text-cyan-400" />
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-cyan-400 mb-1">Real-Time Inference</div>
                  <p className="text-sm text-muted-foreground">
                    Telemetry processed at 10Hz. Event detection pipeline identifies critical situations (low fuel, 
                    anomalies, pit windows). Strategy formatter converts ML predictions into F1-style race engineer 
                    recommendations with clear actions and calculations.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Ecosystem */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-purple-300 to-purple-400 bg-clip-text text-transparent">The 8-Model Ecosystem</CardTitle>
          <CardDescription className="text-slate-400">
            Specialized neural networks working in harmony for comprehensive race intelligence
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="fuel" className="w-full" suppressHydrationWarning>
            <TabsList className="grid grid-cols-4 w-full">
              <TabsTrigger value="fuel">Predictive</TabsTrigger>
              <TabsTrigger value="detection">Detection</TabsTrigger>
              <TabsTrigger value="optimization">Optimization</TabsTrigger>
              <TabsTrigger value="analysis">Analysis</TabsTrigger>
            </TabsList>

            <TabsContent value="fuel" className="space-y-4 mt-6">
              <div className="grid grid-cols-2 gap-4">
                <ModelCard
                  icon={<Fuel className="w-6 h-6" />}
                  title="Fuel Consumption"
                  color="yellow"
                  type="Gradient Boosting"
                  accuracy="96%"
                  latency="12ms"
                  description="Predicts lap-by-lap fuel usage with 0.008L/lap MAE. Accounts for speed, throttle, RPM, track characteristics."
                />
                <ModelCard
                  icon={<Zap className="w-6 h-6" />}
                  title="Lap Time Transformer"
                  color="purple"
                  type="Transformer (Encoder-Decoder)"
                  accuracy="94%"
                  latency="18ms"
                  description="Multi-lap time prediction using attention mechanism. Learns temporal dependencies and tire degradation effects."
                />
              </div>
            </TabsContent>

            <TabsContent value="detection" className="space-y-4 mt-6">
              <div className="grid grid-cols-2 gap-4">
                <ModelCard
                  icon={<AlertTriangle className="w-6 h-6" />}
                  title="Anomaly Detector"
                  color="red"
                  type="Isolation Forest"
                  accuracy="94%"
                  latency="15ms"
                  description="Unsupervised detection of mechanical failures, driver errors, and sensor malfunctions across 24D feature space."
                />
                <ModelCard
                  icon={<AlertTriangle className="w-6 h-6" />}
                  title="FCY Hazard Predictor"
                  color="orange"
                  type="Random Forest Classifier"
                  accuracy="89%"
                  latency="8ms"
                  description="Predicts Full Course Yellow probability within 5 laps based on weather, position, incidents, and lap number."
                />
              </div>
            </TabsContent>

            <TabsContent value="optimization" className="space-y-4 mt-6">
              <div className="grid grid-cols-2 gap-4">
                <ModelCard
                  icon={<Gauge className="w-6 h-6" />}
                  title="Tire Degradation"
                  color="red"
                  type="CNN-LSTM Hybrid"
                  accuracy="91%"
                  latency="22ms"
                  description="Predicts tire wear across 4 corners using 16D input including temperature, pressure, G-forces, and track conditions."
                />
                <ModelCard
                  icon={<TrendingUp className="w-6 h-6" />}
                  title="Pit Loss Model"
                  color="green"
                  type="XGBoost Regressor"
                  accuracy="93%"
                  latency="10ms"
                  description="Circuit-specific pit stop time loss prediction accounting for pit lane length, traffic, and tire compound changes."
                />
              </div>
            </TabsContent>

            <TabsContent value="analysis" className="space-y-4 mt-6">
              <div className="grid grid-cols-2 gap-4">
                <ModelCard
                  icon={<User className="w-6 h-6" />}
                  title="Driver Embedding"
                  color="blue"
                  type="Autoencoder Neural Network"
                  accuracy="88%"
                  latency="14ms"
                  description="32D latent representation learning driver-specific racing styles: braking patterns, throttle control, tire management."
                />
                <ModelCard
                  icon={<Network className="w-6 h-6" />}
                  title="Traffic GNN"
                  color="cyan"
                  type="Graph Convolutional Network"
                  accuracy="89%"
                  latency="25ms"
                  description="Models multi-car interactions for overtaking predictions, pit exit traffic forecasting, and blue flag scenarios."
                />
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Performance Benchmarks */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-cyan-300 to-cyan-400 bg-clip-text text-transparent">Performance Benchmarks</CardTitle>
          <CardDescription className="text-slate-400">
            Industry-leading inference speed and prediction accuracy
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Zap className="w-4 h-4 text-yellow-400" />
                Inference Latency (per model)
              </h3>
              <div className="space-y-3">
                <PerformanceBar label="FCY Hazard" value={8} max={30} color="bg-green-500" />
                <PerformanceBar label="Pit Loss" value={10} max={30} color="bg-green-500" />
                <PerformanceBar label="Fuel Consumption" value={12} max={30} color="bg-green-500" />
                <PerformanceBar label="Driver Embedding" value={14} max={30} color="bg-yellow-500" />
                <PerformanceBar label="Anomaly Detector" value={15} max={30} color="bg-yellow-500" />
                <PerformanceBar label="Lap Time Transformer" value={18} max={30} color="bg-yellow-500" />
                <PerformanceBar label="Tire Degradation" value={22} max={30} color="bg-orange-500" />
                <PerformanceBar label="Traffic GNN" value={25} max={30} color="bg-orange-500" />
              </div>
              <p className="text-xs text-muted-foreground mt-3">
                All models achieve sub-30ms inference, enabling real-time strategy at 10Hz telemetry rate
              </p>
            </div>

            <div>
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-cyan-400" />
                Model Accuracy (R² / F1-Score)
              </h3>
              <div className="space-y-3">
                <AccuracyBar label="Fuel Consumption" value={96} color="bg-green-500" />
                <AccuracyBar label="Lap Time Transformer" value={94} color="bg-green-500" />
                <AccuracyBar label="Anomaly Detector" value={94} color="bg-green-500" />
                <AccuracyBar label="Pit Loss Model" value={93} color="bg-green-500" />
                <AccuracyBar label="Tire Degradation" value={91} color="bg-green-500" />
                <AccuracyBar label="FCY Hazard" value={89} color="bg-yellow-500" />
                <AccuracyBar label="Traffic GNN" value={89} color="bg-yellow-500" />
                <AccuracyBar label="Driver Embedding" value={88} color="bg-yellow-500" />
              </div>
              <p className="text-xs text-muted-foreground mt-3">
                All models exceed 88% accuracy on held-out test sets from 2023 season
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Technology Stack */}
      <Card className="bg-slate-900/20 transition-colors backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-purple-300 to-purple-400 bg-clip-text text-transparent">Technology Stack</CardTitle>
          <CardDescription className="text-slate-400">
            Modern ML infrastructure built for production reliability
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-3">
              <h3 className="font-semibold text-cyan-400 flex items-center gap-2">
                <Cpu className="w-4 h-4" />
                ML & Training
              </h3>
              <div className="space-y-2 text-sm">
                <TechBadge name="PyTorch 2.0" description="Neural network training & inference" />
                <TechBadge name="Scikit-learn 1.3" description="Classical ML algorithms (GB, RF, IF)" />
                <TechBadge name="XGBoost" description="Gradient boosting with GPU acceleration" />
                <TechBadge name="PyTorch Geometric" description="Graph neural network framework" />
                <TechBadge name="Pandas & NumPy" description="Data manipulation & numerical computing" />
              </div>
            </div>

            <div className="space-y-3">
              <h3 className="font-semibold text-green-400 flex items-center gap-2">
                <Cloud className="w-4 h-4" />
                Backend & Deployment
              </h3>
              <div className="space-y-2 text-sm">
                <TechBadge name="FastAPI + Uvicorn" description="High-performance async API server" />
                <TechBadge name="Pydantic V2" description="Data validation & serialization" />
                <TechBadge name="Google Cloud Storage" description="Model storage & versioning" />
                <TechBadge name="Python 3.11" description="Latest language features & performance" />
                <TechBadge name="Docker" description="Containerized deployment (planned)" />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Helper Components
function ModelCard({ icon, title, color, type, accuracy, latency, description }: any) {
  const colors: any = {
    yellow: 'bg-yellow-500/5',
    purple: 'bg-purple-500/5',
    red: 'bg-red-500/5',
    orange: 'bg-orange-500/5',
    green: 'bg-green-500/5',
    blue: 'bg-blue-500/5',
    cyan: 'bg-cyan-500/5',
  };

  return (
    <div className={`${colors[color]} rounded-lg p-4 space-y-3`}>
      <div className="flex items-center gap-3">
        <div className={`text-${color}-400`}>{icon}</div>
        <div className="flex-1">
          <div className="font-semibold">{title}</div>
          <div className="text-xs text-muted-foreground">{type}</div>
        </div>
      </div>
      <p className="text-sm text-muted-foreground">{description}</p>
      <div className="flex gap-4 text-xs">
        <div>
          <span className="text-muted-foreground">Accuracy:</span>{' '}
          <span className="font-semibold text-green-400">{accuracy}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Latency:</span>{' '}
          <span className="font-semibold text-cyan-400">{latency}</span>
        </div>
      </div>
    </div>
  );
}

function PerformanceBar({ label, value, max, color }: any) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono font-semibold">{value}ms</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-500`}
          style={{ width: `${(value / max) * 100}%` }}
        />
      </div>
    </div>
  );
}

function AccuracyBar({ label, value, color }: any) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono font-semibold">{value}%</span>
      </div>
      <Progress value={value} className="h-2" />
    </div>
  );
}

function TechBadge({ name, description }: any) {
  return (
    <div className="bg-gray-900/50 rounded-lg p-2">
      <div className="font-semibold text-xs">{name}</div>
      <div className="text-xs text-muted-foreground">{description}</div>
    </div>
  );
}

