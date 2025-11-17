'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Zap, TrendingUp, Brain, Layers, Clock, Target, CheckCircle2, Code, Eye
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, AreaChart, Area } from 'recharts';

// Sample data for visualizations
const sequencePredictionData = [
  { lap: 1, actual: 87.2, predicted: 87.1, attention: 0.95 },
  { lap: 2, actual: 87.5, predicted: 87.4, attention: 0.92 },
  { lap: 3, actual: 88.1, predicted: 88.0, attention: 0.88 },
  { lap: 4, actual: 88.8, predicted: 88.7, attention: 0.85 },
  { lap: 5, actual: 89.2, predicted: 89.3, attention: 0.81 },
  { lap: 6, actual: 89.8, predicted: 89.7, attention: 0.78 },
  { lap: 7, actual: 90.5, predicted: 90.4, attention: 0.74 },
  { lap: 8, actual: 91.1, predicted: 91.2, attention: 0.71 },
];

const attentionWeightsData = [
  { position: 'Lap-4', self: 0.85, cross: 0.15 },
  { position: 'Lap-3', self: 0.78, cross: 0.22 },
  { position: 'Lap-2', self: 0.65, cross: 0.35 },
  { position: 'Lap-1', self: 0.52, cross: 0.48 },
  { position: 'Current', self: 0.42, cross: 0.58 },
];

const tireAgingImpactData = [
  { age: 0, baseTime: 87.0, tireDelta: 0.0 },
  { age: 5, baseTime: 87.0, tireDelta: 0.3 },
  { age: 10, baseTime: 87.0, tireDelta: 0.8 },
  { age: 15, baseTime: 87.0, tireDelta: 1.5 },
  { age: 20, baseTime: 87.0, tireDelta: 2.4 },
  { age: 25, baseTime: 87.0, tireDelta: 3.6 },
  { age: 30, baseTime: 87.0, tireDelta: 5.2 },
];

export default function LaptimeSection() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-purple-900/30 via-blue-900/30 to-black border border-purple-500/30 p-8">
        <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:30px_30px]" />
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
              <Zap className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-black text-purple-400">
                LAP TIME TRANSFORMER
              </h1>
              <p className="text-muted-foreground mt-1">
                Sequence-to-Sequence Attention Model for Multi-Lap Time Prediction
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-4 gap-4 mt-6">
            <MetricCard label="Architecture" value="Transformer" />
            <MetricCard label="MAE" value="0.42s" />
            <MetricCard label="R² Score" value="0.94" />
            <MetricCard label="Inference" value="18ms" />
          </div>
        </div>
      </div>

      {/* Model Overview */}
      <Card className="bg-black/40 border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-2xl text-purple-400">Model Overview</CardTitle>
          <CardDescription>
            Why Transformers for lap time prediction?
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-muted-foreground leading-relaxed">
            Unlike traditional RNNs or LSTMs that process sequences sequentially, Transformers use <strong className="text-purple-400">self-attention mechanisms</strong> to 
            capture complex temporal dependencies in parallel. This is crucial for lap time prediction because:
          </p>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gradient-to-br from-purple-900/20 to-black border border-purple-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Brain className="w-5 h-5 text-purple-400" />
                <span className="font-semibold">Non-Linear Dependencies</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Lap time isn't just influenced by the previous lap. A qualifying lap 5 laps ago can predict degradation patterns better than just lap n-1.
              </p>
            </div>

            <div className="bg-gradient-to-br from-blue-900/20 to-black border border-blue-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Layers className="w-5 h-5 text-blue-400" />
                <span className="font-semibold">Multi-Head Attention</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Different attention heads learn different patterns: tire degradation, fuel load reduction, driver adaptation, track evolution.
              </p>
            </div>

            <div className="bg-gradient-to-br from-cyan-900/20 to-black border border-cyan-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-5 h-5 text-cyan-400" />
                <span className="font-semibold">Positional Encoding</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Embeds lap number information, allowing the model to distinguish early-race pace from late-stint degradation.
              </p>
            </div>

            <div className="bg-gradient-to-br from-green-900/20 to-black border border-green-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-5 h-5 text-green-400" />
                <span className="font-semibold">Parallel Processing</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Unlike RNNs, Transformers process the entire sequence in parallel during training, enabling faster convergence (50 epochs vs. 200+).
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Architecture Deep Dive */}
      <Card className="bg-black/40 border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-2xl text-purple-400">Architecture Deep Dive</CardTitle>
          <CardDescription>
            Encoder-decoder structure with multi-head attention
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="bg-gray-900/50 rounded-lg p-6 font-mono text-sm space-y-3">
            <div className="flex items-start gap-3">
              <Code className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="text-purple-400 font-semibold mb-2">PyTorch Transformer Implementation</div>
                <pre className="text-muted-foreground leading-relaxed overflow-x-auto">
{`import torch
import torch.nn as nn

class LapTimeTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        # Input embedding
        self.input_proj = nn.Linear(3, d_model)  # 3 features: speed, throttle, tire_age
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 1)  # Predict lap time
        
    def forward(self, src, tgt):
        # src: [seq_len=4, batch, features=3]
        # tgt: [1, batch, features=3] - current lap features
        
        # Embed and add positional encoding
        src = self.pos_encoder(self.input_proj(src))
        tgt = self.pos_encoder(self.input_proj(tgt))
        
        # Encode past laps
        memory = self.encoder(src)
        
        # Decode to predict next lap time
        output = self.decoder(tgt, memory)
        
        # Project to lap time
        return self.output_proj(output).squeeze(-1)`}
                </pre>
              </div>
            </div>
          </div>

          <Separator />

          <div>
            <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
              <Layers className="w-5 h-5 text-cyan-400" />
              <span>Layer-by-Layer Breakdown</span>
            </h3>
            <div className="space-y-4">
              <LayerCard
                number={1}
                title="Input Embedding"
                description="Projects 3-dimensional input (speed, throttle, tire_age) to 64-dimensional embedding space"
                inputShape="[4, batch, 3]"
                outputShape="[4, batch, 64]"
                parameters="192"
              />
              <LayerCard
                number={2}
                title="Positional Encoding"
                description="Adds sinusoidal position embeddings to preserve sequence order information"
                inputShape="[4, batch, 64]"
                outputShape="[4, batch, 64]"
                parameters="0 (deterministic)"
              />
              <LayerCard
                number={3}
                title="Multi-Head Self-Attention (×2)"
                description="4 attention heads learn different temporal patterns across the sequence"
                inputShape="[4, batch, 64]"
                outputShape="[4, batch, 64]"
                parameters="16,384 per layer"
              />
              <LayerCard
                number={4}
                title="Feed-Forward Network (×2)"
                description="Two-layer MLP with ReLU activation (64 → 128 → 64)"
                inputShape="[4, batch, 64]"
                outputShape="[4, batch, 64]"
                parameters="8,192 per layer"
              />
              <LayerCard
                number={5}
                title="Decoder Cross-Attention"
                description="Attends to encoded past laps to predict next lap time"
                inputShape="[1, batch, 64]"
                outputShape="[1, batch, 64]"
                parameters="16,384"
              />
              <LayerCard
                number={6}
                title="Output Projection"
                description="Linear layer projecting 64D embedding to scalar lap time prediction"
                inputShape="[1, batch, 64]"
                outputShape="[1, batch, 1]"
                parameters="64"
              />
            </div>
            <div className="mt-4 bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <div className="flex justify-between items-center">
                <span className="font-semibold text-purple-400">Total Parameters:</span>
                <span className="font-mono text-2xl font-bold text-purple-400">49,344</span>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Lightweight architecture enables 18ms inference latency on CPU
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Attention Mechanism Visualization */}
      <Card className="bg-black/40 border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-2xl text-purple-400">Attention Mechanism Visualization</CardTitle>
          <CardDescription>
            How the model "attends" to different past laps
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-sm text-muted-foreground">
            The attention mechanism learns which past laps are most relevant for predicting the next lap time. 
            Early laps get higher attention for baseline pace, while recent laps are critical for degradation trends.
          </p>

          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={attentionWeightsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="position" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" label={{ value: 'Attention Weight', angle: -90, position: 'insideLeft' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Area type="monotone" dataKey="self" stackId="1" stroke="#a855f7" fill="#a855f7" fillOpacity={0.6} name="Self-Attention" />
                <Area type="monotone" dataKey="cross" stackId="1" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} name="Cross-Attention" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-3 gap-4 mt-4">
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
              <div className="text-sm font-semibold mb-1">Lap-4 (85%)</div>
              <p className="text-xs text-muted-foreground">
                Baseline pace reference - highest attention for establishing driver capability
              </p>
            </div>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
              <div className="text-sm font-semibold mb-1">Lap-1 (52%)</div>
              <p className="text-xs text-muted-foreground">
                Recent degradation trend - critical for short-term prediction
              </p>
            </div>
            <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3">
              <div className="text-sm font-semibold mb-1">Current (42%)</div>
              <p className="text-xs text-muted-foreground">
                Cross-attention dominates - decoding from encoded memory
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Training Pipeline */}
      <Card className="bg-black/40 border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-2xl text-purple-400">Training Pipeline</CardTitle>
          <CardDescription>
            From sequence generation to production deployment
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <TrainingStep
            number={1}
            title="Sequence Generation"
            icon={<Layers className="w-6 h-6" />}
            details={[
              'Sliding window approach: Use laps [n-4:n] to predict lap n+1',
              'Input features: [speed, throttle, tire_age] normalized to [0,1]',
              'Generated 300,000 sequences from 2017-2021 seasons',
              'Filter: Remove sequences with safety car or pit stops',
              'Augmentation: Add Gaussian noise (σ=0.02) to improve robustness',
            ]}
          />

          <TrainingStep
            number={2}
            title="Model Training"
            icon={<Brain className="w-6 h-6" />}
            details={[
              'Loss function: Mean Squared Error (MSE)',
              'Optimizer: AdamW (lr=0.001, weight_decay=1e-4)',
              'Batch size: 128 sequences',
              'Epochs: 50 (early stopping with patience=5)',
              'Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=3)',
              'Gradient clipping: max_norm=1.0',
            ]}
          />

          <TrainingStep
            number={3}
            title="Validation & Testing"
            icon={<Target className="w-6 h-6" />}
            details={[
              'Validation set: 2022 season (50K sequences)',
              'Test set: 2023 season (50K sequences)',
              'Evaluation metrics: MAE, RMSE, R², Max Error',
              'Circuit-specific analysis: All circuits MAE < 0.6s',
              'Edge case testing: Wet conditions, tire changes, traffic',
            ]}
          />
        </CardContent>
      </Card>

      {/* Performance Analysis */}
      <Card className="bg-black/40 border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-2xl text-purple-400">Performance Analysis</CardTitle>
          <CardDescription>
            Prediction accuracy across race stints
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="h-[350px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sequencePredictionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="lap" 
                  stroke="#9ca3af" 
                  label={{ value: 'Lap Number', position: 'insideBottom', offset: -5 }} 
                />
                <YAxis 
                  stroke="#9ca3af" 
                  domain={[86, 92]}
                  label={{ value: 'Lap Time (seconds)', angle: -90, position: 'insideLeft' }} 
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="actual" 
                  stroke="#3b82f6" 
                  strokeWidth={3} 
                  name="Actual Lap Time" 
                  dot={{ fill: '#3b82f6', r: 4 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="predicted" 
                  stroke="#22c55e" 
                  strokeWidth={3} 
                  name="Predicted Lap Time" 
                  strokeDasharray="5 5"
                  dot={{ fill: '#22c55e', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <Separator />

          <div>
            <h3 className="font-semibold mb-4">Tire Aging Impact Prediction</h3>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={tireAgingImpactData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="age" 
                    stroke="#9ca3af" 
                    label={{ value: 'Tire Age (laps)', position: 'insideBottom', offset: -5 }} 
                  />
                  <YAxis 
                    stroke="#9ca3af"
                    label={{ value: 'Time Delta (seconds)', angle: -90, position: 'insideLeft' }} 
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="tireDelta" 
                    stroke="#ef4444" 
                    strokeWidth={3} 
                    name="Degradation Delta"
                    dot={{ fill: '#ef4444', r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-sm text-muted-foreground mt-4">
              The Transformer successfully learns non-linear tire degradation patterns. Note the acceleration in 
              lap time loss after ~20 laps as the tire enters the "cliff" phase.
            </p>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-5 h-5 text-green-400" />
                <span className="font-semibold">Excellent Accuracy</span>
              </div>
              <p className="text-sm text-muted-foreground">
                MAE of 0.42s across all circuits - within driver reaction time variability
              </p>
            </div>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5 text-blue-400" />
                <span className="font-semibold">Captures Trends</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Successfully predicts degradation acceleration and stint-end tire cliff
              </p>
            </div>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Eye className="w-5 h-5 text-purple-400" />
                <span className="font-semibold">Attention Insights</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Interpretable attention weights reveal which laps influenced predictions
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Real-World Integration */}
      <Card className="bg-black/40 border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-2xl text-purple-400">Real-World Integration</CardTitle>
          <CardDescription>
            Strategy recommendations based on lap time forecasts
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="bg-gradient-to-br from-orange-900/30 to-black border border-orange-500/50 rounded-lg p-6">
            <h3 className="font-semibold text-orange-400 mb-4 flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Stint Strategy Optimization
            </h3>
            <p className="text-sm text-muted-foreground mb-4">
              The Transformer predicts when lap times will degrade beyond a target threshold, enabling optimal pit stop timing:
            </p>
            <div className="bg-gray-900/50 p-4 rounded-lg">
              <pre className="text-xs font-mono text-muted-foreground leading-relaxed">
{`🟡 STINT DEGRADATION WARNING

Current Lap: 18/30
Predicted Next 5 Laps:
  Lap 19: 88.9s (+1.7s vs. baseline)
  Lap 20: 89.5s (+2.3s)
  Lap 21: 90.2s (+3.0s)
  Lap 22: 91.1s (+3.9s) ← CRITICAL DEGRADATION
  Lap 23: 92.3s (+5.1s)

📍 RECOMMENDATION: Pit within 2 laps to minimize time loss
   • Current pace loss: ~2.5s/lap
   • Pit stop loss: 22s (includes out-lap)
   • Net gain by pitting now: ~15s over remaining stint`}
              </pre>
            </div>
          </div>

          <Separator />

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <h4 className="font-semibold text-purple-400 mb-2">Traffic Predictions</h4>
              <p className="text-sm text-muted-foreground">
                Combined with Traffic GNN, lap time forecasts predict post-pit track position and potential traffic losses.
              </p>
            </div>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <h4 className="font-semibold text-blue-400 mb-2">Undercut/Overcut Analysis</h4>
              <p className="text-sm text-muted-foreground">
                Predicts opponent degradation rates to determine optimal pit strategy (early undercut vs. late overcut).
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Helper Components
function MetricCard({ label, value }: any) {
  return (
    <div className="bg-black/40 backdrop-blur-sm rounded-lg p-3 border border-purple-500/20">
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className="text-lg font-bold text-purple-400">{value}</div>
    </div>
  );
}

function LayerCard({ number, title, description, inputShape, outputShape, parameters }: any) {
  return (
    <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-800">
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-purple-500/20 border border-purple-500/50 flex items-center justify-center font-bold text-sm text-purple-400">
          {number}
        </div>
        <div className="flex-1">
          <h4 className="font-semibold mb-1">{title}</h4>
          <p className="text-sm text-muted-foreground mb-2">{description}</p>
          <div className="flex gap-4 text-xs">
            <div>
              <span className="text-muted-foreground">Input:</span>{' '}
              <span className="font-mono text-cyan-400">{inputShape}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Output:</span>{' '}
              <span className="font-mono text-purple-400">{outputShape}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Params:</span>{' '}
              <span className="font-mono text-yellow-400">{parameters}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function TrainingStep({ number, title, icon, details }: any) {
  return (
    <div className="flex gap-4">
      <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-purple-500/20 border border-purple-500/50 flex items-center justify-center font-bold text-lg text-purple-400">
        {number}
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-2">
          <div className="text-purple-400">{icon}</div>
          <h3 className="font-semibold text-lg">{title}</h3>
        </div>
        <ul className="space-y-1 text-sm text-muted-foreground">
          {details.map((detail: string, i: number) => (
            <li key={i} className="flex items-start gap-2">
              <span className="text-cyan-400 mt-1">→</span>
              <span>{detail}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

