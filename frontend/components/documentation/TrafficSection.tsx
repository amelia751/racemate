'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Network, Activity, Zap, Target, Code, CheckCircle2, GitBranch
} from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Legend, BarChart, Bar } from 'recharts';
import { DocumentationHeader, CodeBlock, DocumentationTabs } from './shared';

// Sample data
const overtakingProbabilityData = [
  { deltaSpeed: -30, probability: 0.05 },
  { deltaSpeed: -20, probability: 0.12 },
  { deltaSpeed: -10, probability: 0.25 },
  { deltaSpeed: 0, probability: 0.50 },
  { deltaSpeed: 10, probability: 0.72 },
  { deltaSpeed: 20, probability: 0.88 },
  { deltaSpeed: 30, probability: 0.95 },
];

const trackPositionData = [
  { lap: 1, position: 8, predicted: 8 },
  { lap: 5, position: 7, predicted: 7 },
  { lap: 10, position: 6, predicted: 6.2 },
  { lap: 15, position: 5, predicted: 5.1 },
  { lap: 20, position: 5, predicted: 4.8 },
  { lap: 25, position: 4, predicted: 4.2 },
  { lap: 30, position: 4, predicted: 3.9 },
];

const drsImpactData = [
  { scenario: 'No DRS', overtakeProb: 0.28 },
  { scenario: 'DRS Available', overtakeProb: 0.72 },
  { scenario: 'DRS + Tire Δ', overtakeProb: 0.91 },
  { scenario: 'DRS + Tire Δ + Speed Δ', overtakeProb: 0.97 },
];

export default function TrafficSection() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <DocumentationHeader
        icon={Network}
        title="TRAFFIC GNN"
        subtitle="Graph Neural Network for Overtaking Probability & Track Position Prediction"
        color="cyan"
        metrics={[
          { label: 'Architecture', value: 'GNN (GCN)' },
          { label: 'Accuracy', value: '86%' },
          { label: 'Graph Nodes', value: '20 drivers' },
          { label: 'Inference', value: '18ms' }
        ]}
      />

      {/* Model Overview */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-cyan-400">Model Overview</CardTitle>
          <CardDescription>
            Modeling driver interactions as a dynamic graph
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-muted-foreground leading-relaxed">
            Racing isn't 20 independent cars—it's a <strong className="text-cyan-400">dynamic graph</strong> where each driver (node) 
            interacts with nearby cars (edges). Traffic GNN predicts: 
            <strong className="text-blue-400"> overtaking probability</strong>, 
            <strong className="text-indigo-400"> post-pit track position</strong>, and 
            <strong className="text-purple-400"> optimal passing opportunities</strong> using Graph Convolutional Networks.
          </p>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gradient-to-br from-cyan-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Network className="w-5 h-5 text-cyan-400" />
                <span className="font-semibold">Graph Structure</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Nodes: 20 drivers with features (speed, position, tire age). Edges: Proximity relationships (within 5s gap).
              </p>
            </div>

            <div className="bg-gradient-to-br from-blue-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <GitBranch className="w-5 h-5 text-blue-400" />
                <span className="font-semibold">Message Passing</span>
              </div>
              <p className="text-sm text-muted-foreground">
                GCN aggregates neighbor information: Speed differential, tire advantage, DRS availability propagate through graph.
              </p>
            </div>

            <div className="bg-gradient-to-br from-indigo-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-5 h-5 text-indigo-400" />
                <span className="font-semibold">Overtaking Probability</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Model outputs probability (0-1) for each driver pair. DRS + 10 km/h advantage = 72% overtake probability.
              </p>
            </div>

            <div className="bg-gradient-to-br from-purple-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-5 h-5 text-purple-400" />
                <span className="font-semibold">Position Forecasting</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Predict track position 5 laps ahead. Critical for pit stop timing: exit P8, but overtake to P5 within 3 laps.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Architecture */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-cyan-400">GNN Architecture</CardTitle>
          <CardDescription>
            Graph Convolutional Network with PyTorch Geometric
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <CodeBlock
            title="PyTorch Geometric GCN Implementation"
            language="python"
            code={`import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TrafficGNN(torch.nn.Module):
    def __init__(self, num_features=10, hidden_dim=64, output_dim=1):
        super().__init__()
        
        # Graph Convolutional Layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        self.dropout = torch.nn.Dropout(0.3)
    
    def forward(self, x, edge_index):
        # x: Node features [20, 10] (20 drivers, 10 features)
        # edge_index: Graph edges [2, num_edges]
        
        # Layer 1: Aggregate neighbor information
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2: Deeper message passing
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3: Output overtaking probability
        x = self.conv3(x, edge_index)
        x = torch.sigmoid(x)  # Probability [0, 1]
        
        return x

# Graph construction
edge_index = construct_proximity_graph(driver_gaps)  # Connect drivers within 5s
node_features = torch.tensor([speed, position, tire_age, ...])  # [20, 10]

# Prediction
model = TrafficGNN(num_features=10, hidden_dim=64, output_dim=1)
overtake_prob = model(node_features, edge_index)  # [20, 1]`}
          />

          <Separator />

          <div>
            <h3 className="font-semibold text-lg mb-4">Node Features (10 per driver)</h3>
            <div className="space-y-2">
              <FeatureRow label="Speed" description="Current speed (km/h)" />
              <FeatureRow label="Track Position" description="Current race position (1-20)" />
              <FeatureRow label="Gap to Leader" description="Time gap to P1 (seconds)" />
              <FeatureRow label="Tire Age" description="Laps on current compound" />
              <FeatureRow label="Tire Compound" description="Soft/Medium/Hard (one-hot)" />
              <FeatureRow label="DRS Available" description="Binary flag" />
              <FeatureRow label="Fuel Load" description="Remaining fuel (kg)" />
              <FeatureRow label="Driver Aggression" description="From Driver Embedding model" />
              <FeatureRow label="Recent Pace" description="Average lap time (last 3 laps)" />
              <FeatureRow label="Pit Stop Count" description="Number of pit stops completed" />
            </div>
          </div>

          <div className="bg-cyan-500/10 rounded-lg p-4 mt-6">
            <h4 className="font-semibold text-cyan-400 mb-2">Training Details</h4>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="bg-black/40 rounded p-2">
                <span className="text-muted-foreground">Dataset:</span>
                <span className="font-mono ml-2">45,000 race snapshots</span>
              </div>
              <div className="bg-black/40 rounded p-2">
                <span className="text-muted-foreground">Graph Updates:</span>
                <span className="font-mono ml-2">Every lap (dynamic edges)</span>
              </div>
              <div className="bg-black/40 rounded p-2">
                <span className="text-muted-foreground">Loss Function:</span>
                <span className="font-mono ml-2">Binary Cross-Entropy</span>
              </div>
              <div className="bg-black/40 rounded p-2">
                <span className="text-muted-foreground">Accuracy:</span>
                <span className="font-mono ml-2 text-green-400">86%</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Visualizations */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-cyan-400">Performance Analysis</CardTitle>
          <CardDescription>
            Overtaking predictions and position forecasting
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <Tabs defaultValue="overtake" className="w-full" suppressHydrationWarning>
            <TabsList className="grid grid-cols-3 w-full">
              <TabsTrigger value="overtake">Overtaking Probability</TabsTrigger>
              <TabsTrigger value="position">Position Forecast</TabsTrigger>
              <TabsTrigger value="drs">DRS Impact</TabsTrigger>
            </TabsList>

            <TabsContent value="overtake" className="space-y-4 mt-6">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={overtakingProbabilityData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="deltaSpeed" stroke="#9ca3af" label={{ value: 'Speed Differential (km/h)', position: 'insideBottom', offset: -5 }} />
                    <YAxis stroke="#9ca3af" domain={[0, 1]} label={{ value: 'Overtaking Probability', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Line type="monotone" dataKey="probability" stroke="#06b6d4" strokeWidth={3} name="Probability" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="text-sm text-muted-foreground">
                Speed differential is the strongest overtaking predictor. +20 km/h advantage = 88% probability. 
                Graph structure accounts for DRS, tire age differential, and driver aggression.
              </p>
            </TabsContent>

            <TabsContent value="position" className="space-y-4 mt-6">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trackPositionData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="lap" stroke="#9ca3af" label={{ value: 'Lap Number', position: 'insideBottom', offset: -5 }} />
                    <YAxis stroke="#9ca3af" reversed domain={[1, 8]} label={{ value: 'Track Position', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="position" stroke="#06b6d4" strokeWidth={3} name="Actual Position" />
                    <Line type="monotone" dataKey="predicted" stroke="#22c55e" strokeWidth={3} strokeDasharray="5 5" name="Predicted Position" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="text-sm text-muted-foreground">
                GNN accurately predicts track position progression. Starting P8, model forecasts climb to P4 by lap 30 
                based on tire strategy and overtaking opportunities.
              </p>
            </TabsContent>

            <TabsContent value="drs" className="space-y-4 mt-6">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={drsImpactData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="scenario" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" domain={[0, 1]} label={{ value: 'Overtaking Probability', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Bar dataKey="overtakeProb" fill="#06b6d4" name="Overtaking Probability" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-sm text-muted-foreground">
                DRS dramatically increases overtaking probability from 28% to 72%. Combined with tire and speed advantages, 
                probability reaches 97%. Model uses these factors to identify optimal passing opportunities.
              </p>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Real-World Integration */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-cyan-400">Real-World Integration</CardTitle>
          <CardDescription>
            Strategic overtaking and pit stop timing
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="bg-gradient-to-br from-cyan-900/30 to-black rounded-lg p-6">
            <h3 className="font-semibold text-cyan-400 mb-4 flex items-center gap-2">
              <Network className="w-5 h-5" />
              Traffic-Aware Pit Stop Recommendation
            </h3>
            <CodeBlock
              title="Traffic-Aware Pit Stop Recommendation"
              language="text"
              code={`🔵 TRAFFIC GNN ANALYSIS - Lap 18, P5

Current Track Situation:
  P4: Sainz (0.8s ahead, 15-lap old mediums)
  P5: YOU (fresh tires after pit stop)
  P6: Perez (2.3s behind, 18-lap old mediums)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 GNN PREDICTIONS (Next 5 Laps):

Overtaking P4 (Sainz):
  • Speed differential: +12 km/h (fresh tires)
  • DRS available: YES (within 1.0s by lap 20)
  • Tire age gap: 15 laps (major advantage)
  • Driver aggression: High (embedding score)
  → OVERTAKING PROBABILITY: 78% (Lap 20-22)

Defending P5 (Perez):
  • Speed differential: -8 km/h (older tires)
  • DRS available for Perez: YES (closing gap)
  • Tire age gap: 18 laps (your advantage declining)
  → DEFEND PROBABILITY: 68% (can hold position)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 POSITION FORECAST:
  Lap 20: P5 → P4 (overtake Sainz - 78% prob)
  Lap 23: P4 → P4 (hold position - 68% prob)
  Lap 25: P4 → P3 (Leclerc pit stop)
  
  FINAL PREDICTION: P3 by Lap 25 (confidence: 72%)

📍 RECOMMENDATION:
   ✅ Aggressive strategy - push to overtake Sainz immediately
   ✅ Fresh tire window: Next 8 laps
   ✅ Expected outcome: P3 finish (+2 positions)`}
            />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-green-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-5 h-5 text-green-400" />
                <span className="font-semibold text-sm">Dynamic Graphs</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Graph updates every lap as drivers pit, overtake, or experience issues
              </p>
            </div>
            <div className="bg-blue-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Network className="w-5 h-5 text-blue-400" />
                <span className="font-semibold text-sm">Context-Aware</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Considers full field state, not just immediate rivals, for comprehensive analysis
              </p>
            </div>
            <div className="bg-purple-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-5 h-5 text-purple-400" />
                <span className="font-semibold text-sm">Integrated</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Works with Pit Loss Model and Driver Embedding for comprehensive strategy optimization
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Helper Components

function FeatureRow({ label, description }: any) {
  return (
    <div className="flex justify-between items-center py-2 px-3 bg-gray-900/30 rounded text-sm">
      <span className="font-semibold text-cyan-400">{label}</span>
      <span className="text-muted-foreground text-xs">{description}</span>
    </div>
  );
}

