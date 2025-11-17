'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  User, Activity, Zap, Target, Code, CheckCircle2, Users
} from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { DocumentationHeader, CodeBlock, DocumentationTabs } from './shared';

// Sample data
const driverStyleData = [
  { characteristic: 'Aggression', verstappen: 95, hamilton: 75, leclerc: 85, norris: 70, sainz: 72 },
  { characteristic: 'Consistency', verstappen: 92, hamilton: 98, leclerc: 78, norris: 88, sainz: 90 },
  { characteristic: 'Tire Management', verstappen: 85, hamilton: 95, leclerc: 75, norris: 92, sainz: 88 },
  { characteristic: 'Overtaking', verstappen: 98, hamilton: 90, leclerc: 88, norris: 75, sainz: 78 },
  { characteristic: 'Qualifying', verstappen: 97, hamilton: 92, leclerc: 95, norris: 88, sainz: 85 },
];

const embeddingSpace2D = [
  { x: 0.85, y: 0.92, driver: 'Verstappen', cluster: 'Aggressive' },
  { x: 0.78, y: 0.95, driver: 'Hamilton', cluster: 'Consistent' },
  { x: 0.82, y: 0.88, driver: 'Leclerc', cluster: 'Aggressive' },
  { x: 0.72, y: 0.90, driver: 'Norris', cluster: 'Consistent' },
  { x: 0.70, y: 0.92, driver: 'Sainz', cluster: 'Consistent' },
  { x: 0.88, y: 0.85, driver: 'Alonso', cluster: 'Aggressive' },
  { x: 0.75, y: 0.93, driver: 'Russell', cluster: 'Consistent' },
  { x: 0.80, y: 0.87, driver: 'Perez', cluster: 'Balanced' },
];

const featureImportanceData = [
  { feature: 'Brake Point Variance', importance: 28 },
  { feature: 'Throttle Aggression', importance: 24 },
  { feature: 'Cornering Speed', importance: 18 },
  { feature: 'Lap-to-Lap Consistency', importance: 15 },
  { feature: 'Tire Wear Rate', importance: 10 },
  { feature: 'Fuel Efficiency', importance: 5 },
];

const CLUSTER_COLORS: Record<string, string> = {
  'Aggressive': '#ef4444',
  'Consistent': '#22c55e',
  'Balanced': '#3b82f6',
};

export default function DriverSection() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <DocumentationHeader
        icon={User}
        title="DRIVER EMBEDDING MODEL"
        subtitle="Autoencoder Neural Network for Driver Style Characterization"
        color="purple"
        metrics={[
          { label: 'Architecture', value: 'Autoencoder' },
          { label: 'Embedding Dim', value: '32D' },
          { label: 'Clusters', value: '3 Styles' },
          { label: 'Inference', value: '12ms' }
        ]}
      />

      {/* Model Overview */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-purple-400">Model Overview</CardTitle>
          <CardDescription>
            Learning compressed representations of driver behavior
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-muted-foreground leading-relaxed">
            Driver Embedding converts high-dimensional telemetry patterns (throttle application, brake points, cornering lines) into 
            a <strong className="text-purple-400">32-dimensional vector space</strong>. This enables: 
            <strong className="text-indigo-400"> style clustering</strong> (aggressive vs. consistent), 
            <strong className="text-pink-400"> teammate comparison</strong>, and 
            <strong className="text-cyan-400"> personalized strategy recommendations</strong>.
          </p>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-gradient-to-br from-purple-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-5 h-5 text-purple-400" />
                <span className="font-semibold">Autoencoder Architecture</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Encoder: 128D → 64D → 32D. Decoder: 32D → 64D → 128D. Bottleneck layer learns compressed driver "signature".
              </p>
            </div>

            <div className="bg-gradient-to-br from-indigo-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Users className="w-5 h-5 text-indigo-400" />
                <span className="font-semibold">Style Clustering</span>
              </div>
              <p className="text-sm text-muted-foreground">
                K-means (k=3) on embeddings: Aggressive (Verstappen, Leclerc), Consistent (Hamilton, Norris), Balanced (Perez).
              </p>
            </div>

            <div className="bg-gradient-to-br from-pink-900/20 to-black rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-5 h-5 text-pink-400" />
                <span className="font-semibold">Real-Time Adaptation</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Strategy recommendations adapt based on driver embedding: aggressive drivers get undercut timing, consistent drivers get overcut.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Architecture */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-purple-400">Autoencoder Architecture</CardTitle>
          <CardDescription>
            Dimensionality reduction for driver characterization
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <CodeBlock
            title="PyTorch Autoencoder Implementation"
            language="python"
            code={`import torch
import torch.nn as nn

class DriverAutoencoder(nn.Module):
    def __init__(self, input_dim=128, embedding_dim=32):
        super().__init__()
        
        # Encoder: Compress high-dimensional telemetry
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim),  # Bottleneck: 32D embedding
            nn.ReLU()
        )
        
        # Decoder: Reconstruct input from embedding
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # Normalized telemetry [0, 1]
        )
    
    def forward(self, x):
        embedding = self.encoder(x)  # 32D driver signature
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

# Training: Minimize reconstruction loss
model = DriverAutoencoder(input_dim=128, embedding_dim=32)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Extract embeddings for clustering
embeddings = model.encoder(telemetry_data)`}
          />

          <Separator />

          <div>
            <h3 className="font-semibold text-lg mb-4">Input Features (128 Dimensions)</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2 text-sm">
                <h4 className="font-semibold text-purple-400">Throttle Characteristics (32)</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Application rate (gentle vs. aggressive)</li>
                  <li>• Lift-off behavior (early vs. late)</li>
                  <li>• On-throttle stability (smooth vs. jerky)</li>
                  <li>• Exit acceleration patterns</li>
                </ul>
              </div>
              <div className="space-y-2 text-sm">
                <h4 className="font-semibold text-indigo-400">Brake Characteristics (32)</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Brake point consistency (variance across laps)</li>
                  <li>• Initial pressure (sharp vs. gradual)</li>
                  <li>• Trail braking extent (deep vs. shallow)</li>
                  <li>• Release timing (early vs. late)</li>
                </ul>
              </div>
              <div className="space-y-2 text-sm">
                <h4 className="font-semibold text-pink-400">Cornering Characteristics (32)</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Entry speed variance</li>
                  <li>• Apex precision (hitting optimal line)</li>
                  <li>• Exit trajectory consistency</li>
                  <li>• G-force loading patterns</li>
                </ul>
              </div>
              <div className="space-y-2 text-sm">
                <h4 className="font-semibold text-cyan-400">Race Management (32)</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Lap-to-lap consistency (pace variance)</li>
                  <li>• Tire wear rate</li>
                  <li>• Fuel efficiency</li>
                  <li>• Overtaking frequency</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-purple-500/10 rounded-lg p-4 mt-6">
            <h4 className="font-semibold text-purple-400 mb-2">Training Process</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Dataset:</span>
                <span className="font-mono">180,000 lap samples (20 drivers × 9,000 laps)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Loss Function:</span>
                <span className="font-mono">MSE (Reconstruction Error)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Epochs:</span>
                <span className="font-mono">100 (early stopping at epoch 78)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Final Loss:</span>
                <span className="font-mono text-green-400">0.012</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Visualizations */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-purple-400">Performance Analysis</CardTitle>
          <CardDescription>
            Driver style clustering and characterization
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <Tabs defaultValue="radar" className="w-full" suppressHydrationWarning>
            <TabsList className="grid grid-cols-3 w-full">
              <TabsTrigger value="radar">Driver Profiles</TabsTrigger>
              <TabsTrigger value="embedding">Embedding Space</TabsTrigger>
              <TabsTrigger value="features">Feature Importance</TabsTrigger>
            </TabsList>

            <TabsContent value="radar" className="space-y-4 mt-6">
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={driverStyleData}>
                    <PolarGrid stroke="#374151" />
                    <PolarAngleAxis dataKey="characteristic" stroke="#9ca3af" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#9ca3af" />
                    <Radar name="Verstappen" dataKey="verstappen" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} />
                    <Radar name="Hamilton" dataKey="hamilton" stroke="#22c55e" fill="#22c55e" fillOpacity={0.3} />
                    <Radar name="Leclerc" dataKey="leclerc" stroke="#f97316" fill="#f97316" fillOpacity={0.3} />
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div className="bg-red-500/10 rounded p-2">
                  <strong className="text-red-400">Verstappen:</strong> Extreme aggression (98 overtaking), excellent qualifying (97)
                </div>
                <div className="bg-green-500/10 rounded p-2">
                  <strong className="text-green-400">Hamilton:</strong> Peak consistency (98), masterful tire management (95)
                </div>
                <div className="bg-orange-500/10 rounded p-2">
                  <strong className="text-orange-400">Leclerc:</strong> Qualifying specialist (95), high aggression (88 overtaking)
                </div>
              </div>
            </TabsContent>

            <TabsContent value="embedding" className="space-y-4 mt-6">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="x" stroke="#9ca3af" label={{ value: 'Embedding Dimension 1', position: 'insideBottom', offset: -5 }} />
                    <YAxis dataKey="y" stroke="#9ca3af" label={{ value: 'Embedding Dimension 2', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                      cursor={{ strokeDasharray: '3 3' }}
                      content={({ payload }: any) => {
                        if (!payload || !payload[0]) return null;
                        const data = payload[0].payload;
                        return (
                          <div className="bg-gray-900 rounded p-2 text-xs">
                            <div className="font-bold">{data.driver}</div>
                            <div className="text-muted-foreground">Cluster: {data.cluster}</div>
                          </div>
                        );
                      }}
                    />
                    {embeddingSpace2D.map((entry, index) => (
                      <Scatter
                        key={index}
                        data={[entry]}
                        fill={CLUSTER_COLORS[entry.cluster]}
                        name={entry.driver}
                      />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <p className="text-sm text-muted-foreground">
                t-SNE projection of 32D embeddings to 2D. Clustering reveals 3 distinct styles: 
                <span className="text-red-400 font-semibold"> Aggressive</span> (top-right), 
                <span className="text-green-400 font-semibold"> Consistent</span> (right), 
                <span className="text-blue-400 font-semibold"> Balanced</span> (center).
              </p>
            </TabsContent>

            <TabsContent value="features" className="space-y-4 mt-6">
              <div className="h-[350px]">
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
                    <Bar dataKey="importance" fill="#a855f7" name="Importance (%)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-sm text-muted-foreground">
                Brake point variance (28%) most distinguishes driver styles. Aggressive drivers have high variance (late braking attempts), 
                consistent drivers show low variance (repeatable brake points).
              </p>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Real-World Integration */}
      <Card className="bg-black/40">
        <CardHeader>
          <CardTitle className="text-2xl text-purple-400">Real-World Integration</CardTitle>
          <CardDescription>
            Personalized strategy recommendations
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="bg-gradient-to-br from-purple-900/30 to-black rounded-lg p-6">
            <h3 className="font-semibold text-purple-400 mb-4 flex items-center gap-2">
              <Target className="w-5 h-5" />
              Driver-Specific Strategy Example
            </h3>
            <CodeBlock
              title="Driver-Specific Strategy Example"
              language="text"
              code={`🏎️ PERSONALIZED STRATEGY - Verstappen (Aggressive) vs. Hamilton (Consistent)

Scenario: Lap 25, both drivers on Medium tires (20 laps old)
Target: Undercut rival ahead, pit window closing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 VERSTAPPEN (Embedding: Aggressive Cluster)
   Driver Style Characteristics:
   • Aggression: 95/100 (high overtaking propensity)
   • Consistency: 92/100 (can maintain pace on older tires)
   • Tire Management: 85/100 (harder on tires, but fast)
   
   📍 RECOMMENDATION: AGGRESSIVE UNDERCUT
   • Pit THIS LAP (maximize track position gain)
   • Fresh tire advantage + aggressive overtaking style = high success
   • Accept risk: Can push hard on out-lap without concern for tire life
   • Expected: 2-3 position gain from undercut + aggressive passes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 HAMILTON (Embedding: Consistent Cluster)
   Driver Style Characteristics:
   • Aggression: 75/100 (strategic overtakes, not desperate)
   • Consistency: 98/100 (exceptional lap-to-lap precision)
   • Tire Management: 95/100 (masterful at extending stint)
   
   📍 RECOMMENDATION: PATIENT OVERCUT
   • Stay out 3-4 more laps (extend tire life advantage)
   • Leverage consistency to maintain pace on older tires
   • Overcut rivals when they pit (gain track position without risk)
   • Expected: 1-2 position gain from overcut + tire strategy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ RESULT: Personalized strategies match driver strengths`}
            />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-green-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-5 h-5 text-green-400" />
                <span className="font-semibold text-sm">Personalized</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Recommendations adapt to driver style, maximizing strengths and minimizing weaknesses
              </p>
            </div>
            <div className="bg-blue-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Users className="w-5 h-5 text-blue-400" />
                <span className="font-semibold text-sm">Teammate Comparison</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Embedding distance quantifies style differences for balanced team strategies
              </p>
            </div>
            <div className="bg-purple-500/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-5 h-5 text-purple-400" />
                <span className="font-semibold text-sm">Real-Time Updates</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Embeddings update during race as driver adapts to conditions or tire strategy
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Helper Components

