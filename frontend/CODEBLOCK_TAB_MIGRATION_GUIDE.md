# CodeBlock and DocumentationTabs Migration Guide

## Status

### ✅ Completed
- FuelSection: 2 CodeBlocks replaced
- LaptimeSection: 2 CodeBlocks replaced  
- All sections: DocumentationHeader with white text
- CodeBlock component: Copy button moved to top-right

### 🔄 Remaining Tasks

#### CodeBlock Replacements Needed (5 sections, ~10 code blocks)
1. **TireSection** (2 blocks): Lines 129, 472
2. **PitSection** (2 blocks): Check for `<pre` tags
3. **AnomalySection** (2 blocks): Check for `<pre` tags
4. **DriverSection** (2 blocks): Check for `<pre` tags
5. **TrafficSection** (2 blocks): Check for `<pre` tags

#### Tab Replacements Needed (9 sections)
1. OverviewSection
2. FuelSection
3. LaptimeSection
4. TireSection
5. PitSection
6. AnomalySection
7. DriverSection
8. TrafficSection
9. ArchitectureSection

## Pattern for CodeBlock Replacement

### Step 1: Add Import
```typescript
import { DocumentationHeader, CodeBlock, DocumentationTabs } from './shared';
```

### Step 2: Replace `<pre>` with `<CodeBlock>`

**Before:**
```tsx
<div className="bg-gray-900/50 rounded-lg p-6">
  <pre className="text-muted-foreground">
{`code here`}
  </pre>
</div>
```

**After:**
```tsx
<CodeBlock
  title="Title Here"
  language="python"
  code={`code here`}
/>
```

## Pattern for Tab Replacement

### Replace Shadcn Tabs with DocumentationTabs

**Before:**
```tsx
<Tabs defaultValue="tab1">
  <TabsList>
    <TabsTrigger value="tab1">Tab 1</TabsTrigger>
    <TabsTrigger value="tab2">Tab 2</TabsTrigger>
  </TabsList>
  <TabsContent value="tab1">
    Content 1
  </TabsContent>
  <TabsContent value="tab2">
    Content 2
  </TabsContent>
</Tabs>
```

**After:**
```tsx
<DocumentationTabs
  defaultTab="tab1"
  tabs={[
    {
      id: 'tab1',
      label: 'Tab 1',
      content: <div>Content 1</div>
    },
    {
      id: 'tab2',
      label: 'Tab 2',
      content: <div>Content 2</div>
    }
  ]}
/>
```

## Quick Commands

### Find all sections with code blocks:
```bash
cd /Users/anhlam/hack-the-track/frontend/components/documentation
for file in *.tsx; do 
  count=$(grep -c "<pre" "$file" 2>/dev/null || echo 0)
  if [ "$count" -gt 0 ]; then 
    echo "$file: $count blocks"
  fi
done
```

### Find all sections with Shadcn Tabs:
```bash
cd /Users/anhlam/hack-the-track/frontend/components/documentation
for file in *.tsx; do 
  if grep -q "from '@/components/ui/tabs'" "$file" && ! grep -q "DocumentationTabs" "$file"; then 
    echo "$file needs DocumentationTabs"
  fi
done
```

## Next Steps

To complete all remaining tasks:
1. Add imports to remaining 5 sections (Tire, Pit, Anomaly, Driver, Traffic)
2. Replace ~10 code blocks across those sections
3. Replace Shadcn Tabs in 9 sections with DocumentationTabs

Estimated: ~1-2 hours of systematic work OR continue AI-assisted completion

