# Config Comparison Analysis Report
============================================================

## Data Overview
- Total configurations analyzed: 4
- Total episodes: 186
- Obstacle densities: [0.0, 0.003, 0.005]

## Configuration Descriptions
- **Config A**: VFH-Fuzzy only (No branching/integration, No learning)
- **Config B**: Pre-trained model (No branching/integration)
- **Config C**: Branching/Integration enabled (No learning)
- **Config D**: Branching/Integration + Learning

## Performance Summary
### Config A
- Episodes analyzed: 16
- Average exploration rate: 0.3891 ± 0.2058
- Average steps taken: 50.0
- Exploration efficiency: 0.007782

### Config B
- Episodes analyzed: 110
- Average exploration rate: 0.2023 ± 0.0524
- Average steps taken: 50.0
- Exploration efficiency: 0.004046

### Config C
- Episodes analyzed: 30
- Average exploration rate: 0.3392 ± 0.1674
- Average steps taken: 50.0
- Exploration efficiency: 0.006784

### Config D
- Episodes analyzed: 30
- Average exploration rate: 0.2710 ± 0.0981
- Average steps taken: 50.0
- Exploration efficiency: 0.005420

## Key Findings
- **Highest exploration rate**: Config A (0.3891)
- **Most efficient**: Config A (0.007782 exploration/step)
- **Fastest completion**: Config A (50.0 steps)

## Statistical Significance
Based on pairwise t-tests (p < 0.05):
- Config A vs B: p=0.0000, effect size=1.2437
- Config A vs D: p=0.0113, effect size=0.7324
- Config B vs C: p=0.0000, effect size=1.1037
- Config B vs D: p=0.0000, effect size=0.8732
