# Quick Reference: Layer Generation Rules

## 🎯 Use This Guide To Understand Layer Behavior

Each layer has **specific terrain requirements**. This table shows where each layer appears and why.

---

## 📋 Layer Formation Rules Summary

| Layer | Where It Forms | Elevation | Slope | Curvature | Why |
|-------|---------------|-----------|-------|-----------|-----|
| **Topsoil** | Stable vegetated slopes | Any (below treeline) | <30° | Any | Needs stability for organic accumulation |
| **Subsoil** | Below topsoil | Any | <30° | Any | Pedogenic B horizon development |
| **Clay** | Lake centers, swamps | LOW (<30%) | FLAT (<5°) | Concave | Only settles in still water |
| **Silt** | Floodplains, lake margins | LOW-MID (<60%) | <15° | Flat to concave | Slow-moving water |
| **Sand** | Rivers, beaches, dunes | LOW-MID (<70%) | <15° | Any | Moderate energy transport |
| **Gravel** | Alluvial fans, channels | MID-HIGH (50-80%) | 15-30° | Any | High energy, near source |
| **Colluvium** | Hillslope hollows | MID (30-80%) | 10-30° | Concave | Gravity accumulation |
| **Saprolite** | Weathered interfluves | MID-HIGH (40-85%) | <15° | Any | Stable surfaces, deep weathering |
| **Weathered BR** | Above fresh bedrock | Any | Any | Any | Transition zone (universal but thin) |
| **Sandstone** | Ancient desert/beach | Any (buried) | Any | Any | Consolidated ancient sand |
| **Shale** | Ancient deep basins | Any (buried) | Any | Any | Consolidated ancient mud |
| **Limestone** | Ancient shallow marine | Any (buried) | Any | Any | Carbonate platform deposits |
| **Granite** | Continental crust | Any | Any | Any | Crystalline basement |
| **Basement** | Bottom layer | Any | Any | Any | Foundation (always present) |

---

## 🚫 Critical "DO NOT APPEAR" Rules

### ❌ Clay NEVER on:
- Slopes >5° (any slope prevents settling)
- High elevations (>40%)
- Flowing water (stays suspended)
- Convex terrain (no water accumulation)

### ❌ Sand NEVER on:
- Steep slopes >30° (erodes immediately)
- Very high elevations >70% (too far from source)
- Deep still-water basins (too fine settles first)

### ❌ Topsoil NEVER on:
- Steep slopes >30° (erodes faster than forms)
- Bare rock faces (no parent material)
- Active erosion zones (removed as it forms)

### ❌ Colluvium NEVER on:
- Flat terrain (no gravity transport)
- Ridgetops (source area, not accumulation)
- Valley floors (fluvial processes dominate)

### ❌ Saprolite NEVER:
- Thick on steep slopes (erosion removes it)
- In young valleys (recently cut, not weathered)

---

## 📐 Slope Thresholds Explained

| Slope Range | Classification | Dominant Process | Typical Deposits |
|-------------|---------------|------------------|------------------|
| **0-5°** | Flat | Deposition (lowest energy) | Clay, silt, thick topsoil |
| **5-15°** | Gentle | Deposition to transport | Sand, silt, stable soils |
| **15-30°** | Moderate | Transport dominant | Gravel, colluvium, thin soils |
| **>30°** | Steep | Erosion dominant | Bedrock, minimal cover |

**Key Insight:** As slope increases, grain size increases (finer particles don't stick).

---

## 🌍 Elevation Zones and Their Geology

### 🏔️ Highlands (>70% max elevation)
**Characteristics:**
- High erosion rates
- Thin sediment cover
- Exposed bedrock common
- Minimal soil development

**Expected Layers:**
- Thin topsoil (0-0.3m)
- Minimal colluvium
- Thin saprolite
- **ABSENT:** Clay, silt, sand (eroded away)
- Exposed crystalline basement (granite, gneiss)

---

### 🏞️ Midlands (30-70% elevation)
**Characteristics:**
- Mixed erosion and deposition
- Variable sediment thickness
- Active hillslope processes
- Developed weathering profiles

**Expected Layers:**
- Moderate topsoil (0.3-1.0m)
- **Thick colluvium** (major accumulation zone)
- Deep saprolite (up to 20m on stable slopes)
- Some sand (if fluvial environment)
- **LIMITED:** Clay (requires flat terrain)

---

### 🏞️ Lowlands (<30% elevation)
**Characteristics:**
- Deposition dominates
- **Thick sediment accumulation**
- Fine-grained deposits
- Mature soil profiles

**Expected Layers:**
- Thick topsoil (0.5-1.5m)
- **Clay layers** (up to 20m in basins)
- **Silt layers** (up to 15m in floodplains)
- **Sand layers** (up to 25m in channels)
- Deep alluvial sequences

---

## 🔄 Terrain Type → Layer Stack Examples

### 🏔️ Mountain Peak (Elevation: 90%, Slope: 40°)
```
LAYER STACK:
Surface
  ↓ 0.0m Topsoil       ← ABSENT (too steep)
  ↓ 0.0m Clay          ← ABSENT (too steep)
  ↓ 0.0m Sand          ← ABSENT (too steep)
  ↓ 0.5m Colluvium     ← MINIMAL (slides off)
  ↓ 1.0m Saprolite     ← THIN (rapid erosion)
  ↓ 0.6m Weathered BR
  ↓ Granite/Gneiss     ← EXPOSED BASEMENT
```
**Total Sediment: 2.1m** (very thin)

---

### 🏞️ Hillslope (Elevation: 50%, Slope: 18°)
```
LAYER STACK:
Surface
  ↓ 0.3m Topsoil       ← PRESENT (stable enough)
  ↓ 0.4m Subsoil
  ↓ 0.0m Clay          ← ABSENT (too much slope)
  ↓ 0.0m Sand          ← ABSENT (not fluvial)
  ↓ 12m Colluvium      ← THICK (primary deposit)
  ↓ 8m Saprolite       ← DEEP (stable surface)
  ↓ 1.5m Weathered BR
  ↓ Sandstone/Shale    ← SEDIMENTARY BEDROCK
```
**Total Sediment: 22.2m** (thick regolith mantle)

---

### 🏞️ Valley Floor (Elevation: 10%, Slope: 2°)
```
LAYER STACK:
Surface
  ↓ 0.8m Topsoil       ← THICK (very stable)
  ↓ 1.2m Subsoil
  ↓ 18m Clay           ← MAXIMUM (lake deposit)
  ↓ 12m Silt           ← THICK (floodplain)
  ↓ 20m Sand           ← THICK (channel)
  ↓ 5m Gravel          ← BASE LAG
  ↓ 0.0m Colluvium     ← ABSENT (not a slope)
  ↓ 3m Saprolite
  ↓ Bedrock
```
**Total Sediment: 60m** (extremely thick valley fill)

---

### 🏜️ Plateau (Elevation: 65%, Slope: 4°)
```
LAYER STACK:
Surface
  ↓ 0.4m Topsoil       ← PRESENT (flat, stable)
  ↓ 0.0m Clay          ← ABSENT (not a basin)
  ↓ 8m Silt (Loess)    ← PRESENT (wind-blown)
  ↓ 0.0m Colluvium     ← ABSENT (too flat)
  ↓ 15m Saprolite      ← VERY DEEP (old surface)
  ↓ 2m Weathered BR
  ↓ Ancient Crust      ← SHIELD AREA
```
**Total Sediment: 25.4m** (weathered mantle)

---

## 🧠 Mental Model: "Read the Terrain"

### Step 1: Look at ELEVATION
- **Low** → Sediment sink (thick deposits)
- **Mid** → Transition zone (mixed)
- **High** → Erosion source (thin cover)

### Step 2: Look at SLOPE
- **Flat** → Fine sediments (clay, silt)
- **Gentle** → Medium sediments (sand, loam)
- **Moderate** → Coarse sediments (gravel, colluvium)
- **Steep** → Bedrock exposure

### Step 3: Look at CURVATURE
- **Concave** (hollow) → Accumulation zone
- **Flat** → Stable zone
- **Convex** (ridge) → Erosion zone

### Step 4: Combine to Predict Layers
**Example:** Low + Flat + Concave = **Clay-rich basin**  
**Example:** Mid + Moderate + Concave = **Colluvium-filled hollow**  
**Example:** High + Steep + Convex = **Exposed bedrock ridge**

---

## 📊 Expected Coverage Percentages

Based on typical mixed terrain (mountains, valleys, hillslopes):

| Layer | Expected Coverage | Why |
|-------|------------------|-----|
| Topsoil | 5-15% | Only stable slopes |
| Clay | <1-5% | Only deep, flat basins |
| Silt | 5-20% | Floodplains, lake margins |
| Sand | 2-10% | Channels, dunes |
| Gravel | 1-5% | Mountain fronts, channels |
| Colluvium | 20-40% | Hillslopes (most common terrain) |
| Saprolite | 80-100% | Nearly universal weathering |
| Bedrock | 100% | Always present at depth |

**If you see different percentages, check your terrain characteristics!**

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Expecting Every Layer Everywhere
**Wrong thinking:** "Why doesn't my map have clay everywhere?"  
**Reality:** Clay is RARE - only forms in specific flat, low-energy basins.

### ❌ Mistake 2: Uniform Thickness
**Wrong thinking:** "Why is colluvium 0m here and 15m there?"  
**Reality:** Thickness varies with local terrain - hollows accumulate, ridges don't.

### ❌ Mistake 3: Ignoring Slope Effects
**Wrong thinking:** "Sand should be on this 25° slope."  
**Reality:** Sand erodes on slopes >15° - can't accumulate.

### ❌ Mistake 4: Wrong Elevation Expectations
**Wrong thinking:** "Why no clay at 500m elevation?"  
**Reality:** Clay forms in lowlands (<30% of total range), not mid-elevations.

---

## ✅ Validation Checklist

Use this to check if your generated terrain is realistic:

- [ ] **Clay appears on <5% of map** (if more, check slope thresholds)
- [ ] **Sand appears on 2-10% of map** (if more, check energy conditions)
- [ ] **Topsoil absent on steep slopes** (>30°)
- [ ] **Colluvium concentrated in mid-elevations** (30-70%)
- [ ] **Saprolite nearly universal** (but variable thickness)
- [ ] **Thickest sediments in valleys** (lowlands)
- [ ] **Thinnest sediments on peaks** (highlands)
- [ ] **Different terrains have different stacks** (not all identical)
- [ ] **Lateral gradients are smooth** (no abrupt jumps)
- [ ] **Vertical sequences make sense** (clay above sand in valleys, etc.)

---

## 🔬 Scientific Basis Summary

| Principle | What It Means | Implementation |
|-----------|---------------|----------------|
| **Hjulström Curve** | Grain size ∝ flow energy | Clay in still water, gravel in fast flow |
| **Walther's Law** | Vertical facies = lateral environments | Valley sequences mirror lateral gradients |
| **Erosion-Deposition Balance** | Slope controls sediment retention | Thick on flat, thin on steep |
| **Weathering Profile** | Time + stability → deep weathering | Saprolite thick on old, stable surfaces |
| **Stratigraphic Superposition** | Older layers below younger | No floating layers, proper stacking |

---

## 📚 Further Reading

- **Boggs (2011)** - Ch. 4-5: Sediment types and deposition
- **USGS Professional Paper 1396** - Sediment transport mechanics
- **Dunne & Leopold (1978)** - Hillslope processes
- **Reading (1996)** - Depositional environment models

---

**Quick Reference Version:** 1.0  
**Date:** December 8, 2025  
**Purpose:** Fast lookup for layer behavior rules
