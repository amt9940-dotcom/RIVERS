
# Geological Layer Generation Rules
## Comprehensive Rules for Realistic Stratigraphy Based on Topography and Layer Relationships

---

## Table of Contents
1. [Topographic Region Classification](#topographic-region-classification)
2. [Classification System](#classification-system)
3. [Surface Parameter Rules](#surface-parameter-rules)
4. [Explicit Parameter Rules by Layer Family](#explicit-parameter-rules-by-layer-family)
5. [Layer-Specific Rules by Region](#layer-specific-rules-by-region)
6. [Layer Relationship Rules](#layer-relationship-rules)
7. [Thickness Constraints](#thickness-constraints)
8. [Implementation Guidelines](#implementation-guidelines)

---

## 1. Topographic Region Classification

### 1.1 Region Types

The terrain is classified into distinct geomorphic zones, each with characteristic depositional and erosional processes:

#### **Ridges and Steep Mountain Slopes**
- **Detection Criteria:**
  - Slope > 25-30° (steep threshold)
  - Elevation in top 30% of map
  - Convex curvature (negative Laplacian)
  - High local relief

- **Geological Characteristics:**
  - Active erosion zone
  - Minimal deposition
  - Bedrock exposure common
  - Thin or absent regolith

#### **Steep Slope Toes and Alluvial Fans**
- **Detection Criteria:**
  - Moderate slope: 10-25°
  - Adjacent to steep upslope terrain
  - High local relief
  - Positioned at base of steep slopes

- **Geological Characteristics:**
  - Coarse-grained deposition
  - High-energy environment
  - Fan-shaped deposits
  - Transitional between erosional and depositional zones

#### **Valley Bottoms and Low-Slope Lowlands**
- **Detection Criteria:**
  - Slope < 5-8° (low slope threshold)
  - Elevation in bottom 30% of map
  - High flow accumulation
  - Low Height Above Nearest Drainage (HAND)

- **Geological Characteristics:**
  - Fine-grained deposition
  - Low-energy environment
  - Fluvial/alluvial processes
  - Thick sediment accumulation

#### **Broad, Low-Relief Platforms**
- **Detection Criteria:**
  - Slope < 2-3° (very low slope)
  - Elevation near "sea level" (median ± 15% of range)
  - Low local relief over broad area
  - Large smoothing window shows uniform elevation

- **Geological Characteristics:**
  - Carbonate platform environment
  - Shallow marine conditions
  - Low siliciclastic input
  - Sheet-like geometry

#### **Closed Depressions / Interior Basins**
- **Detection Criteria:**
  - Local topographic minima
  - Low flow accumulation (sinks)
  - Low slope and low local relief
  - No drainage outlet

- **Geological Characteristics:**
  - Evaporite deposition
  - Restricted circulation
  - Fine-grained surrounding sediments
  - Thickest at basin center

---

## 2. Classification System

### 2.1 Elevation and Slope Classification

**Elevation Normalization:**
```
E_rel = (z - z_min) / (z_max - z_min)   # 0 = global low, 1 = global high
```

**Elevation Classes:**
- **LOW:** E_rel < 0.3
- **MID:** 0.3 ≤ E_rel ≤ 0.7
- **HIGH:** E_rel > 0.7

**Slope Classes (S_deg = slope in degrees):**
- **FLAT:** S_deg < 3°
- **GENTLE:** 3° ≤ S_deg < 8°
- **MODERATE:** 8° ≤ S_deg ≤ 25°
- **STEEP:** S_deg > 25°

### 2.2 Regional Classification

**Region Types** (computed in large window, e.g., 5-10% of map width):

- **UPLAND:**
  - Mean E_rel > 0.6
  - Mean S_deg > 10°
  - High relief, rugged terrain

- **BASIN:**
  - Mean E_rel < 0.4
  - Mean S_deg < 8°
  - Low relief, depositional environment

- **PLATFORM:**
  - 0.4 ≤ Mean E_rel ≤ 0.6
  - Mean S_deg < 4°
  - Low relief, broad areas

- **FOOTHILL:**
  - Transitional between UPLAND and BASIN
  - Mean S_deg 8-20°
  - Moderate relief

### 2.3 "Not in Mountains" Enforcement Rule

**Core Principle:** For lithologies that "don't belong in mountains":

For cells with `region_type = UPLAND` AND `E_rel > 0.7`:
- Set deposition probability = 0 for that lithology
- Only allow at depth if pre-existing older layers put it there
- Do NOT add it as a new package near the surface

**Examples:**
- Evaporite, Limestone, Dolomite, fine mudstones: Deposition allowed only in BASIN/PLATFORM, and E_rel < 0.6
- Thick sandstones: Deposition only where region_type ≠ UPLAND, and S_deg < 10°
- Conglomerate: Deposition only where region_type is FOOTHILL/UPLAND edge, not deep BASIN or high ridge crest

---

## 3. Surface Parameter Rules

### 3.1 Elevation-Based Rules

| Elevation Zone | Regolith Thickness | Sedimentary Cover | Basement Exposure |
|---------------|---------------------|-------------------|-------------------|
| **High (>70th percentile)** | Very thin (0.1-0.3x normal) | Strongly suppressed (0.05-0.2x) | Common |
| **Moderate (30-70th percentile)** | Normal (0.8-1.2x) | Normal (0.8-1.2x) | Rare |
| **Low (<30th percentile)** | Thick (1.2-2.0x) | Enhanced (1.5-2.5x) | Never |

**Rules:**
- Higher elevation → thinner regolith, more bedrock exposure
- Lower elevation → thicker sediments, deeper burial
- Basement only crops out at high elevations with thin cover

### 2.2 Slope-Based Rules

| Slope Range | Regolith Response | Sediment Preservation | Deposition Type |
|------------|-------------------|----------------------|----------------|
| **>30° (Very Steep)** | 5-15% of normal | 5-20% preserved | None (erosion only) |
| **20-30° (Steep)** | 20-40% of normal | 20-40% preserved | Minimal |
| **10-20° (Moderate)** | 60-100% of normal | 60-100% preserved | Colluvium, fan deposits |
| **5-10° (Gentle)** | 100-150% of normal | 100-150% preserved | Alluvium, fine sediments |
| **<5° (Very Gentle)** | 150-200% of normal | 150-250% preserved | Fine-grained, carbonates |

**Rules:**
- Steeper slopes → active erosion, thin regolith
- Gentle slopes → deposition, thick regolith
- Optimal saprolite formation: 5-20° slopes
- Colluvium accumulation: 10-25° slopes with concave curvature

### 2.3 Curvature/Shape Rules

**Convex (Ridges):**
- Negative Laplacian
- Thin regolith (10-20% of normal)
- Bedrock exposure common
- Suppress all sedimentary deposition (0.05-0.1x)

**Concave (Hollows/Valleys):**
- Positive Laplacian
- Thick regolith (150-200% of normal)
- Colluvium accumulation
- Enhanced fine-grained deposition (1.5-2.0x)

**Planar (Platforms):**
- Near-zero Laplacian
- Moderate regolith
- Carbonate deposition favored
- Sheet-like geometry

### 2.4 Flow Accumulation Rules

| Flow Accumulation | Alluvium Thickness | Fine-Grained Facies | Channel Deposits |
|------------------|-------------------|---------------------|------------------|
| **High (>75th percentile)** | 2-8 m | Enhanced (1.5-2.0x) | Common |
| **Moderate (25-75th percentile)** | 0.5-3 m | Normal (0.8-1.2x) | Occasional |
| **Low (<25th percentile)** | 0-0.5 m | Suppressed (0.5-0.8x) | Rare |

**Rules:**
- High flow accumulation → valley bottoms, thick alluvium
- Low flow accumulation → interfluves, thin alluvium
- Alluvium restricted to cells with flow_accum > 50th percentile AND slope < 5°

---

## 4. Explicit Parameter Rules by Layer Family

**Note:** The thickness ranges and parameter thresholds below are **implementation parameters for the generator**, not universal geologic laws. They provide concrete, testable rules for where each layer can exist and how thick it can be, based on local and regional topographic conditions.

Each layer family includes:
- **Thickness ranges** (min-max for the generator)
- **Where it is allowed** (based on local elevation, slope, and regional classification)
- **Explicit rules** (deposition masks and suppression conditions)

### 4.1 Basement & Crystalline Rocks

**Layers:** BasementFloor, Basement, AncientCrust, Gneiss, Granite

#### Thickness Ranges
- **Basement stack (all together):** 500-3000 m visible
- **Individual subdivision:** Variable, but total basement stack is effectively "infinite" for modeling purposes
- **Note:** Real basins have sedimentary shells ~1-10+ km thick over basement; average sedimentary shell on continents is ~1.8 km

#### Where They Exist
- **Always:** Form the lowest units everywhere
- **Surface exposure:** Only where erosion has stripped cover
  - Preferentially in **UPLAND** regions with **HIGH** elevation (E_rel > 0.7) and **STEEP/MODERATE** slopes (S_deg > 8°)
  - Should NOT create new basement at shallow depth in low basins

#### Rules
1. Basement always fills from the bottom up until other layers are added
2. At cells where total cover thickness ≤ 0 → surface lithology = one of these crystalline units
3. **Do NOT tie basement geometry to detailed surface roughness** - use a smooth regional structural surface
4. Basement exposure mask: `(region_type == UPLAND) AND (E_rel > 0.7) AND (S_deg > 8°)`

---

### 4.2 Basalt

**Layer:** Basalt (volcanic flows or plateau basalts)

#### Thickness Ranges
- **Single basalt package:** 10-100 m typical
- **Thick stack:** Up to 200-300 m (real flood basalts can be >1 km, but that's overkill for model)
- **Minimum:** 5 m

#### Where They Exist
- Can occur in both uplands and basins
- Common as relatively flat-lying flow tops or plateau caps
- Not tied strongly to current slope (treat as older event)

#### Rules
1. Allow basalt only where S_deg (paleo-slope) was **GENTLE** or **FLAT** when emplaced (S_deg < 8°)
2. In **UPLAND** regions, let basalt form flat caps on highs (good for mesa/plateau feel)
3. **Do NOT let basalt be a thin noisy ribbon under every cell** - it should be present in belts/patches, not everywhere
4. Deposition mask: `(S_deg < 8°) AND (region_type ∈ {UPLAND, BASIN, PLATFORM})`
5. Spatial clustering: Use patchy distribution, not uniform

---

### 4.3 Carbonates & Evaporites

#### Limestone

**Thickness Ranges:**
- **Individual formations:** 10-100 m per layer
- **Stacked packages:** Can total several hundred meters
- **Minimum:** 5 m
- **Maximum:** 350 m (in thick platform successions)

**Where They Exist:**
- **PLATFORM** regions: low relief, mean slope small
- Elevation near "sea level": **MID** elevation band (0.3 ≤ E_rel ≤ 0.7)
- Local slopes mostly **FLAT-GENTLE** (S_deg < 5°)

**Rules:**
1. Only deposit limestone where:
   - `region_type ∈ {PLATFORM, BASIN with very low relief}`
   - `S_deg < 5°`
   - `0.3 ≤ E_rel ≤ 0.7`
2. In **UPLANDs / HIGH E_rel (E_rel > 0.7):** Do NOT deposit new limestone
3. Limestone there, if any, should be older units that have been uplifted/eroded into view, not fresh platforms on steep mountains
4. Deposition mask: `(region_type ∈ {PLATFORM, BASIN}) AND (S_deg < 5°) AND (0.3 ≤ E_rel ≤ 0.7) AND (region_type ≠ UPLAND)`
5. Suppression mask: `(region_type == UPLAND) OR (E_rel > 0.7) OR (S_deg > 10°)` → weight = 0

#### Dolomite

**Thickness Ranges:**
- **Per layer:** 5-100 m (same order as limestone)
- **Stacked:** Can total 100-200 m
- **Minimum:** 3 m

**Where They Exist:**
- Same environments as limestone, maybe slightly more restricted to inner/platform or restricted settings
- **MID** elevation (0.3 ≤ E_rel ≤ 0.7)
- **FLAT-GENTLE** slopes (S_deg < 5°)

**Rules:**
1. Only allowed where limestone is allowed
2. Optionally bias dolomite toward more interior / slightly higher E_rel within platforms, but still not in steep mountains
3. Deposition mask: Same as limestone, with optional E_rel bias toward 0.5-0.7
4. Suppression: Same as limestone

#### Evaporite

**Thickness Ranges:**
- **Single layer:** 10-200 m
- **Real evaporite sequences:** Can be 10s to 1000s of meters in large basins
- **Minimum:** 5 m
- **Maximum:** 200 m (thickest at basin center)

**Where They Exist:**
- Only in **BASIN** regions that are hydrologically closed:
  - **LOW** elevation (E_rel < 0.3, bottom 20-30%)
  - Very low slope (**FLAT-GENTLE**, S_deg < 3°)
  - Ideally near centers of topographic depressions

**Rules:**
1. Require:
   - `E_rel < 0.3`
   - `S_deg < 3°`
   - `region_type == BASIN`
   - Plus "closed-basin mask" (topographic depression)
2. **Never allow evaporite deposition in UPLAND / HIGH elevation regions**
3. Thickness maximum at basin center; taper to zero at basin margins
4. Deposition mask: `(region_type == BASIN) AND (E_rel < 0.3) AND (S_deg < 3°) AND (closed_basin_mask == True)`
5. Suppression: `(region_type == UPLAND) OR (E_rel > 0.3) OR (S_deg > 3°)` → weight = 0

---

### 4.4 Clastic Sedimentary Rocks

#### Sandstone

**Thickness Ranges:**
- **Per layer:** 5-150 m
- **Single formations:** Commonly 10-200 m
- **Stacked packages:** Can total >1 km
- **Minimum:** 3 m
- **Maximum:** 300 m (regional variation)

**Where They Exist:**
- Deposits in rivers, deltas, shorelines, eolian dunes
- Mainly where local slope is **low to moderate** (S_deg < 10-12°)
- Thickest in **BASIN** and **PLATFORM** regions
- Scarce as new deposits in rugged uplands

**Rules:**
1. For deposition, allow if:
   - `S_deg < 10-12°`
   - `region_type ∈ {BASIN, PLATFORM, FOOTHILL margin}`
   - Weight higher in **LOW/MID** elevation zones (E_rel < 0.7)
2. In **UPLAND, HIGH E_rel (E_rel > 0.7) and STEEP (S_deg > 25°):**
   - Deposition weight ≈ 0
   - Sandstone can appear only as previously deposited units now uplifted/eroded, so you don't grow new sandstone packages there
3. Deposition mask: `(S_deg < 12°) AND (region_type ∈ {BASIN, PLATFORM, FOOTHILL}) AND (E_rel < 0.7)`
4. Suppression mask: `(region_type == UPLAND) AND (E_rel > 0.7) AND (S_deg > 25°)` → weight = 0.05-0.1

#### Conglomerate

**Thickness Ranges:**
- **Per layer:** 10-200 m
- **Biggest fans:** Up to 300 m
- **Real proximal alluvial-fan deposits:** Can be hundreds to >1000 m
- **Minimum:** 5 m
- **Maximum:** 300 m

**Where They Exist:**
- Near mountain fronts:
  - Regions with high mean slope (**UPLAND edge / FOOTHILL**)
  - Local slopes **GENTLE-MODERATE** (3° ≤ S_deg ≤ 20°) at the toe of steeper slopes upslope

**Rules:**
1. Only deposit conglomerate where:
   - `region_type ≈ FOOTHILL / UPLAND edge` (high relief, mean S_deg 8-20°)
   - `3° ≤ S_deg ≤ 20°` (local slope)
   - There is steeper terrain upslope (use aspect + neighborhood check)
2. In deep **BASINS** far from steep sources: conglomerate weight ~0
3. Deposition mask: `(region_type ∈ {FOOTHILL, UPLAND_edge}) AND (3° ≤ S_deg ≤ 20°) AND (steep_upslope_exists)`
4. Suppression mask: `(region_type == BASIN) OR (S_deg < 3°) OR (S_deg > 25°)` → weight = 0.1-0.3

#### Siltstone, Mudstone, Shale, Clay (Lithified)

**Group:** Fine clastics

**Thickness Ranges:**
- **Per named layer:** 5-100 m
- **Individual formations:** Often 10-200 m
- **Thick mud/shale successions:** Can be 100s of meters to >1 km
- **Minimum:** 3 m
- **Maximum:** 400 m (for shale in deep basins)

**Where They Exist:**
- Low-energy parts of **BASINS** and floodplains:
  - `E_rel < 0.4`
  - `S_deg < 5-8°`
  - Strongest in **BASIN** regions

**Rules:**
1. In **BASIN** and low-slope zones:
   - High weights for mudstone/shale/siltstone/clay
2. In **HIGH-elevation UPLAND** regions:
   - Heavily suppress new deposition of these
   - They can still be present as deeply buried or uplifted older units, but not capping steep mountains
3. This ensures deep basins are mud-prone, not sandstone blankets
4. Deposition mask: `(region_type == BASIN) AND (E_rel < 0.4) AND (S_deg < 8°)`
5. Suppression mask: `(region_type == UPLAND) AND (E_rel > 0.7) AND (S_deg > 10°)` → weight = 0.05-0.2

**Individual Rules:**

- **Shale:** Primary fine-grained facies in basins, 20-400 m
- **Mudstone:** Deepest, quietest parts of basins, 30% of shale package in valleys, 5-120 m
- **Siltstone:** Intermediate between shale and sandstone, 20% of shale + 10% of sandstone in valleys, 5-80 m
- **Clay (lithified):** Very fine-grained, 5-50 m, restricted to deepest basins

---

### 4.5 Unconsolidated Clastics (Near-Surface)

#### Sand (Unlithified)

**Thickness Ranges:**
- **Typical:** 1-20 m
- **Locally more:** Up to 30 m in dune fields
- **Beach sands:** Usually a few meters thick
- **Minimum:** 0.5 m
- **Maximum:** 30 m

**Where They Exist:**
- Desert or coastal analogues: low slope areas, often mid to low relief
- Can exist at any E_rel, but needs **GENTLE/FLAT** slopes

**Rules:**
1. Only allow if `S_deg < 10°`
2. Emphasize plateau tops or broad valley floors, not steep slopes
3. In **HIGH, STEEP** uplands: sand thickness forced to near-zero (blown/washed away)
4. Deposition mask: `(S_deg < 10°) AND (NOT (region_type == UPLAND AND E_rel > 0.7 AND S_deg > 25°))`
5. Suppression: `(S_deg > 10°) OR (region_type == UPLAND AND E_rel > 0.7 AND S_deg > 25°)` → weight = 0.05

#### Silt & Clay (Unlithified)

**Thickness Ranges:**
- **Floodplain and lacustrine muds:** Typically 1-10 m
- **Big lakes:** Sometimes tens of meters
- **Minimum:** 0.5 m
- **Maximum:** 15 m

**Where They Exist:**
- Valley bottoms and lake basins:
  - `S_deg < 3-5°`
  - High flow accumulation or closed depressions

**Rules:**
1. Only deposit where valley-bottom or lake masks say so
2. Not allowed in steep uplands or strongly convex ridges
3. Deposition mask: `(S_deg < 5°) AND (valley_bottom_mask OR closed_basin_mask) AND (high_flow_accumulation)`
4. Suppression: `(S_deg > 5°) OR (region_type == UPLAND AND E_rel > 0.7)` → weight = 0

---

### 4.6 Regolith / Soil Layers

#### WeatheredBR (Weathered Bedrock) & Saprolite

**Thickness Ranges:**
- **Weathered rock + saprolite commonly:** 3-30 m
- **Can exceed:** 30 m in deeply weathered terrains
- **WeatheredBR:** 2-10 m
- **Saprolite:** 5-30 m
- **Minimum:** 0.5 m
- **Maximum:** 30 m (saprolite), 10 m (weatheredBR)

**Where They Exist:**
- Under most landscapes except very young/steep rock
- Thickest in low-moderate slopes and **MID** elevation zones

**Rules:**
1. Saprolite thickness ∝ exp(-k × S_deg):
   - At **STEEP** slopes (S_deg > 25°): cap at 0-5 m
   - At **GENTLE** slopes (S_deg < 8°): allow up to max
2. In very **HIGH UPLANDS** with strong relief: reduce saprolite/WeatheredBR (active erosion)
3. Optimal range: **GENTLE-MODERATE** slopes (5-20°)
4. Thickness formula: `T_saprolite = base × exp(-0.1 × S_deg) × (1 - 0.5 × (E_rel > 0.7))`
5. Suppression: `(S_deg > 30°) OR (region_type == UPLAND AND E_rel > 0.7 AND S_deg > 20°)` → weight = 0.1-0.2

#### Colluvium

**Thickness Ranges:**
- **Usually:** 1-10 m
- **Major hollows/fans:** Can be thicker, up to 15-18 m
- **Minimum:** 0.5 m
- **Maximum:** 18 m (can exceed in foot slopes)

**Where They Exist:**
- Concave midslopes and hollows:
  - **MODERATE** slopes (8-25°)
  - Positive curvature (concave)

**Rules:**
1. `T_colluvium = f(upslope_area, curvature, mid-slope S_deg)`
2. In valley bottoms (S_deg < 3°), reclass part of this as alluvium instead
3. Optimal: **MODERATE** slopes (10-25°) with concave curvature
4. Thickness increases with upslope contributing area of steep terrain
5. Deposition mask: `(8° ≤ S_deg ≤ 25°) AND (curvature > 0) AND (NOT valley_bottom)`
6. Suppression: `(S_deg < 8°) OR (S_deg > 25°) OR (curvature < 0)` → weight = 0.1-0.3

#### Subsoil / Soil

**Thickness Ranges:**
- **Mean soil thickness:** Often 0.5-1.5 m
- **Up to:** About 2 m
- **Tropical settings:** Can be thicker, but that's mostly weathered rock/saprolite below
- **Subsoil:** 0.3-1.5 m
- **Soil (if separated):** Similar range
- **Minimum:** 0.1 m
- **Maximum:** 2 m

**Where They Exist:**
- Everywhere except bare rock outcrops

**Rules:**
1. Tie to regolith thickness:
   - Soil/Subsoil thickness ≤ some fraction (e.g., 0.2-0.3) of total regolith
2. Reduce toward zero on the steepest convex ridges (exposed rock)
3. Thickness formula: `T_soil = 0.4 × T_total_regolith × (1 - 0.8 × (S_deg > 30°)) × (1 - 0.5 × (curvature < -threshold))`
4. Suppression: `(S_deg > 35°) OR (region_type == UPLAND AND E_rel > 0.7 AND S_deg > 25° AND curvature < -threshold)` → weight = 0.05-0.1

---

### 4.7 Summary Table: Deposition Masks by Layer

| Layer | Thickness Range | Region Type | Elevation (E_rel) | Slope (S_deg) | Additional Constraints |
|-------|----------------|-------------|------------------|---------------|----------------------|
| **Basement** | 500-3000 m | All | All | All | Always lowest; exposed in UPLAND, HIGH, STEEP |
| **Basalt** | 10-300 m | UPLAND, BASIN, PLATFORM | All | < 8° | Patchy distribution, not everywhere |
| **Limestone** | 10-350 m | PLATFORM, BASIN | 0.3-0.7 | < 5° | NOT in UPLAND or HIGH elevation |
| **Dolomite** | 5-200 m | PLATFORM, BASIN | 0.3-0.7 | < 5° | Same as limestone, slightly more restricted |
| **Evaporite** | 10-200 m | BASIN only | < 0.3 | < 3° | Closed basin required, NEVER in UPLAND |
| **Sandstone** | 5-300 m | BASIN, PLATFORM, FOOTHILL | < 0.7 | < 12° | NOT in UPLAND with HIGH elevation |
| **Conglomerate** | 10-300 m | FOOTHILL, UPLAND edge | All | 3-20° | Requires steep upslope, NOT in deep BASIN |
| **Shale** | 5-400 m | BASIN | < 0.4 | < 8° | NOT in UPLAND with HIGH elevation |
| **Mudstone** | 5-120 m | BASIN | < 0.4 | < 8° | Deepest basins, 30% of shale in valleys |
| **Siltstone** | 5-80 m | BASIN | < 0.4 | < 8° | Intermediate, 20% shale + 10% sandstone |
| **Sand (unlit.)** | 1-30 m | All | All | < 10° | NOT in HIGH, STEEP uplands |
| **Silt/Clay (unlit.)** | 0.5-15 m | BASIN, valleys | < 0.4 | < 5° | Valley bottoms, high flow accumulation |
| **Saprolite** | 5-30 m | All | All | 5-20° optimal | Suppressed on STEEP slopes |
| **Colluvium** | 0.5-18 m | All | All | 8-25° | Concave curvature, foot slopes |
| **Soil/Subsoil** | 0.1-2 m | All | All | All | Suppressed on very STEEP, convex ridges |

**Key Suppression Rules:**
- **UPLAND + HIGH (E_rel > 0.7) + STEEP (S_deg > 25°):** Suppress all new sedimentary deposition (weight = 0.05-0.1)
- **UPLAND + HIGH:** Suppress carbonates, evaporites, fine clastics (weight = 0)
- **Deep BASIN:** Suppress conglomerate (weight = 0.1-0.3)

---

## 5. Layer-Specific Rules by Region

### 3.1 Regolith Layers

#### **Topsoil (A Horizon)**
- **Thickness Range:** 0.1-1.8 m
- **Regional Rules:**
  - Ridges: 0.1-0.3 m (thin, often absent)
  - Steep slopes: 0.2-0.5 m
  - Gentle slopes: 0.5-1.2 m
  - Valleys: 0.8-1.8 m (thickest)
- **Slope Factor:** Thickness ∝ 1/slope (inverse relationship)
- **Curvature Factor:** 1.5x in concave areas, 0.5x on convex ridges

#### **Subsoil (B Horizon)**
- **Thickness Range:** 0.2-1.5 m
- **Regional Rules:**
  - Same pattern as topsoil but 60% of topsoil thickness
  - More persistent on steep slopes than topsoil
- **Slope Factor:** Similar to topsoil but less sensitive

#### **Colluvium**
- **Thickness Range:** 0.5-36 m (can exceed normal max in foot slopes)
- **Regional Rules:**
  - Ridges: 0.1-1 m (nearly absent)
  - Steep slopes: 0.5-3 m (thin)
  - **Foot slopes (10-25°, concave):** 5-36 m (THICKEST)
  - Valleys: 1-5 m (moderate)
- **Slope Factor:** Optimal at 10-25°, zero above 30°
- **Curvature Factor:** 2.0x in concave areas at foot of slopes
- **Upslope Area Factor:** Thickness increases with upslope contributing area of steep terrain

#### **Saprolite**
- **Thickness Range:** 0.5-30 m
- **Regional Rules:**
  - Ridges: 0.05-0.5 m (nearly absent)
  - Steep slopes (>25°): 0.2-2 m (very thin)
  - **Optimal slopes (5-20°):** 3-30 m (THICKEST)
  - Valleys: 1-8 m (moderate)
- **Slope Factor:** Peak at 5-20°, drops to zero above 30°
- **Elevation Factor:** Slightly enhanced on interfluves (moderate elevation)

#### **Weathered Bedrock Rind**
- **Thickness Range:** 0.4-6 m
- **Regional Rules:**
  - Ridges: 0.1-0.5 m (thin)
  - Steep slopes: 0.2-1 m
  - Gentle slopes: 1-4 m
  - Valleys: 1.5-6 m (thickest)
- **Slope Factor:** Moderate inverse relationship
- **Patchiness:** High spatial variability (patchy distribution)

### 3.2 Valley-Fill Sediments

#### **Clay**
- **Thickness Range:** 0-25 m
- **Regional Rules:**
  - **Valleys only:** Thickest in flattest, deepest parts
  - Ridges: 0 m (absent)
  - Steep slopes: 0 m (absent)
- **Slope Factor:** Thickness ∝ (1 - slope_norm)² (strong inverse)
- **Flow Accumulation:** Enhanced in high flow accumulation areas
- **Elevation Factor:** Only in bottom 30% of elevation range

#### **Silt**
- **Thickness Range:** 0-20 m
- **Regional Rules:**
  - **Valleys only:** Slightly broader distribution than clay
  - Ridges: 0 m (absent)
- **Slope Factor:** Moderate inverse relationship
- **Flow Accumulation:** Enhanced in moderate-high flow areas

#### **Sand (Valley-Fill)**
- **Thickness Range:** 0-30 m
- **Regional Rules:**
  - **Valleys only:** Higher-energy parts of valleys
  - Ridges: 0 m (absent)
- **Slope Factor:** Slight positive relationship (higher energy)
- **Flow Accumulation:** Enhanced in channel areas

### 3.3 Lithified Sedimentary Units

#### **Sandstone**
- **Thickness Range:** 0-300 m (regional variation)
- **Regional Rules:**
  - **Ridges:** 0-15 m (strongly suppressed, 95% reduction)
  - **Steep slopes:** 5-30 m (80% suppression)
  - **Fan toes:** 30-150 m (1.5x boost)
  - **Valleys:** 50-200 m (1.3x boost for fluvial sandstone)
  - **Platforms:** 10-50 m (suppressed, 0.4x)
  - **Closed basins:** 20-100 m (moderate)
- **Energy Rule:** Moderate-high energy environments
- **Slope Factor:** Suppressed on steep slopes, enhanced in moderate slopes
- **Basin Factor:** Intermediate basin positions (not deepest, not highest)

#### **Conglomerate**
- **Thickness Range:** 0-150 m
- **Regional Rules:**
  - **Ridges:** 0-5 m (nearly absent, 95% suppression)
  - **Steep slopes:** 0-10 m (strongly suppressed)
  - **Fan toes:** 20-150 m (3.0x boost - PRIMARY LOCATION)
  - **Valleys:** 0-20 m (suppressed, 0.3x - only in proximal, steeper segments)
  - **Platforms:** 0 m (absent)
  - **Closed basins:** 0-10 m (suppressed)
- **Energy Rule:** High-energy, proximal to source
- **Slope Factor:** Optimal at 10-25° (fan slopes)
- **Upslope Factor:** Requires steep upslope terrain nearby

#### **Shale**
- **Thickness Range:** 0-400 m (regional variation)
- **Regional Rules:**
  - **Ridges:** 0-20 m (strongly suppressed, 95% reduction)
  - **Steep slopes:** 10-40 m (80% suppression)
  - **Fan toes:** 20-80 m (suppressed, 0.5x)
  - **Valleys:** 100-400 m (2.5x boost - PRIMARY LOCATION)
  - **Platforms:** 20-60 m (suppressed, 0.4x)
  - **Closed basins:** 80-300 m (enhanced, 1.5x)
- **Energy Rule:** Low-energy, quiet water
- **Slope Factor:** Enhanced in low-slope areas
- **Basin Factor:** Deepest parts of basins

#### **Mudstone**
- **Thickness Range:** 0-120 m (typically 20-40% of shale package)
- **Regional Rules:**
  - **Valleys only:** Deepest, quietest parts
  - Forms as 30% of shale package in valleys
  - Absent or minimal elsewhere
- **Energy Rule:** Very low energy, quietest water
- **Basin Factor:** Deepest basins only

#### **Siltstone**
- **Thickness Range:** 0-80 m
- **Regional Rules:**
  - Intermediate between shale and sandstone
  - Forms as 20% of shale + 10% of sandstone in valleys
  - Slightly coarser than mudstone
- **Energy Rule:** Low-moderate energy

#### **Limestone**
- **Thickness Range:** 0-350 m (regional variation)
- **Regional Rules:**
  - **Ridges:** 0-15 m (suppressed, 95% reduction)
  - **Steep slopes:** 5-30 m (suppressed)
  - **Fan toes:** 0-20 m (suppressed, 0.3x)
  - **Valleys:** 20-100 m (suppressed, 0.5x)
  - **Platforms:** 100-350 m (3.0x boost - PRIMARY LOCATION)
  - **Closed basins:** 10-50 m (suppressed)
- **Energy Rule:** Low-energy, clear water, low siliciclastic input
- **Slope Factor:** Very low slope required (<2-3°)
- **Platform Factor:** Broad, low-relief platforms near "sea level"

#### **Dolomite**
- **Thickness Range:** 0-140 m (typically 30-40% of limestone package)
- **Regional Rules:**
  - **Platforms only:** Forms as 40% of limestone on platforms
  - Dolomitized limestone
  - Absent elsewhere
- **Energy Rule:** Same as limestone but more persistent

#### **Evaporite**
- **Thickness Range:** 0-50 m
- **Regional Rules:**
  - **Closed basins ONLY:** 5.0x boost, zero elsewhere
  - Thickest at basin center, thinning outward
  - Maximum 50 m at center
- **Energy Rule:** Very low energy, restricted circulation
- **Basin Factor:** Must be in closed topographic depression
- **Distance Factor:** Thickness decreases with distance from basin center

### 3.4 Crystalline Basement Units

#### **Granite**
- **Thickness:** Variable, fraction of basement total
- **Regional Rules:**
  - More common in low, gentle basins
  - Less common in high, steep areas
- **Elevation Factor:** Slight negative correlation
- **Slope Factor:** Slight negative correlation

#### **Gneiss**
- **Thickness:** Variable, fraction of basement total
- **Regional Rules:**
  - More common in high, steep areas
  - Less common in low basins
- **Elevation Factor:** Positive correlation
- **Slope Factor:** Positive correlation

#### **Basalt**
- **Thickness:** Variable, typically 1-6% of basement
- **Regional Rules:**
  - Slightly more in basins
  - Patchy distribution
- **Elevation Factor:** Slight negative correlation

#### **Ancient Crust**
- **Thickness:** Variable, background component
- **Regional Rules:**
  - Modest background everywhere
  - Slightly enhanced at higher elevations

---

## 6. Layer Relationship Rules

### 4.1 Vertical Ordering (Top to Bottom)

**Strict Order (Never Violated):**
1. **Deposits (if present):** Loess → DuneSand → Till → Alluvium
2. **Regolith:** Topsoil → Subsoil → Colluvium → Saprolite → WeatheredBR
3. **Valley-Fill:** Clay → Silt → Sand (valley-fill)
4. **Sedimentary:** Sandstone → Conglomerate → Shale → Mudstone → Siltstone → Limestone → Dolomite → Evaporite
5. **Crystalline:** Granite → Gneiss → Basalt → AncientCrust
6. **Base:** Basement → BasementFloor

### 4.2 Layer Presence Rules

#### **Alluvium**
- **Requires:** Valley bottom (slope < 5°, high flow accumulation)
- **Cannot exist:** On ridges, steep slopes, or platforms
- **Thickness:** Proportional to flow accumulation and inverse of slope

#### **Colluvium**
- **Requires:** Moderate slopes (10-25°) with concave curvature OR foot of steep slopes
- **Cannot exist:** On very steep slopes (>30°) or flat valley bottoms
- **Thickness:** Increases with upslope contributing area

#### **Saprolite**
- **Requires:** Moderate slopes (5-20°)
- **Cannot exist:** On very steep slopes (>30°) or sharp ridges
- **Thickness:** Peak at 5-20° slopes

#### **Conglomerate**
- **Requires:** Fan toes (moderate slope at base of steep terrain)
- **Cannot exist:** On ridges, platforms, or deep valleys
- **Thickness:** Limited to fan settings, max 150 m

#### **Evaporite**
- **Requires:** Closed topographic basin
- **Cannot exist:** Outside closed basins (zero probability)
- **Thickness:** Thickest at basin center, thinning to edges

#### **Carbonates (Limestone/Dolomite)**
- **Requires:** Broad, low-relief platforms (slope < 2-3°)
- **Cannot exist:** On steep slopes or in deep basins
- **Thickness:** Enhanced on platforms, suppressed elsewhere

### 4.3 Layer Thickness Relationships

#### **Regolith Thickness Relationships**
- **Total Regolith = Topsoil + Subsoil + Colluvium + Saprolite + WeatheredBR**
- **Topsoil:** 40% of total soil (Topsoil + Subsoil)
- **Subsoil:** 60% of total soil
- **Colluvium:** Independent, but competes with saprolite for space
- **Saprolite:** Independent, but suppressed where colluvium is thick
- **WeatheredBR:** Independent, patchy

#### **Valley-Fill Relationships**
- **Clay + Silt + Sand (valley-fill) = Total valley-fill package**
- **Clay:** Thickest in flattest, deepest valleys
- **Silt:** Broader distribution than clay
- **Sand (valley-fill):** Higher-energy parts of valleys
- **Total thickness:** 0-75 m in valleys, 0 m elsewhere

#### **Sedimentary Package Relationships**
- **Total Sedimentary = Sandstone + Conglomerate + Shale + Mudstone + Siltstone + Limestone + Dolomite + Evaporite**
- **Mudstone:** 30% of shale package in valleys
- **Siltstone:** 20% of shale + 10% of sandstone in valleys
- **Dolomite:** 40% of limestone package on platforms
- **Evaporite:** Independent, only in closed basins

#### **Basement Relationships**
- **Total Basement = Granite + Gneiss + Basalt + AncientCrust + Generic Basement**
- **Fractions sum to ≤ 0.85** (remaining is generic Basement)
- **Granite + Gneiss + Basalt + AncientCrust = 85% of basement total**

### 4.4 Layer Interface Rules

#### **Smoothness Requirements**
- **Deep interfaces (basement, evaporite, limestone):** Strong smoothing (sigma = 3.0+)
- **Intermediate interfaces (shale, sandstone):** Moderate smoothing (sigma = 2.0)
- **Shallow interfaces (regolith):** Light smoothing (sigma = 1.0)
- **All interfaces:** Additional box blur for lateral continuity

#### **Interface Continuity**
- **No pixel-scale jumps:** Interfaces must be smooth over many grid cells
- **Minimum thickness:** 0.05 m to be considered present
- **Ordering enforcement:** After smoothing, re-enforce vertical ordering

#### **Basement Interface**
- **Long-wavelength only:** Basement surface reflects tectonic structure, not surface noise
- **Smoothing window:** 15% of domain size minimum
- **Tectonic control:** Regional dip + smooth undulation + crustal flexure
- **No short-wavelength noise:** Strongly filtered

---

## 7. Thickness Constraints

### 5.1 Maximum Thickness Limits

| Layer | Maximum Thickness | Regional Variation |
|-------|------------------|-------------------|
| Topsoil | 1.8 m | Valleys only |
| Subsoil | 1.5 m | Valleys only |
| Colluvium | 36 m | Foot slopes only |
| Saprolite | 30 m | Optimal slopes only |
| WeatheredBR | 6 m | Valleys |
| Clay | 25 m | Valleys only |
| Silt | 20 m | Valleys only |
| Sand (valley-fill) | 30 m | Valleys only |
| Alluvium | 8 m | Valleys only |
| Sandstone | 300 m | Regional |
| Conglomerate | 150 m | Fan toes only |
| Shale | 400 m | Valleys/basins |
| Mudstone | 120 m | Valleys only |
| Siltstone | 80 m | Valleys only |
| Limestone | 350 m | Platforms |
| Dolomite | 140 m | Platforms only |
| Evaporite | 50 m | Closed basins only |

### 5.2 Minimum Thickness Thresholds

- **All layers:** 0.05 m minimum to be considered present
- **Regolith on ridges:** Can be 0 m (bare bedrock)
- **Sedimentary on ridges:** Can be 0 m (eroded away)
- **Evaporite outside basins:** 0 m (absent)

### 5.3 Thickness Scaling Factors by Region

| Region | Regolith Factor | Sedimentary Factor | Notes |
|--------|----------------|-------------------|-------|
| Ridges | 0.1-0.2x | 0.05-0.1x | Strong suppression |
| Steep Slopes | 0.2-0.4x | 0.2-0.4x | Moderate suppression |
| Fan Toes | 0.6-1.0x | 1.5-3.0x (conglomerate) | Conglomerate boost |
| Valleys | 1.2-2.0x | 1.5-2.5x (fine-grained) | Fine-grained boost |
| Platforms | 0.8-1.2x | 3.0x (carbonates) | Carbonate boost |
| Closed Basins | 1.0-1.5x | 5.0x (evaporites) | Evaporite boost |

---

## 8. Implementation Guidelines

### 6.1 Processing Order

1. **Classify geomorphic zones** (ridges, fan toes, valleys, platforms, closed basins)
2. **Compute surface parameters** (slope, curvature, flow accumulation, elevation percentiles)
3. **Generate regolith layers** (top-down, strongly topography-dependent)
4. **Generate valley-fill sediments** (valleys only)
5. **Generate sedimentary units** (energy-based facies assignment)
6. **Generate basement** (smooth, tectonically controlled)
7. **Apply smoothing** (progressive, deeper = smoother)
8. **Enforce ordering** (ensure no violations of vertical order)

### 6.2 Key Functions

#### **Geomorphic Zone Classification**
```python
geo_zones = classify_geomorphic_zones(surface_elev, pixel_scale_m)
# Returns: ridges, fan_toes, valleys, platforms, closed_basins
```

#### **Energy-Based Facies Assignment**
```python
# Suppress on ridges
ridge_suppression = np.where(ridges, 0.05, 1.0)

# Boost in appropriate zones
fan_conglomerate_boost = np.where(fan_toes, 3.0, 1.0)
valley_fine_boost = np.where(valleys, 2.5, 1.0)
platform_carbonate_boost = np.where(platforms, 3.0, 1.0)
basin_evaporite_boost = np.where(closed_basins, 5.0, 0.0)
```

#### **Topography-Dependent Regolith**
```python
# Strong suppression on ridges
regolith[ridges] *= 0.15
regolith[steep_mask] *= 0.3

# Enhancement in hollows
regolith[concave_mask & moderate_slope] *= 1.8
```

#### **Smooth Basement Surface**
```python
# Long-wavelength undulation
undul_smooth = fractional_surface(N, beta=2.5, rng=rng)  # Low beta = smooth
undul_smooth = _box_blur(undul_smooth, k=large_window)

# Tectonic flexure
basement = regional_dip + undul_smooth + crustal_flexure
basement = _box_blur(basement, k=15%_of_domain)  # Strong smoothing
```

### 6.3 Validation Checks

After generation, validate:
1. **No negative thicknesses**
2. **Vertical ordering maintained** (top > bottom for all interfaces)
3. **Regional constraints met** (e.g., evaporite only in closed basins)
4. **Smoothness achieved** (no pixel-scale jumps)
5. **Thickness limits respected** (max thicknesses not exceeded)

### 6.4 Parameter Tuning

Key parameters to tune for different geological settings:

- **Steep threshold:** 25-30° (adjust for terrain type)
- **Low slope threshold:** 5-8° (adjust for valley definition)
- **Platform slope threshold:** 2-3° (adjust for carbonate platforms)
- **Suppression factors:** 0.05-0.2x on ridges (adjust for erosion intensity)
- **Boost factors:** 2.5-5.0x in appropriate zones (adjust for deposition intensity)
- **Smoothing strength:** sigma = 1.0-5.0 (adjust for interface smoothness)

---

## 9. Scientific Basis

### 7.1 Energy-Based Deposition

**Principle:** Grain size scales with energy
- **High energy** (steep slopes, high relief) → Erosion and transport, no deposition
- **Moderate energy** (fan toes, moderate slopes) → Coarse deposition (conglomerate, sandstone)
- **Low energy** (valleys, platforms) → Fine deposition (mudstone, shale, carbonates)

### 7.2 Hillslope Processes

**Principle:** Regolith thickness reflects balance between weathering and erosion
- **Steep slopes:** Erosion > Weathering → Thin regolith
- **Gentle slopes:** Weathering > Erosion → Thick regolith
- **Optimal saprolite:** 5-20° slopes (enough time for weathering, not too much erosion)

### 7.3 Facies Distribution

**Principle:** Facies cluster in predictable geomorphic settings
- **Walther's Law:** Vertically adjacent facies were laterally adjacent
- **Sequence Stratigraphy:** Facies distribution follows base level changes
- **Energy Gradients:** Coarse → Fine with decreasing energy

### 7.4 Tectonic Control

**Principle:** Basement geometry reflects long-wavelength tectonic structure
- **Isostasy:** Higher topography → Thicker crust
- **Flexure:** Crustal loading creates long-wavelength flexure
- **Faulting:** Creates blocky steps, not random noise

---

## 10. Summary of Key Rules

### 8.1 Topographic Rules
- **Ridges:** Thin regolith, suppressed sediments, exposed bedrock
- **Fan Toes:** Conglomerate emphasis
- **Valleys:** Fine-grained emphasis, thick alluvium
- **Platforms:** Carbonate emphasis
- **Closed Basins:** Evaporite emphasis

### 8.2 Surface Parameter Rules
- **Elevation:** Higher → thinner regolith, more bedrock
- **Slope:** Steeper → thinner regolith, less deposition
- **Curvature:** Convex → thin, Concave → thick
- **Flow Accumulation:** Higher → thicker alluvium

### 8.3 Layer Relationship Rules
- **Strict vertical ordering:** Never violated
- **Regional constraints:** Layers only where geomorphically appropriate
- **Thickness relationships:** Some layers are fractions of others
- **Smooth interfaces:** Deep layers are smoother than shallow

### 8.4 Implementation Rules
- **Process in order:** Zones → Parameters → Layers → Smoothing → Validation
- **Energy-based assignment:** High energy → coarse, Low energy → fine
- **Strong topography response:** Regolith strongly varies with topography
- **Smooth basement:** Long-wavelength only, no surface noise

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Author:** Geological Layer Generation System
