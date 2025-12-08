# Geological Layer Generation Rules
## Comprehensive Rules for Realistic Stratigraphy Based on Topography and Layer Relationships

---

## Table of Contents
1. [Topographic Region Classification](#topographic-region-classification)
2. [Surface Parameter Rules](#surface-parameter-rules)
3. [Layer-Specific Rules by Region](#layer-specific-rules-by-region)
4. [Layer Relationship Rules](#layer-relationship-rules)
5. [Thickness Constraints](#thickness-constraints)
6. [Implementation Guidelines](#implementation-guidelines)

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

## 2. Surface Parameter Rules

### 2.1 Elevation-Based Rules

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

## 3. Layer-Specific Rules by Region

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

## 4. Layer Relationship Rules

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

## 5. Thickness Constraints

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

## 6. Implementation Guidelines

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

## 7. Scientific Basis

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

## 8. Summary of Key Rules

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
