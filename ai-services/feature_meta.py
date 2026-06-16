"""Human-readable English descriptions for each model feature.

Used to turn raw column names into plain language before sending evidence to the LLM
and in API responses. Unknown features fall back to their raw name. The phrasing is
kept as natural noun phrases so it reads cleanly inside a sentence.
"""

FEATURE_DESCRIPTIONS: dict[str, str] = {
    # Building footprint (Google Open Buildings)
    "b_area_mean": "average building size",
    "b_area_median": "median building size",
    "b_area_std": "variation in building size",
    "b_area_sum": "total built-up floor area",
    "b_count": "number of buildings",
    "b_coverage": "share of land covered by buildings",
    "b_density_km2": "building density per square kilometre",
    "b_size_cv": "uniformity of building sizes",
    "building_fractional_count_mean": "estimated building density",
    "building_fractional_count_stdDev": "variation in estimated building density",
    "building_height_mean": "average building height",
    "building_height_stdDev": "variation in building height",
    "building_presence_mean": "how consistently buildings are present",
    "building_presence_stdDev": "patchiness of building coverage",
    # NDBI built-up + texture (Sentinel-2)
    "ndbi_mean": "built-up intensity",
    "ndbi_stdDev": "variation in built-up intensity",
    "ndbi_contrast_mean": "built-up texture contrast",
    "ndbi_contrast_stdDev": "variation in built-up texture contrast",
    "ndbi_ent_mean": "built-up texture randomness",
    "ndbi_ent_stdDev": "variation in built-up texture randomness",
    "ndbi_var_mean": "built-up texture variability",
    "ndbi_var_stdDev": "variation in built-up texture variability",
    # SAR radar (Sentinel-1)
    "VH_mean": "average VH radar backscatter",
    "VH_stdDev": "variation in VH radar backscatter",
    "VV_mean": "average VV radar backscatter",
    "VV_stdDev": "variation in VV radar backscatter",
    "vv_vh_mean": "average VV/VH radar ratio",
    "vv_vh_stdDev": "variation in VV/VH radar ratio",
    # Optical indices (Sentinel-2)
    "brightness_mean": "average image brightness",
    "brightness_stdDev": "variation in image brightness",
    "bsi_mean": "bare-soil exposure",
    "bsi_stdDev": "variation in bare-soil exposure",
    "ndvi_mean": "vegetation greenness",
    "ndvi_stdDev": "variation in vegetation greenness",
    # Engineered (ratios + interactions)
    "b_area_med_mean_ratio": "dominance of small buildings (median-to-mean size ratio)",
    "b_coverage_per_count": "building coverage per building",
    "density_lowrise": "a dense, low-rise building pattern",
    "coverage_per_area": "how tightly buildings are packed together",
    "ndbi_minus_ndvi": "built-up land dominating over greenery",
    "ndbi_texture_ratio": "texture roughness relative to the built-up level",
    "vh_vv_diff": "surface roughness from the radar signal",
}
