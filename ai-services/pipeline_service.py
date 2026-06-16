"""Live feature pipeline: map point -> kelurahan polygon -> GEE feature extraction.

This MIRRORS `04_extract_features_buildings.py` exactly so that features produced at
inference match the ones the model was trained on (no train/serve skew). The only
difference is that it extracts ONE polygon on demand instead of looping a CSV.

Heavy deps (earthengine-api, geopandas, shapely) are imported here, NOT in app.py,
so the rest of the API still runs even when GEE is not configured.
"""
from __future__ import annotations

import json
from pathlib import Path

import ee
import geopandas as gpd
from shapely.geometry import Point

YEAR_DEFAULT = 2023

# Exact reducers from script 04 -------------------------------------------------
def _build_reducers():
    area_red = (ee.Reducer.count()
                .combine(ee.Reducer.sum(), sharedInputs=True)
                .combine(ee.Reducer.mean(), sharedInputs=True)
                .combine(ee.Reducer.stdDev(), sharedInputs=True)
                .combine(ee.Reducer.median(), sharedInputs=True))
    rast_red = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
    return area_red, rast_red


class PipelineService:
    def __init__(
        self,
        boundaries_path: str | Path,
        gee_project: str,
        year: int = YEAR_DEFAULT,
        cache_dir: str | Path | None = None,
        id_col: str = "unit_id",
        city_col: str = "city",
        area_col: str = "unit_area_m2",
        label_col: str = "slum",
    ):
        self.year = int(year)
        self.s2_start, self.s2_end = f"{self.year - 1}-01-01", f"{self.year}-12-31"
        self.s1_start, self.s1_end = f"{self.year}-01-01", f"{self.year}-12-31"
        self.id_col, self.city_col, self.area_col, self.label_col = (
            id_col, city_col, area_col, label_col)

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.gdf = gpd.read_file(boundaries_path)
        if self.gdf.crs is None:
            self.gdf = self.gdf.set_crs("EPSG:4326")
        else:
            self.gdf = self.gdf.to_crs("EPSG:4326")
        # Compute area only if the boundary file does not already carry it.
        if self.area_col not in self.gdf.columns:
            eq = self.gdf.to_crs("EPSG:6933")  # World Cylindrical Equal Area (meters)
            self.gdf[self.area_col] = eq.geometry.area

        ee.Initialize(project=gee_project)
        self.OB = (ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
                   .filter(ee.Filter.gte("confidence", 0.65)))
        self.AREA_RED, self.RAST_RED = _build_reducers()

    # ---- spatial lookup -------------------------------------------------------
    def find_unit(self, lat: float, lon: float) -> dict | None:
        pt = Point(float(lon), float(lat))  # shapely is (x=lon, y=lat)
        # `covers` includes the boundary, so clicks on a kelurahan edge still resolve.
        hit = self.gdf[self.gdf.covers(pt)]
        if hit.empty:
            return None
        row = hit.iloc[0]
        return {
            "unit_id": None if self.id_col not in row else str(row[self.id_col]),
            "city": None if self.city_col not in row else str(row[self.city_col]),
            "unit_area_m2": float(row[self.area_col]),
            "ground_truth": (int(row[self.label_col])
                             if self.label_col in row and row[self.label_col] is not None
                             and str(row[self.label_col]) != "nan" else None),
            "geometry": row.geometry,
        }

    # ---- GEE composites (identical to script 04) ------------------------------
    def _s2(self, region):
        csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterDate(self.s2_start, self.s2_end).filterBounds(region)
              .linkCollection(csp, ["cs"])
              .map(lambda i: i.updateMask(i.select("cs").gte(0.6))))
        img = s2.median().divide(10000)
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
        ndbi = img.normalizedDifference(["B11", "B8"]).rename("ndbi")
        bsi = img.expression(
            "((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))",
            {"B11": img.select("B11"), "B4": img.select("B4"),
             "B8": img.select("B8"), "B2": img.select("B2")}).rename("bsi")
        bright = img.select(["B2", "B3", "B4"]).reduce(ee.Reducer.mean()).rename("brightness")
        gray = ndbi.add(1).multiply(50).toInt()
        glcm = gray.glcmTexture(size=3).select(["ndbi_contrast", "ndbi_ent", "ndbi_var"])
        return ndvi.addBands([ndbi, bsi, bright, glcm])

    def _s1(self, region):
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterDate(self.s1_start, self.s1_end).filterBounds(region)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
              .select(["VV", "VH"]))
        m = s1.median()
        return m.addBands(m.select("VV").subtract(m.select("VH")).rename("vv_vh"))

    def _temporal(self, region):
        return (ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1")
                .filterBounds(region)
                .filterDate(f"{self.year}-01-01", f"{self.year}-12-31")
                .select(["building_fractional_count", "building_height", "building_presence"])
                .mosaic())

    # ---- extraction -----------------------------------------------------------
    def _extract_raw(self, geom, unit_area_m2: float) -> dict:
        # Match script 04's to_fc(): simplify with the same tolerance before EE
        # extraction, so the extraction footprint is identical to training.
        geom = geom.simplify(0.0002, preserve_topology=True)
        g = ee.Geometry(geom.__geo_interface__, proj="EPSG:4326", geodesic=False)
        region = g.bounds()
        raster = self._s2(region).addBands(self._s1(region)).addBands(self._temporal(region))
        bs = self.OB.filterBounds(g).reduceColumns(self.AREA_RED, ["area_in_meters"])
        rs = raster.reduceRegion(reducer=self.RAST_RED, geometry=g, scale=10,
                                 maxPixels=1e9, bestEffort=True, tileScale=16)
        area = ee.Number(unit_area_m2)
        n = ee.Number(bs.get("count"))
        stats = ee.Dictionary({
            "b_count": n,
            "b_area_sum": bs.get("sum"),
            "b_area_mean": bs.get("mean"),
            "b_area_std": bs.get("stdDev"),
            "b_area_median": bs.get("median"),
            "b_coverage": ee.Number(bs.get("sum")).divide(area),
            "b_density_km2": n.divide(area).multiply(1e6),
        })
        out = ee.Dictionary(rs).combine(stats, overwrite=True).getInfo()
        return self._postprocess(out)

    @staticmethod
    def _postprocess(out: dict) -> dict:
        # Mirror script 04: fillna(0) for these, then b_size_cv ratio.
        for c in ["b_count", "b_area_sum", "b_coverage", "b_density_km2"]:
            if out.get(c) is None:
                out[c] = 0.0
        mean, std = out.get("b_area_mean"), out.get("b_area_std")
        out["b_size_cv"] = (std / mean) if (mean not in (None, 0)) and std is not None else None
        return out

    # ---- public: point -> raw feature dict ------------------------------------
    def features_for_point(self, lat: float, lon: float) -> dict | None:
        unit = self.find_unit(lat, lon)
        if unit is None:
            return None

        cache_key = f"{unit['city']}__{unit['unit_id']}"
        cache_file = (self.cache_dir / f"{cache_key}.json") if self.cache_dir else None
        if cache_file and cache_file.exists():
            features = json.loads(cache_file.read_text(encoding="utf-8"))
            source = "cache"
        else:
            features = self._extract_raw(unit["geometry"], unit["unit_area_m2"])
            source = "gee"
            if cache_file:
                cache_file.write_text(json.dumps(features), encoding="utf-8")

        return {
            "unit_id": unit["unit_id"],
            "city": unit["city"],
            "unit_area_m2": unit["unit_area_m2"],
            "ground_truth": unit["ground_truth"],
            "geometry": unit["geometry"].__geo_interface__,
            "feature_source": source,
            "features": features,
        }

    # ---- public: arbitrary point (radius buffer) -> raw feature dict ----------
    def features_for_buffer(self, lat: float, lon: float, radius_m: float = 500.0) -> dict:
        """Extract features for a circular buffer around an ARBITRARY point.

        Used when the click is outside every known kelurahan, so detection no
        longer depends on the training areas. NOTE: the model was trained on
        kelurahan-shaped footprints, so a fixed radius adds some train/serve skew
        — treat these results as indicative rather than authoritative.
        """
        pt = gpd.GeoSeries([Point(float(lon), float(lat))], crs="EPSG:4326")
        buf_m = pt.to_crs("EPSG:6933").buffer(float(radius_m))  # meters, equal-area
        area = float(buf_m.area.iloc[0])
        geom = buf_m.to_crs("EPSG:4326").iloc[0]
        features = self._extract_raw(geom, area)
        return {
            "unit_id": None,
            "city": None,
            "unit_area_m2": area,
            "ground_truth": None,
            "geometry": geom.__geo_interface__,
            "feature_source": "gee_radius",
            "features": features,
        }

    # ---- boundaries for the map frontend --------------------------------------
    def boundaries_geojson(self, city: str | None = None, simplify: float = 0.0002) -> dict:
        gdf = self.gdf if city is None else self.gdf[self.gdf[self.city_col] == city]
        cols = [c for c in (self.id_col, self.city_col) if c in gdf.columns]
        out = gdf[cols + ["geometry"]].copy()
        out["geometry"] = out.geometry.simplify(simplify, preserve_topology=True)
        return json.loads(out.to_json())
