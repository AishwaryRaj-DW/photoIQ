import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_shooting_styles(df: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, list[dict]]:
    """
    Clusters photos into shooting archetypes using normalized exposure parameters.
    """
    df = df.copy()
    feature_cols = ["aperture", "shutter_speed", "iso", "focal_length"]
    
    # Filter valid rows
    valid_mask = df[feature_cols].notnull().all(axis=1) & (df["shutter_speed"] > 0) & (df["iso"] > 0)
    if valid_mask.sum() < n_clusters:
        df["cluster"] = 0
        df["archetype_name"] = "General Photography"
        return df, []

    sub_df = df[valid_mask].copy()
    
    # Feature transformations (log2 scale for shutter speed and ISO mirrors photography stops)
    X = np.column_stack([
        sub_df["aperture"].values,
        np.log2(sub_df["shutter_speed"].values),
        np.log2(sub_df["iso"].values),
        sub_df["focal_length"].values,
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = min(n_clusters, len(sub_df))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    sub_df["cluster"] = cluster_labels

    # Describe each cluster based on centroid properties
    cluster_summaries = []
    cluster_names = {}
    
    for c_id in range(k):
        c_rows = sub_df[sub_df["cluster"] == c_id]
        med_ap = c_rows["aperture"].median()
        med_ss = c_rows["shutter_speed"].median()
        med_iso = c_rows["iso"].median()
        med_fl = c_rows["focal_length"].median()

        # Semantic naming heuristic based on photographic archetypes
        if med_ap <= 2.8 and med_iso >= 800:
            name = "Low-Light / Bokeh Portrait"
        elif med_ss <= (1.0 / 500.0):
            name = "Fast Action / Street Snapshot"
        elif med_ap >= 8.0:
            name = "Deep Depth of Field / Landscape"
        elif med_fl >= 85:
            name = "Telephoto Focus"
        elif med_iso <= 200 and med_ss >= 0.1:
            name = "Long Exposure / Tripod"
        else:
            name = f"Style Group {c_id + 1}"

        cluster_names[c_id] = name
        cluster_summaries.append({
            "cluster_id": c_id,
            "name": name,
            "count": len(c_rows),
            "pct": round(len(c_rows) / len(sub_df) * 100, 1),
            "median_aperture": f"f/{med_ap:.1f}",
            "median_shutter": f"{med_ss:.4f}s",
            "median_iso": int(med_iso),
            "median_focal": f"{med_fl:.0f}mm"
        })

    sub_df["archetype_name"] = sub_df["cluster"].map(cluster_names)
    df["cluster"] = -1
    df["archetype_name"] = "Uncategorized"
    df.loc[valid_mask, "cluster"] = sub_df["cluster"]
    df.loc[valid_mask, "archetype_name"] = sub_df["archetype_name"]

    return df, cluster_summaries

def compute_photographer_radar(df: pd.DataFrame) -> dict[str, float]:
    """
    Computes a scientifically grounded 5-dimensional skill/intent radar score (0 - 100):
    1. Dynamic Range Mastery (Contrast breadth & highlight/shadow preservation)
    2. Stability & Sharpness Discipline (Reciprocal rule & handshake safety margin)
    3. Exposure Intentionality (Manual mode mastery, deliberate aperture, setting adaptation)
    4. Lighting Adaptability (Diversity of lighting conditions, hours, and ISO range)
    5. Optical Sweet-Spot Discipline (Using lens in peak resolving range vs diffraction)
    """
    if len(df) == 0:
        return {k: 50.0 for k in ["Dynamic Range", "Stability", "Intentionality", "Lighting Adaptability", "Optical Sweet-Spot"]}

    total = len(df)

    # 1. Dynamic Range Mastery:
    # Requires genuine contrast breadth (RMS contrast) WITHOUT clipping.
    # Flat grey scenes are NOT high dynamic range.
    avg_contrast = df["rms_contrast"].mean() if "rms_contrast" in df and not df["rms_contrast"].isnull().all() else 40.0
    contrast_factor = min(1.0, max(0.3, avg_contrast / 55.0))  # Scales with tonal spread
    clipping_avg = df[["shadow_clipped_pct", "highlight_clipped_pct"]].mean().sum()
    dr_score = (contrast_factor * 50.0) + max(0.0, 50.0 - (clipping_avg * 3.5))
    dr_score = max(25.0, min(95.0, dr_score))

    # 2. Stability: 
    # Starts at 95; deducts realistically for shake risk rate
    shake_pct = (df["risk_camera_shake"].sum() / total) * 100
    stability_score = max(25.0, min(96.0, 96.0 - (shake_pct * 2.8)))

    # 3. Intentionality (Calibrated):
    # - Manual exposure mode vs auto
    # - Deliberate aperture choice (wide-open for subject isolation or f/8-f/11 for landscapes)
    # - Adapting settings across the shoot
    manual_pct = (df.get("exposure_program", pd.Series(dtype=str)).str.contains("Manual", case=False, na=False) |
                  df.get("exposure_mode", pd.Series(dtype=str)).str.contains("Manual", case=False, na=False)).sum() / total * 100
    semi_manual_pct = df.get("exposure_program", pd.Series(dtype=str)).str.contains("Priority", case=False, na=False).sum() / total * 100
    
    deliberate_aperture = ((df["aperture"] <= 5.6) | ((df["aperture"] >= 8.0) & (df["aperture"] <= 11.0))).sum() / total * 100
    
    # Check if settings were varied or locked to a single fixed value
    shutter_unique = df["shutter_speed"].dropna().nunique()
    aperture_unique = df["aperture"].dropna().nunique()
    adaptation_score = min(20.0, (shutter_unique * 2.5) + (aperture_unique * 1.5))
    
    mode_points = (manual_pct * 0.45) + (semi_manual_pct * 0.35)
    aperture_points = deliberate_aperture * 0.25
    intent_score = 15.0 + mode_points + aperture_points + adaptation_score
    intent_score = max(30.0, min(94.0, intent_score))

    # 4. Lighting Adaptability (Grounded):
    # Evaluates whether the shoot spanned multiple lighting scenarios or just one afternoon walk
    iso_unique = df["iso"].dropna().nunique()
    hour_unique = df["hour_of_day"].dropna().nunique()
    iso_spread = (df["iso"].max() - df["iso"].min()) if df["iso"].notnull().any() else 0
    
    # A single afternoon session gets an honest ~50-65 score; shooting day + night reaches 85+
    lighting_score = 35.0 + (min(6, hour_unique) * 4.5) + (min(8, iso_unique) * 3.0) + (min(15.0, (iso_spread / 1600.0) * 15.0))
    lighting_score = max(30.0, min(92.0, lighting_score))

    # 5. Optical Sweet-Spot Discipline:
    # Sony 18-135mm resolves peak sharpness at f/5.6 - f/8.
    # Stopping down past f/16 causes severe diffraction softening.
    sweet_spot_pct = ((df["aperture"] >= 3.5) & (df["aperture"] <= 9.0)).sum() / total * 100
    diffraction_pct = (df["aperture"] >= 16.0).sum() / total * 100
    
    optical_score = (sweet_spot_pct * 0.82) - (diffraction_pct * 3.0) + 12.0
    optical_score = max(25.0, min(95.0, optical_score))

    return {
        "Dynamic Range": round(dr_score, 1),
        "Stability": round(stability_score, 1),
        "Intentionality": round(intent_score, 1),
        "Lighting Adaptability": round(lighting_score, 1),
        "Optical Sweet-Spot": round(optical_score, 1),
    }

def generate_photographer_card_profile(df: pd.DataFrame, radar_scores: dict[str, float]) -> dict:
    """
    Generates an RPG-style Photographer ID card profile with titles, loadout,
    signature sweet-spot, and quest/growth areas.
    """
    total = len(df)
    primary_lens = df["lens_model"].value_counts().index[0] if df["lens_model"].value_counts().any() else "Prime Lens"
    primary_cam = df["camera_model"].value_counts().index[0] if df["camera_model"].value_counts().any() else "Mirrorless Camera"
    
    # Archetype determination
    med_fl = df["focal_length"].median() if df["focal_length"].notnull().any() else 50
    med_ss = df["shutter_speed"].median() if df["shutter_speed"].notnull().any() else 0.004
    med_ap = df["aperture"].median() if df["aperture"].notnull().any() else 5.6
    med_iso = int(df["iso"].median()) if df["iso"].notnull().any() else 100
    
    if med_fl >= 70:
        title = "Telephoto Daylight Tactician"
    elif med_ss <= (1.0 / 500.0):
        title = "High-Speed Action Specialist"
    elif med_ap <= 2.8:
        title = "Shallow Depth-of-Field Sculptor"
    elif med_ap >= 8.0:
        title = "Deep-Focus Landscape Chaser"
    else:
        title = "Natural Light Street Scout"

    # Overall Skill Index (Weighted average of radar)
    overall_level = int(np.mean(list(radar_scores.values())))
    
    # Find strongest and growth dimensions
    sorted_stats = sorted(radar_scores.items(), key=lambda x: x[1], reverse=True)
    top_stat = sorted_stats[0]
    lowest_stat = sorted_stats[-1]

    # Growth Quest Recommendation
    quest_map = {
        "Lighting Adaptability": "Expand into Dusk & Indoor: Challenge yourself with Blue Hour and low-light sessions to diversify your exposure adaptability.",
        "Intentionality": "Dynamic Control: Try varying your shutter speed deliberately to contrast frozen action with motion blur.",
        "Dynamic Range": "High-Contrast Challenge: Experiment with backlighting or silhouettes to test exposure compensation.",
        "Stability": "Shutter Discipline: Raise auto-ISO minimum shutter speed threshold when shooting at telephoto ends.",
        "Optical Sweet-Spot": "Avoid Diffraction: Keep apertures at f/5.6 - f/11 for peak resolving power; avoid stopping down past f/16."
    }

    return {
        "title": title,
        "overall_level": overall_level,
        "camera": primary_cam,
        "lens": primary_lens,
        "sweet_spot": f"f/{med_ap:.1f} • {df['shutter_str'].mode()[0] if not df['shutter_str'].mode().empty else '1/400s'} • ISO {med_iso} • {med_fl:.0f}mm",
        "top_stat": f"{top_stat[0]} ({top_stat[1]}/100)",
        "lowest_stat": f"{lowest_stat[0]} ({lowest_stat[1]}/100)",
        "growth_quest": quest_map.get(lowest_stat[0], "Keep exploring new focal lengths and lighting conditions!")
    }