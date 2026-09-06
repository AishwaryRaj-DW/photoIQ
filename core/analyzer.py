import numpy as np
import pandas as pd

def analyze_photo_heuristics(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates technical photographic decisions for each photo."""
    df = df.copy()

    # 1. Reciprocal Rule (Handshake / Motion Blur Risk)
    # Camera shake happens when the shutter is SLOWER (time is LONGER in seconds) than 1/focal_length
    def check_shake_risk(row):
        ss = row.get("shutter_speed")
        fl = row.get("focal_length")
        lens = str(row.get("lens_model", "")).upper()
        
        if ss is None or fl is None or fl <= 0 or ss <= 0:
            return False
            
        # Intentional tripod/long exposure (> 1 second) is not handheld shake
        if ss >= 1.0:
            return False

        # Lens Stabilization Bonus:
        # Lenses with OSS, IS, VR, or OIS give 3 stops of leeway
        has_stabilization = any(tag in lens for tag in ["OSS", "IS", "VR", "OIS", "STEADY"])
        effective_fl = fl * 1.5  # APS-C 1.5x crop factor
        
        if has_stabilization:
            # 3 stops slower is safe (e.g. 1/90s -> ~1/15s)
            safe_limit_seconds = 1.0 / (effective_fl / 4.0)
        else:
            safe_limit_seconds = 1.0 / effective_fl
            
        # RISK happens when shutter speed takes LONGER than the safe limit
        return bool(ss > safe_limit_seconds)

    df["risk_camera_shake"] = df.apply(check_shake_risk, axis=1)

    # 2. Optical Diffraction Softness Risk (Stopping down past f/16)
    def check_diffraction(row):
        ap = row.get("aperture")
        return bool(ap is not None and ap >= 16.0)

    df["risk_diffraction"] = df.apply(check_diffraction, axis=1)

    # 3. Sub-optimal High ISO in Bright Light
    # e.g., ISO >= 1600 while shutter speed is faster than 1/2000s
    def check_iso_penalty(row):
        iso = row.get("iso")
        ss = row.get("shutter_speed")
        if iso is None or ss is None:
            return False
        return bool(iso >= 1600 and ss <= (1.0 / 2000.0))

    df["risk_unnecessary_iso"] = df.apply(check_iso_penalty, axis=1)

    # 4. Severe Dynamic Range Clipping
    def check_clipping(row):
        s_clip = row.get("shadow_clipped_pct", 0) or 0
        h_clip = row.get("highlight_clipped_pct", 0) or 0
        return bool(s_clip > 5.0 or h_clip > 4.0)

    df["risk_high_clipping"] = df.apply(check_clipping, axis=1)

    # 5. Keeper Candidate Heuristic (Optimal sweet spot + crisp shutter + zero clipping)
    def check_keeper(row):
        if row["risk_camera_shake"] or row["risk_diffraction"] or row["risk_unnecessary_iso"] or row["risk_high_clipping"]:
            return False
        ap = row.get("aperture") or 0
        iso = row.get("iso") or 9999
        rms = row.get("rms_contrast") or 0
        # In lens sweet spot, clean ISO, and good contrast
        return bool(3.5 <= ap <= 10.0 and iso <= 800 and rms >= 30.0)

    df["is_keeper"] = df.apply(check_keeper, axis=1)

    # 6. Human-Readable Diagnostic Tag & Reason
    def get_diagnostic(row):
        reasons = []
        if row["risk_diffraction"]:
            ap = row.get("aperture", 0)
            reasons.append(f"Diffraction softness: f/{ap:.1f} exceeds optical limit")
        if row["risk_camera_shake"]:
            reasons.append(f"Camera shake risk: shutter {row.get('shutter_str')} too slow for {row.get('focal_length')}mm")
        if row["risk_unnecessary_iso"]:
            reasons.append(f"Sub-optimal ISO {int(row.get('iso', 0))} with {row.get('shutter_str')}")
        if row["risk_high_clipping"]:
            reasons.append(f"Clipping: shadows {row.get('shadow_clipped_pct')}% / highlights {row.get('highlight_clipped_pct')}%")
            
        if reasons:
            tag = "🚨 Anomaly"
            reason_str = " • ".join(reasons)
        elif row["is_keeper"]:
            tag = "⭐ Prime Keeper"
            reason_str = f"Optimal sweet-spot (f/{row.get('aperture')}), crisp {row.get('shutter_str')}, clean ISO {int(row.get('iso', 0))}"
        else:
            tag = "Standard Shot"
            reason_str = "Solid technical settings within safe tolerances"
            
        return pd.Series([tag, reason_str], index=["diagnostic_tag", "diagnostic_reason"])

    diag_df = df.apply(get_diagnostic, axis=1)
    df["diagnostic_tag"] = diag_df["diagnostic_tag"]
    df["diagnostic_reason"] = diag_df["diagnostic_reason"]

    return df

def generate_coaching_insights(df: pd.DataFrame) -> list[str]:
    """Generates personalized coaching insights based on portfolio trends."""
    insights = []
    total = len(df)
    if total == 0:
        return ["Not enough photos to generate insights."]

    shake_pct = (df["risk_camera_shake"].sum() / total) * 100
    diffraction_pct = (df["risk_diffraction"].sum() / total) * 100
    iso_penalty_pct = (df["risk_unnecessary_iso"].sum() / total) * 100
    clipped_pct = (df["risk_high_clipping"].sum() / total) * 100

    if shake_pct > 15:
        insights.append(
            f"⚠️ **Handshake Risk in {shake_pct:.1f}% of shots**: Several shots violated the reciprocal rule (shutter speed slower than 1/focal length). Consider raising your minimum auto-ISO shutter speed threshold."
        )
    else:
        insights.append("✅ **Sharpness Discipline**: Excellent shutter speed discipline; minimal camera shake risk detected.")

    if diffraction_pct > 8:
        insights.append(
            f"ℹ️ **Diffraction Softening Detected ({diffraction_pct:.1f}%)**: You frequently shoot above f/16. Modern lenses usually hit peak sharpness between f/5.6 and f/8. Consider backing off to f/8 - f/11 to avoid optical diffraction."
        )

    if iso_penalty_pct > 5:
        insights.append(
            f"💡 **ISO Optimization ({iso_penalty_pct:.1f}%)**: Several shots used high ISOs (≥1600) with ultra-fast shutter speeds (1/2000s+). You can safely reduce ISO for cleaner images by dropping the shutter speed."
        )

    if clipped_pct > 20:
        insights.append(
            f"🎯 **Exposure Warning ({clipped_pct:.1f}% clipped)**: Notable shadow crushing or highlight blowout observed. Try underexposing by -0.3 to -0.7 EV to preserve highlight dynamic range."
        )
    else:
        insights.append("🌟 **Dynamic Range Mastery**: Well-balanced histograms with minimal clipping across your portfolio.")

    return insights