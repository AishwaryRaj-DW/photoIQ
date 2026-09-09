import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.extractor import process_single_image, get_thumbnail_image, RAW_EXTENSIONS, IMAGE_EXTENSIONS
from core.analyzer import analyze_photo_heuristics, generate_coaching_insights
from core.ml_clustering import (
    cluster_shooting_styles, 
    compute_photographer_radar, 
    generate_photographer_card_profile
)
from core.card_generator import render_photographer_card_image

st.set_page_config(page_title="PhotoIQ | Photography Analytics & ML Profiler", layout="wide", page_icon="📷")

# Header
st.title("📷 PhotoIQ: Photography Skills & Settings Profiler")
st.markdown("Quantify your photographic habits, camera execution, and shooting DNA using EXIF & Tonal Signal Analysis.")

# Mode Selector
mode = st.sidebar.radio("Data Ingestion Mode", ["Scan Local Folder", "Upload Files (Public Demo)"])
df = None

if mode == "Scan Local Folder":
    st.sidebar.subheader("Local Folder Path")
    default_path = r"C:\SNAPS\PHOTOWALK\Photowalks\Feb 26, 2026"
    folder_path = st.sidebar.text_input("Enter directory containing RAW/JPEG photos:", value=default_path)
    
    # Filter & Deduplication Controls
    st.sidebar.subheader("File Filters")
    format_filter = st.sidebar.selectbox(
        "Formats to include:",
        ["RAW only (.ARW, .CR2, etc.)", "All Images (RAW + JPEG)", "JPEGs only"]
    )
    dedup_pairs = st.sidebar.checkbox(
        "Auto-deduplicate RAW+JPEG pairs", 
        value=True,
        help="If both DSC00032.ARW and DSC00032.JPG exist, only the RAW file is analyzed so shots aren't counted twice."
    )
    
    scan_btn = st.sidebar.button("🚀 Analyze Photos", type="primary")
    if scan_btn or "cached_df" in st.session_state:
        if scan_btn:
            if not os.path.isdir(folder_path):
                st.error(f"Directory not found: {folder_path}")
            else:
                files_to_process = []
                seen_basenames = set()
                for root, _, files in os.walk(folder_path):
                    # Sort so RAW files come first (.ARW before .JPG)
                    sorted_files = sorted(
                        files, 
                        key=lambda x: (0 if os.path.splitext(x)[1].lower() in RAW_EXTENSIONS else 1)
                    )
                    for f in sorted_files:
                        ext = os.path.splitext(f)[1].lower()
                        base_name = os.path.splitext(f)[0].lower()
                        
                        # Apply Format Filter
                        if format_filter == "RAW only (.ARW, .CR2, etc.)" and ext not in RAW_EXTENSIONS:
                            continue
                        elif format_filter == "JPEGs only" and ext not in {".jpg", ".jpeg"}:
                            continue
                        elif ext not in IMAGE_EXTENSIONS:
                            continue
                            
                        # Deduplicate RAW + JPEG pairs
                        if dedup_pairs:
                            if base_name in seen_basenames:
                                continue
                            seen_basenames.add(base_name)
                            
                        files_to_process.append(os.path.join(root, f))

                if not files_to_process:
                    st.warning("No supported RAW or JPEG images found in this folder.")
                else:
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    records = []
                    
                    total = len(files_to_process)
                    for idx, fpath in enumerate(files_to_process):
                        records.append(process_single_image(fpath, fpath))
                        if (idx + 1) % 10 == 0 or idx == total - 1:
                            pct = (idx + 1) / total
                            progress_bar.progress(pct)
                            status_text.text(f"Processed {idx + 1}/{total} photos...")

                    status_text.empty()
                    progress_bar.empty()
                    
                    raw_df = pd.DataFrame(records)
                    analyzed_df = analyze_photo_heuristics(raw_df)
                    clustered_df, clusters = cluster_shooting_styles(analyzed_df)
                    st.session_state["cached_df"] = clustered_df
                    st.session_state["cached_clusters"] = clusters

        if "cached_df" in st.session_state:
            df = st.session_state["cached_df"]
            clusters = st.session_state["cached_clusters"]

else:
    st.sidebar.subheader("Upload Photos")
    uploaded_files = st.sidebar.file_uploader(
        "Upload RAW (.ARW, .CR2, .NEF, .DNG) or JPEGs", 
        accept_multiple_files=True,
        type=[ext.lstrip(".") for ext in IMAGE_EXTENSIONS]
    )
    if uploaded_files:
        with st.spinner(f"Analyzing {len(uploaded_files)} photos..."):
            records = [process_single_image(f, f.name) for f in uploaded_files]
            raw_df = pd.DataFrame(records)
            analyzed_df = analyze_photo_heuristics(raw_df)
            df, clusters = cluster_shooting_styles(analyzed_df)

# Render Dashboard if data is loaded
if df is not None and len(df) > 0:
    # Top KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Photos", len(df))
    col2.metric("Median Aperture", f"f/{df['aperture'].median():.1f}" if df['aperture'].notnull().any() else "N/A")
    col3.metric("Median ISO", int(df['iso'].median()) if df['iso'].notnull().any() else "N/A")
    shake_pct = (df["risk_camera_shake"].sum() / len(df)) * 100
    col4.metric("Shake Risk Rate", f"{shake_pct:.1f}%")
    dominant_lens = df["lens_model"].value_counts().index[0] if df["lens_model"].value_counts().any() else "Unknown"
    col5.metric("Primary Lens", dominant_lens[:18] + "..." if len(dominant_lens) > 18 else dominant_lens)

    st.divider()

    # Tabs for in-depth analytics
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Exposure Triangle & Optics", 
        "🎨 Tonal Curves & Clipping", 
        "🤖 ML Shooting Archetypes", 
        "🎯 Photographer DNA / Radar",
        "🖼️ Anomaly & Keeper Gallery",
        "🪪 Pro Photographer Card",
        "💡 Coaching Insights"
    ])

    with tab1:
        st.subheader("Exposure Triangle Distribution")
        c1, c2 = st.columns(2)
        
        with c1:
            fig_iso = px.histogram(df.dropna(subset=["iso"]), x="iso", nbins=30, title="ISO Distribution", log_y=True, color_discrete_sequence=["#636EFA"])
            st.plotly_chart(fig_iso, use_container_width=True)
            
        with c2:
            fig_ap = px.histogram(df.dropna(subset=["aperture"]), x="aperture", nbins=25, title="Aperture (f-stop) Usage", color_discrete_sequence=["#EF553B"])
            st.plotly_chart(fig_ap, use_container_width=True)

        st.subheader("Aperture vs. Shutter Speed (Exposure Heatmap)")
        valid_scatter = df.dropna(subset=["aperture", "shutter_speed", "iso"]).copy()
        if len(valid_scatter) > 0:
            valid_scatter["log_shutter"] = np.log2(valid_scatter["shutter_speed"])
            fig_scatter = px.scatter(
                valid_scatter,
                x="aperture",
                y="log_shutter",
                color="iso",
                size="focal_length",
                hover_data=["filename", "shutter_str", "lens_model"],
                title="Aperture vs. Shutter Speed (Sized by Focal Length, Colored by ISO)",
                labels={"log_shutter": "Log2 Shutter Speed (EV Steps)", "aperture": "Aperture (f-number)"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Light Metering & Optical Field of View")
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            if "metering_mode" in df and df["metering_mode"].notnull().any():
                fig_metering = px.pie(df, names="metering_mode", title="Light Metering Mode Distribution", hole=0.45,
                                      color_discrete_sequence=px.colors.sequential.Teal)
                st.plotly_chart(fig_metering, use_container_width=True)
        with m_c2:
            if "focal_length_35mm" in df and df["focal_length_35mm"].notnull().any():
                fig_fl = px.histogram(df.dropna(subset=["focal_length_35mm"]), x="focal_length_35mm", nbins=25,
                                      title="35mm Equivalent Focal Length (Field of View)",
                                      color_discrete_sequence=["#00CC96"])
                st.plotly_chart(fig_fl, use_container_width=True)

    with tab2:
        st.subheader("Tonal Curve & Dynamic Range Utilization")
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            fig_clip = px.scatter(
                df,
                x="shadow_clipped_pct",
                y="highlight_clipped_pct",
                color="risk_high_clipping",
                hover_data=["filename"],
                title="Shadow Loss vs. Highlight Blowout (%)",
                labels={"shadow_clipped_pct": "Shadows Clipped (%)", "highlight_clipped_pct": "Highlights Blown (%)"}
            )
            st.plotly_chart(fig_clip, use_container_width=True)

        with col_c2:
            st.markdown("#### Portfolio Average Luminance Curve")
            all_hists = [h for h in df["luminance_hist"] if h is not None]
            if all_hists:
                avg_hist = np.mean(all_hists, axis=0)
                curve_df = pd.DataFrame({"Luminance (0=Black, 255=White)": range(256), "Pixel Frequency": avg_hist})
                fig_hist = px.line(curve_df, x="Luminance (0=Black, 255=White)", y="Pixel Frequency", title="Composite Portfolio Tone Curve")
                st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("Unsupervised Style Clustering (ML Archetypes)")
        st.markdown("An unsupervised K-Means algorithm grouped your photos based on your settings decisions:")
        
        for c in clusters:
            with st.expander(f"📌 {c['name']} ({c['pct']}% of your shots - {c['count']} photos)"):
                st.write(f"- **Median Aperture**: {c['median_aperture']}")
                st.write(f"- **Median Shutter Speed**: {c['median_shutter']}")
                st.write(f"- **Median ISO**: {c['median_iso']}")
                st.write(f"- **Median Focal Length**: {c['median_focal']}")

        fig_cluster = px.scatter(
            df[df["cluster"] >= 0],
            x="aperture",
            y="focal_length",
            color="archetype_name",
            size="iso",
            hover_data=["filename", "shutter_str"],
            title="Photographic Archetype Clusters"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    with tab4:
        st.subheader("Photographer Skill & Intent Radar")
        st.caption("Calibrated against camera sensor physics, optical diffraction thresholds, and exposure discipline.")
        radar_scores = compute_photographer_radar(df)
        
        categories = list(radar_scores.keys())
        values = list(radar_scores.values())
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Your Profile',
            line_color='#00CC96'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Stat Breakdown Table
        stat_cols = st.columns(5)
        for i, (k, v) in enumerate(radar_scores.items()):
            stat_cols[i].metric(k, f"{v}/100")

    with tab5:
        st.subheader("🖼️ Interactive Anomaly & Keeper Gallery")
        st.markdown("Inspect photos flagged for technical attention or recognized as prime technical keepers.")
        
        gallery_filter = st.radio(
            "Filter Gallery View:",
            ["🚨 Flagged Anomalies", "⭐ Prime Keepers", "Diffraction Risk (f/16+)", "Handshake Risk", "All Photos"],
            horizontal=True
        )
        
        # Filter dataframe based on selection
        if gallery_filter == "🚨 Flagged Anomalies":
            display_df = df[df["diagnostic_tag"] == "🚨 Anomaly"]
        elif gallery_filter == "⭐ Prime Keepers":
            display_df = df[df["diagnostic_tag"] == "⭐ Prime Keeper"]
        elif gallery_filter == "Diffraction Risk (f/16+)":
            display_df = df[df["risk_diffraction"] == True]
        elif gallery_filter == "Handshake Risk":
            display_df = df[df["risk_camera_shake"] == True]
        else:
            display_df = df

        st.caption(f"Showing {len(display_df)} photos matching criteria")
        
        if len(display_df) == 0:
            st.info("No photos found in this category.")
        else:
            # Pagination for fast snappy rendering
            page_size = 9
            total_pages = max(1, (len(display_df) + page_size - 1) // page_size)
            page = 1
            if total_pages > 1:
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
            
            start_idx = (page - 1) * page_size
            page_slice = display_df.iloc[start_idx : start_idx + page_size]
            
            # 3-column photo grid
            grid_cols = st.columns(3)
            for idx, (_, row) in enumerate(page_slice.iterrows()):
                col = grid_cols[idx % 3]
                with col:
                    fpath = row.get("filepath")
                    img_thumb = None
                    if fpath:
                        img_thumb = get_thumbnail_image(fpath)
                    
                    if img_thumb:
                        st.image(img_thumb, use_container_width=True)
                    else:
                        st.text("Preview unavailable")
                        
                    tag = row.get("diagnostic_tag", "")
                    if "Anomaly" in tag:
                        badge_color = "red"
                    elif "Keeper" in tag:
                        badge_color = "green"
                    else:
                        badge_color = "blue"
                        
                    st.markdown(f"**`{row['filename']}`** :{badge_color}[{tag}]")
                    fl_eq_str = f" ({int(row['focal_length_35mm'])}mm eq.)" if pd.notnull(row.get("focal_length_35mm")) else ""
                    st.caption(f"⚙️ **f/{row.get('aperture')}** • **{row.get('shutter_str')}** • **ISO {int(row.get('iso', 0))}** • **{row.get('focal_length')}mm{fl_eq_str}**")
                    st.caption(f"🎯 Metering: **{row.get('metering_mode', 'Auto')}** • WB: **{row.get('white_balance', 'Auto')}** • Flash: **{row.get('flash_status', 'Off')}**")
                    st.info(row.get("diagnostic_reason", "Standard shot"))
                    st.divider()

    with tab6:
        st.subheader("🪪 Pro Photographer DNA Card")
        st.markdown("Your official, shareable photographer persona pass computed from your EXIF signal data.")
        
        radar_scores = compute_photographer_radar(df)
        card = generate_photographer_card_profile(df, radar_scores)
        
        col_card_preview, col_card_actions = st.columns([1.6, 1.4])
        
        with st.spinner("Generating high-resolution card..."):
            card_img_bytes = render_photographer_card_image(card, radar_scores)
            
        with col_card_preview:
            st.image(card_img_bytes, caption="Official Photographer ID Pass (1280x1760 HD)", use_container_width=True)
            
        with col_card_actions:
            st.markdown("### 📥 Export Your Pass")
            st.markdown("Download your personalized **Photographer ID Card** as a high-resolution PNG, formatted for Instagram, Twitter/X, or your portfolio.")
            
            safe_title = card["title"].replace(" ", "_").replace("/", "_")
            st.download_button(
                label="💾 Download High-Res Card (PNG)",
                data=card_img_bytes,
                file_name=f"PhotoIQ_Pass_{safe_title}.png",
                mime="image/png",
                type="primary",
                use_container_width=True
            )
            
            st.divider()
            st.markdown("#### 📋 Profile Breakdown")
            st.markdown(f"- **Archetype Class:** **{card['title']}**")
            st.markdown(f"- **Rank:** **Level {card['overall_level']} Specialist**")
            st.markdown(f"- **Primary Camera:** `{card['camera']}`")
            st.markdown(f"- **Primary Lens:** `{card['lens']}`")
            st.markdown(f"- **Signature Sweet-Spot:** `{card['sweet_spot']}`")
            st.markdown(f"- **★ Superpower:** :green[**{card['top_stat']}**]")
            st.markdown(f"- **🎯 Growth Target:** :red[**{card['lowest_stat']}**]")
            st.info(f"**Next Quest:** {card['growth_quest']}")

    with tab7:
        st.subheader("Tailored Coaching Insights")
        insights = generate_coaching_insights(df)
        for ins in insights:
            st.markdown(ins)

        st.subheader("Photos Needing Review (Potential Anomaly Flags)")
        flagged_df = df[df["risk_camera_shake"] | df["risk_diffraction"] | df["risk_unnecessary_iso"] | df["risk_high_clipping"]]
        st.dataframe(
            flagged_df[["filename", "aperture", "shutter_str", "iso", "focal_length", "risk_camera_shake", "risk_diffraction", "risk_unnecessary_iso", "risk_high_clipping", "diagnostic_reason"]],
            use_container_width=True
        )
else:
    st.info("👈 Select a folder on the left or upload photos to start profiling!")