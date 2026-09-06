# 📷 PhotoIQ: Photography Skills & Camera Settings Analytics

> **Quantify photographic habits, technical camera execution, and shooting DNA using EXIF metadata and Tonal Signal Analysis.**

PhotoIQ is an end-to-end Machine Learning and Data Science dashboard that analyzes collections of RAW (`.ARW`, `.CR2`, `.CR3`, `.NEF`, `.DNG`) and standard JPEG photos. By extracting camera controls and embedded tonal histograms, PhotoIQ profiles your shooting style, detects technical mistakes (camera shake risk, optical diffraction), and generates an RPG-style **Photographer DNA Card**.

---

## ✨ Key Features

- **⚡ Blazing-Fast Ingestion**: Analyzes RAW files in under **50ms** per image by extracting embedded JPEG previews and headers rather than heavy demosaicing.
- **📊 Exposure Triangle Visualizer**: Interactive 2D and 3D Plotly heatmaps correlating Aperture ($f$-stop), Shutter Speed, ISO, and Focal Length.
- **🎨 Tonal Curves & Clipping Detection**: 256-bin luminance histogram analysis detecting shadow crushing and highlight blowout rates.
- **🤖 Unsupervised Style Clustering (ML)**: Uses normalized K-Means clustering to uncover your natural shooting archetypes (e.g. *Telephoto Daylight Tactician*, *Fast Action / Street*, *Low-Light Bokeh*).
- **🎯 5-Axis Photographer Radar**:
  1. **Dynamic Range Mastery** (Contrast breadth & highlight/shadow preservation)
  2. **Stability & Sharpness** (Reciprocal rule discipline with optical stabilization factors)
  3. **Exposure Intentionality** (Manual mode mastery & deliberate depth-of-field control)
  4. **Lighting Adaptability** (Lighting variety, time-of-day spread, ISO range)
  5. **Optical Sweet-Spot** (Lens peak resolving range vs. optical diffraction softening)
- **🖼️ Interactive Anomaly & Keeper Gallery**: Visual inspection grid filtering flagged photos (diffraction at $f/16+$, handshake risks) and prime keepers.
- **🪪 Pro Photographer DNA Card**: Shareable persona pass summarizing your rank, loadout, signature sweet-spot, and next growth quest.

---

## 🛠️ Project Structure

```text
photography_analytics/
│
├── core/
│   ├── __init__.py
│   ├── extractor.py        # Universal EXIF & preview histogram extraction
│   ├── analyzer.py         # Photographic heuristics, shake risk & keeper scoring
│   └── ml_clustering.py    # K-Means clustering, radar calculation & DNA card
│
├── app.py                  # Full interactive Streamlit web dashboard
├── requirements.txt        # Project dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/photography-analytics.git
cd photography-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📷 Supported File Formats

- **Sony**: `.ARW`, `.SR2`
- **Canon**: `.CR2`, `.CR3`
- **Nikon**: `.NEF`, `.NRW`
- **Fujifilm**: `.RAF`
- **Universal RAW**: `.DNG`
- **Export Formats**: `.JPG`, `.JPEG`, `.TIFF`, `.PNG`

---

## 📄 License
MIT License.
