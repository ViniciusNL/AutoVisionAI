# 🚗 AutoVision AI — Intelligent Vehicle Inspection Dashboard

**AutoVision AI** is an end-to-end prototype for **automated vehicle inspection using computer vision**. The system processes vehicle images and transforms visual information into structured insurance and fleet-inspection decisions:

> **Image Quality → Vehicle/Plate Analysis → Damage Detection → Severity Assessment → Cost Estimation → Fraud Detection → Inspection Report → Decision**

The project combines a **Python computer vision engine** with an interactive, self-contained web dashboard for visualizing inspection results, KPIs, detected damages, fraud indicators, and automated decisions.

---

## 🎯 Project Overview

Vehicle inspection is a critical process for insurance companies, fleet operators, and automotive services. AutoVision AI demonstrates how a computer vision pipeline can automate several stages of this workflow.

For each vehicle, the system can:

* Analyze image quality and authenticity
* Identify potential license plate regions
* Detect visual anomalies associated with vehicle damage
* Estimate damage severity
* Estimate repair costs
* Identify potential image manipulation and fraud indicators
* Generate a structured inspection report
* Automatically recommend an inspection decision

The current implementation is designed as a **functional proof of concept**, using classical computer vision techniques rather than GPU-based deep learning models.

---

## 🧠 Computer Vision Pipeline

The main pipeline is implemented by:

```python
vehicle_inspection_ai.run_inspection()
```

The system performs the following steps:

### 1. 📷 Image Acquisition

Loads and processes one or more vehicle images provided for inspection.

### 2. 🔍 Image Quality & Authenticity

Evaluates image characteristics such as:

* Sharpness
* Brightness
* Resolution
* Laplacian variance

These metrics help identify images that may be unsuitable for reliable inspection.

### 3. 🚘 Vehicle & License Plate Analysis

Uses contour-based heuristics to locate regions that may correspond to a vehicle's license plate.

> This component is intentionally heuristic and serves as a placeholder for a production ANPR/LPR solution.

### 4. 🛠️ Damage Detection

The engine analyzes the image using:

* Grid-based scanning
* Edge-density analysis
* Texture anomalies
* Local visual irregularities

These regions are used to identify potential:

* Scratches
* Dents
* Surface damage
* Other visual anomalies

### 5. 📊 Severity Assessment

A severity score from **0–100** is calculated based on factors such as:

* Maximum damage score across images
* Number of detected damage regions
* Diversity of damage categories

### 6. 💰 Cost Estimation

Estimated repair costs are calculated using a configurable base-cost table combined with the estimated damage severity.

### 7. 🕵️ Fraud Detection

The system analyzes potential signs of image manipulation through:

* **Error Level Analysis (ELA)**
* EXIF metadata inspection
* Detection of image-editing software metadata
* Missing or suspicious metadata

The resulting indicators contribute to the overall fraud assessment.

### 8. 📄 Inspection Report

All results are consolidated into a structured **JSON inspection report**, containing the information required by the dashboard and downstream applications.

### 9. ⚖️ Automated Decision

Based on configurable thresholds for:

* Image quality
* Damage severity
* Fraud indicators

the system generates one of the following recommendations:

* ✅ **Automatic Approval**
* 🔎 **Manual Review**
* 🚨 **Suspected Fraud**

---

## 🖥️ Dashboard

The project includes a fully self-contained dashboard built with:

* HTML
* CSS
* JavaScript

No backend server is required to visualize the current demonstration.

Simply open:

```text
dashboard.html
```

in any modern web browser.

The dashboard provides:

* Inspection KPIs
* Recent inspection cases
* Damage statistics
* Severity trends
* Fraud indicators
* Vehicle inspection reports
* Step-by-step analysis pipeline
* Visual vehicle damage diagrams

Clicking any case in **"Casos Recentes"** opens its complete inspection report.

---

## 📂 Project Structure

| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `dashboard.html`             | Self-contained interactive dashboard                                               |
| `vehicle_inspection_ai.py`   | Core computer vision inspection engine                                             |
| `generate_demo_data.py`      | Generates synthetic inspection images and processes them through the real pipeline |
| `build_dashboard_payload.py` | Aggregates inspection results into dashboard KPIs, trends, and tables              |
| `dashboard_data.json`        | Aggregated dashboard data embedded into the HTML                                   |
| `data/`                      | Generated inspection data and synthetic images                                     |

---

## ⚙️ Running the Project

### Requirements

* Python 3.9+
* OpenCV
* Pillow
* NumPy

Install the dependencies:

```bash
pip install opencv-python pillow numpy
```

> Depending on your Python environment, you may need additional permissions or a virtual environment.

### Run the Computer Vision Engine

Process one or more vehicle images:

```bash
python3 vehicle_inspection_ai.py foto1.jpg foto2.jpg foto3.jpg
```

The inspection report will be printed as structured JSON in the terminal.

---

## 🔄 Regenerating the Dashboard Data

To generate a new dataset and rebuild the dashboard payload:

```bash
pip install piexif
```

Then:

```bash
python3 generate_demo_data.py
python3 build_dashboard_payload.py
```

This generates:

```text
data/inspections_raw.json
dashboard_data.json
```

The resulting `dashboard_data.json` can then be embedded into `dashboard.html` through the:

```html
<script id="dashboard-data">
```

section.

---

## 🧪 Synthetic Demonstration Dataset

The current dashboard demonstration uses **synthetically generated vehicle inspection images**.

Instead of manually creating arbitrary dashboard numbers, the project generates synthetic images containing visual patterns representing:

* Scratches
* Dents
* Surface anomalies
* Image manipulation
* Simulated editing metadata

These images are then processed by the **same computer vision pipeline used for individual inspections**.

The current demonstration processes **140 synthetic images**.

The business-level information, such as:

* License plate
* Insurance company
* Inspection date
* Vehicle model

is assigned separately for demonstration purposes.

This approach keeps the dashboard metrics connected to actual outputs from the inspection engine while avoiding the use of real customer or vehicle data.

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │   Vehicle Images    │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │   Image Quality Analysis │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Vehicle / Plate Analysis │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    Damage Detection      │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Severity Assessment    │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    Cost Estimation       │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Fraud Analysis       │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    JSON Inspection       │
                 │         Report           │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Automated Decision       │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │       Dashboard          │
                 └──────────────────────────┘
```

---

## 🚀 Production Evolution

The current project intentionally uses **classical computer vision**, making it lightweight and executable without a GPU or trained model.

For a production-grade system, the architecture can evolve without changing the overall pipeline.

Potential upgrades include:

### Damage Detection

Replace heuristic detection with a trained deep learning detector such as:

* YOLO
* Faster R-CNN
* Mask R-CNN

trained on a labeled vehicle-damage dataset.

### License Plate Recognition

Replace contour-based detection with a dedicated:

* ANPR/LPR detector
* OCR pipeline
* License plate recognition model

### Damage Classification

Introduce specialized models to classify:

* Scratches
* Dents
* Cracks
* Broken components
* Paint damage

### Fraud Detection

Develop a machine learning classifier trained using historical insurance claims and manipulated-image examples.

### Cost Estimation

Replace static cost tables with a predictive model incorporating:

* Damage type
* Severity
* Vehicle model
* Vehicle year
* Parts
* Labor
* Historical repair costs

### Deployment

The pipeline can also be extended into a production architecture with:

```text
API
 ↓
Image Storage
 ↓
Computer Vision Inference
 ↓
Fraud Detection
 ↓
Cost Prediction
 ↓
Decision Engine
 ↓
Database
 ↓
Web Dashboard
```

---

## ⚠️ Current Limitations

This repository is a **proof of concept**, not a production insurance-inspection system.

The current damage detectors rely on classical computer vision heuristics and **are not trained machine learning models**.

Therefore, the system should not be interpreted as having production-level accuracy or reliability.

The main purpose of the project is to demonstrate:

* Computer vision pipeline design
* Image analysis
* Feature extraction
* Automated decision logic
* Fraud-analysis concepts
* Structured AI/ML system architecture
* Data aggregation
* Interactive visualization

The architecture was intentionally designed so individual components can later be replaced by trained models without redesigning the complete pipeline.

---

## 🛠️ Technologies

**Programming**

* Python
* JavaScript
* HTML5
* CSS3

**Computer Vision**

* OpenCV
* Pillow
* NumPy
* Laplacian-based sharpness analysis
* Edge detection
* Texture analysis
* Error Level Analysis (ELA)
* EXIF metadata analysis

**Data & Visualization**

* JSON
* Interactive HTML dashboard
* Synthetic data generation

---

## 🤖 Development & AI Assistance

This project was developed with assistance from **Claude Code**, Anthropic's AI-powered coding agent, which was used as a development and engineering assistance tool during implementation and iteration.

The use of Claude Code does **not** imply that the project is owned, endorsed, sponsored, or officially licensed by Anthropic.

For information about Claude Code and its applicable terms, refer to the official Anthropic documentation and terms.

---

## 📜 License

This project is distributed under the license specified in the repository's `LICENSE` file.

**Claude Code is a product of Anthropic and is subject to Anthropic's applicable terms and policies.** Its use in the development of this project does not transfer ownership of Claude Code or Anthropic's intellectual property to this repository.

---

## 👨‍💻 Author

**Vinícius Nunes Leal**

Physicist · Data Scientist · Research Scientist · Machine Learning · Computer Vision

---

⭐ If you found this project interesting, consider giving the repository a star.

