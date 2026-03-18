🚀 Real-World Problem Solving

Designed an end-to-end system to automatically interpret CAD floor plans

Bridges the gap between architecture, AI, and HVAC engineering

Eliminates manual room measurement and heat load estimation

🧠 AI + Engineering Integration

Integrated GPT-4o Vision for intelligent room classification

Combines computer vision + structured geometry data for better accuracy

Implements fallback-safe architecture for AI-dependent workflows

📐 Advanced Geometry Processing

Built custom room detection algorithms (Mode A & Mode B)

Handles:

Glass partitions

Wall gaps & door bridging

Complex polygon filtering

Uses Shapely + graph-based logic for spatial analysis

⚙️ Scalable System Design

Modular architecture:

geometry_engine.py → detection

layer_classifier.py → intelligent layer parsing

heat_load.py → engineering calculations

Optimized using Streamlit caching + hash-based state management

🏢 Industry-Relevant Application

Implements real HVAC formulas:

TR (Tons of Refrigeration)

CFM (Airflow)

Applicable to:

Commercial buildings

Residential planning

Smart building systems

📊 End-to-End Pipeline

DWG → DXF → Geometry → AI → Engineering Output

File upload → parsing → detection → classification → calculation → export

Outputs:

Annotated floor plan

CSV with room-wise analytics
