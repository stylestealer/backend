
# Stylestealer Backend

This repository contains the **backend inference** service for [Leffa](https://huggingface.co/spaces/franciszzj/Leffa). It exposes a **Flask HTTP API** for:

1. **Virtual Try-on** - Transfer a garment from one image onto a person image.  
2. **Pose Transfer** - Transfer the appearance of one person onto the pose of another image.

⚠️ **Important:** This model requires at least **80 GB VRAM** for inference. Running locally is feasible only on high-end machines with sufficient GPU memory (e.g., NVIDIA A100 GPUs). For most users, deployment on cloud-based solutions (e.g., AWS, Azure, GCP) or dedicated servers is recommended.

---

## Installation

### Prerequisites

- A machine with **80 GB VRAM or more** (e.g., multi-GPU setup or A100 GPUs).  
- Docker (optional but recommended for consistent environment setup).
- Python 3.8+ for local installations.

---

### Server Deployment

1. **Clone this repository**:

   ```bash
   git clone git@github.com:stylestealer/backend.git
   cd backend
   ```

2. **Build and Run**:
   

   ```bash
   pip install -r requirements.txt
   python3 app.py
   ```
