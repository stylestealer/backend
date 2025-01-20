
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
   cd stylestealer/backend
   ```

2. **Install Python dependencies** (for local installations):

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Build and Run with Docker**:

   If deploying on a cloud server with Docker installed, build the container:

   ```bash
   docker build -t stylestealer-backend .
   docker run --gpus all -p 5000:5000 stylestealer-backend
   ```

4. **Run the service**:

   On a machine with sufficient VRAM:

   ```bash
   python app.py
   ```

   The service will be available at `http://0.0.0.0:5000/`.

---

### Endpoints

#### 1) Virtual Try-on (POST `/virtual_tryon`)

**Description**: Transfer a garment onto a person image.

- **URL**: `http://YOUR_SERVER_IP:5000/virtual_tryon`
- **Method**: **POST**
- **Form Fields**: (multipart/form-data)
  | Field              | Required | Default       | Description                                                                      |
  |--------------------|----------|---------------|----------------------------------------------------------------------------------|
  | `src_image`        | yes      | N/A           | Image of the **person** (the subject wearing the new garment).                   |
  | `ref_image`        | yes      | N/A           | Image of the **garment** to transfer.                                           |
  | `ref_acceleration` | no       | "False"       | "True"/"False" – toggles reference U-Net acceleration.                           |
  | `step`             | no       | 50            | Number of diffusion steps (integer).                                             |
  | `scale`            | no       | 2.5           | Guidance scale (float).                                                          |
  | `seed`             | no       | 42            | Random seed (integer). Use `-1` for random.                                      |
  | `vt_model_type`    | no       | "viton_hd"    | `"viton_hd"` or `"dress_code"`.                                                 |
  | `vt_garment_type`  | no       | "upper_body"  | `"upper_body"`, `"lower_body"`, or `"dresses"`.                                 |
  | `vt_repaint`       | no       | "False"       | "True"/"False" – toggles repaint mode.                                           |

- **Response**: PNG image (content-type `image/png`).

---

#### 2) Pose Transfer (POST `/pose_transfer`)

**Description**: Transfer a person’s appearance onto the **pose** of another image.

- **URL**: `http://YOUR_SERVER_IP:5000/pose_transfer`
- **Method**: **POST**
- **Form Fields**: (multipart/form-data)
  | Field              | Required | Default | Description                                                      |
  |--------------------|----------|---------|------------------------------------------------------------------|
  | `src_image`        | yes      | N/A     | The **target pose** image.                                      |
  | `ref_image`        | yes      | N/A     | The **original** person image (whose appearance you are moving). |
  | `ref_acceleration` | no       | "False" | "True"/"False".                                                  |
  | `step`             | no       | 50      | Number of diffusion steps.                                       |
  | `scale`            | no       | 2.5     | Guidance scale.                                                  |
  | `seed`             | no       | 42      | Random seed.                                                     |

- **Response**: PNG image (content-type `image/png`).

---

## Examples

**Virtual Try-on with `curl`**:

```bash
curl -X POST   -F "src_image=@/path/to/person.jpg"   -F "ref_image=@/path/to/garment.jpg"   -F "step=40"   -F "scale=3.0"   http://127.0.0.1:5000/virtual_tryon --output result_vton.png
```

**Pose Transfer with Python**:

```python
import requests

# Pose Transfer
with open("/path/to/pose_target.jpg", "rb") as pose_img, open("/path/to/original_person.jpg", "rb") as ref_img:
    files = {
        "src_image": pose_img,
        "ref_image": ref_img
    }
    data = {
        "step": "30",
        "scale": "2.5"
    }
    response = requests.post("http://127.0.0.1:5000/pose_transfer", files=files, data=data)
    if response.status_code == 200:
        with open("result_pt.png", "wb") as f:
            f.write(response.content)
        print("Saved pose transfer result to result_pt.png")
    else:
        print("Error:", response.text)
```