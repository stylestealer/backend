import os
import requests
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# ------------------ Download/Upload Helpers ------------------ #

def download_file(url, local_path):
    """
    Downloads a file from `url` and saves it to `local_path`.
    """
    print(f"Downloading: {url} -> {local_path}")
    r = requests.get(url)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(r.content)

def upload_via_put(local_path, upload_url):
    """
    Uploads the local file to `upload_url` (a one-time or pre-signed PUT URL).
    """
    print(f"Uploading {local_path} to {upload_url} via PUT...")
    headers = {"Content-Type": "application/octet-stream"}
    with open(local_path, "rb") as f:
        response = requests.put(upload_url, data=f, headers=headers)
    response.raise_for_status()
    print(f"Upload success (status code {response.status_code}).")

def terminate_vm():
    """
    Optionally terminate a VM via Cudo's API, if environment variables are set:
      TERMINATE_URL   e.g. https://rest.compute.cudo.org/v1/projects/PROJECT_ID/vms/VM_ID/terminate
      TERMINATE_TOKEN e.g. your bearer token
    """
    terminate_url = os.environ.get("TERMINATE_URL")
    token = os.environ.get("TERMINATE_TOKEN")
    if not terminate_url or not token:
        print("Skipping VM termination (TERMINATE_URL or TERMINATE_TOKEN not set).")
        return False

    print(f"Terminating VM via {terminate_url}...")
    headers = {
        "Authorization": f"bearer {token}",
        "Accept": "application/json"
    }
    try:
        resp = requests.post(terminate_url, headers=headers)
        resp.raise_for_status()
        print("VM termination initiated.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to terminate VM: {e}")
        return False

# ------------------ LeffaPredictor Class ------------------ #

class LeffaPredictor:
    def __init__(self):
        # Download checkpoints if not already present
        snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")

        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

        vt_model_dc = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon_dc.pth",
            dtype="float16",
        )
        self.vt_inference_dc = LeffaInference(model=vt_model_dc)

        pt_model = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
            pretrained_model="./ckpts/pose_transfer.pth",
            dtype="float16",
        )
        self.pt_inference = LeffaInference(model=pt_model)

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=False,
        step=50,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False
    ):
        # Basic validations
        assert control_type in ["virtual_tryon", "pose_transfer"], \
            f"Invalid control_type: {control_type}"
        
        src_image = Image.open(src_image_path).convert("RGB")
        ref_image = Image.open(ref_image_path).convert("RGB")

        # Resize to 768x1024
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)
        src_image_array = np.array(src_image)

        # Build mask
        if control_type == "virtual_tryon":
            model_parse, _ = self.parsing(src_image.resize((384, 512)))
            keypoints = self.openpose(src_image.resize((384, 512)))

            if vt_model_type == "viton_hd":
                mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
            else:  # "dress_code"
                mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)

            mask = mask.resize((768, 1024))
            # Densepose
            if vt_model_type == "viton_hd":
                seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
                densepose = Image.fromarray(seg_array)
            else:  # "dress_code"
                iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
                seg_array = iuv_array[:, :, 0:1]
                seg_array = np.concatenate([seg_array] * 3, axis=-1)
                densepose = Image.fromarray(seg_array)
            
            # Decide which inference model
            inference = self.vt_inference_hd if vt_model_type == "viton_hd" else self.vt_inference_dc

        else:  # control_type == "pose_transfer"
            # Full white mask
            mask = Image.fromarray(np.ones_like(src_image_array) * 255)
            iuv_array = self.densepose_predictor.predict_iuv(src_image_array)[:, :, ::-1]
            densepose = Image.fromarray(iuv_array)
            inference = self.pt_inference

        # Transform input
        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose]
        }
        data = transform(data)

        # Run inference
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint
        )
        # output["generated_image"] is a list of PIL images
        gen_image = output["generated_image"][0]
        return gen_image, mask, densepose

# ------------------ Main Script ------------------ #

def main():
    """
    1. Reads environment vars:
       - PROCESS_TYPE = "dress" or "pose"
         => "virtual_tryon" or "pose_transfer"
       - IMAGE_URL_1 (the user/person image)
       - IMAGE_URL_2 (the garment or pose reference)
       - UPLOAD_URL   (pre-signed PUT link for the result)
       - (Optional) steps, scale, seed, garment type, repaint, etc.
    2. Downloads images.
    3. Runs the appropriate Leffa function.
    4. Saves final image to local file.
    5. PUT uploads the result to UPLOAD_URL.
    6. Optionally calls terminate_vm().
    """

    # ----- Read environment -----
    process_type = os.environ.get("PROCESS_TYPE", "dress")  # "dress" or "pose"
    image_url_1 = os.environ.get("IMAGE_URL_1")
    image_url_2 = os.environ.get("IMAGE_URL_2")
    upload_url  = os.environ.get("UPLOAD_URL")

    if not image_url_1 or not image_url_2 or not upload_url:
        print("Error: Must set IMAGE_URL_1, IMAGE_URL_2, and UPLOAD_URL environment variables.")
        return
    
    # Additional optional params
    steps_str = os.environ.get("STEPS", "30")
    scale_str = os.environ.get("SCALE", "2.5")
    seed_str  = os.environ.get("SEED", "42")

    # Garment or model type only relevant if process_type == "dress"
    vt_garment_type = os.environ.get("VT_GARMENT_TYPE", "upper_body")  # or "lower_body", "dresses"
    vt_model_type   = os.environ.get("VT_MODEL_TYPE", "viton_hd")      # "viton_hd" or "dress_code"
    vt_repaint_str  = os.environ.get("VT_REPAINT", "False")            # True/False

    # Convert them to correct data types
    step_count = int(steps_str)
    scale_val  = float(scale_str)
    seed_val   = int(seed_str)
    vt_repaint = (vt_repaint_str.lower() == "true")

    # Download the images
    local_path_1 = "./src_image.jpg"
    local_path_2 = "./ref_image.jpg"
    download_file(image_url_1, local_path_1)
    download_file(image_url_2, local_path_2)

    # Initialize Leffa
    predictor = LeffaPredictor()

    # Decide which control_type
    if process_type.lower() == "dress":
        control_type = "virtual_tryon"
        # We'll pass garment type and model type
    else:
        control_type = "pose_transfer"
        # vt_garment_type and vt_model_type won't be used in pose_transfer

    print(f"Running Leffa with control_type={control_type}, steps={step_count}, scale={scale_val}, seed={seed_val}")

    # Run the model
    gen_image, mask, densepose = predictor.leffa_predict(
        src_image_path=local_path_1,
        ref_image_path=local_path_2,
        control_type=control_type,
        step=step_count,
        scale=scale_val,
        seed=seed_val,
        vt_model_type=vt_model_type,
        vt_garment_type=vt_garment_type,
        vt_repaint=vt_repaint
    )

    # Save final image
    output_path = "./output.jpg"
    gen_image.save(output_path)
    print(f"Saved output to {output_path}")

    # Upload result
    upload_via_put(output_path, upload_url)

    # Terminate VM
    terminate_vm()

    print("Done!")

if __name__ == "__main__":
    main()
