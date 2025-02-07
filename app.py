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
from google.cloud import storage

# ------------------ Download/Upload Helpers ------------------ #

def download_file(url, local_path):
    """
    Downloads a file from `url` and saves it to `local_path`.
    """  
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.download_to_filename(local_path)
    print(f"Downloaded gs://{bucket_name}/{blob_name} to {local_path}")

def upload_file(local_path, bucket_name, blob_name=None):
    """
    Uploads the local file to the specified Cloud Storage bucket.

    Parameters:
      local_path (str): Path to the local file.
      bucket_name (str): Name of the destination bucket.
      blob_name (str): Name of the blob in the bucket.
                       If not provided, defaults to the file's basename.
    """
    if blob_name is None:
        blob_name = os.path.basename(local_path)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path, content_type="application/octet-stream")
    print(f"Upload success: '{local_path}' has been uploaded to bucket '{bucket_name}' as '{blob_name}'.")

def terminate_vm():
    api_key = os.environ.get("API_KEY")
    vm_name = os.environ.get("VM_NAME")
    get_url = os.environ.get("API_URL")
    if not api_key or not vm_name or not get_url:
        print("Skipping VM termination (API_KEY or VM_NAME or API_URL not set).")
        return False
    
    headers = {
        "accept": "application/json",
        "api_key": api_key
    }

    print(f"Retrieving instances from {get_url}...")
    try:
        # Step 1: Retrieve instances to find the VM by name
        get_response = requests.get(get_url, headers=headers)
        get_response.raise_for_status()

        data = get_response.json()
        if not data.get("status"):
            print(f"GET request failed: {data.get('message', 'No error message provided.')}")
            return False

        instances = data.get("instances", [])
        if not instances:
            print("No instances found in the response.")
            return False

        # Find the instance with the matching name
        target_instance = None
        for instance in instances:
            if instance.get("name") == vm_name:
                target_instance = instance
                break

        if not target_instance:
            print(f"No instance found with name '{vm_name}'.")
            return False

        instance_id = target_instance.get("id")
        if instance_id is None:
            print("Instance ID not found in the instance data.")
            return False

        # Step 2: Delete the identified instance
        delete_url = f"{get_url}/{instance_id}"
        print(f"Terminating VM with ID {instance_id} via {delete_url}...")

        delete_response = requests.delete(delete_url, headers=headers)
        delete_response.raise_for_status()

        delete_data = delete_response.json()
        if not delete_data.get("status"):
            print(f"DELETE request failed: {delete_data.get('message', 'No error message provided.')}")
            return False

        print("VM termination successful.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Failed to terminate VM: {e}")
        return False

# ------------------ LeffaPredictor Class ------------------ #

class LeffaPredictor:
    def __init__(self):        
        # Download checkpoints
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
    try:
        # ----- Read environment -----        
        process_type = os.environ.get("PROCESS_TYPE", "dress")  # "dress" or "pose"
        image_url_1 = os.environ.get("IMAGE_URL_1")
        image_url_2 = os.environ.get("IMAGE_URL_2")
        bucket_name = os.environ.get("BUCKET_NAME")
        blob_name  = os.environ.get("BLOB_NAME")

        if not image_url_1 or not image_url_2 or not bucket_name:
            print("Error: Must set IMAGE_URL_1, IMAGE_URL_2, and BUCKET_NAME environment variables.")
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
        local_path_1 = "/tmp/src_image.jpg"
        local_path_2 = "/tmp/ref_image.jpg"
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
        upload_file(output_path, bucket_name, blob_name)

        print("Done!")

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        print("Attempting to terminate VM...")
        try:
            # Terminate VM
            if not terminate_vm():
                print("Failed to terminate the VM. Manual intervention is required. ")
        except Exception as e:
            print(f"Error during VM termination: {str(e)}")



if __name__ == "__main__":
    main()