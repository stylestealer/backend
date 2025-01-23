import os
import numpy as np
from io import BytesIO
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from flask import Flask, request, jsonify, send_file

# Download checkpoints
snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")


class LeffaPredictor(object):
    def __init__(self):
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
        assert control_type in [
            "virtual_tryon", "pose_transfer"], "Invalid control type: {}".format(control_type)
        src_image = Image.open(src_image_path)
        ref_image = Image.open(ref_image_path)
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        # Mask
        if control_type == "virtual_tryon":
            src_image = src_image.convert("RGB")
            model_parse, _ = self.parsing(src_image.resize((384, 512)))
            keypoints = self.openpose(src_image.resize((384, 512)))
            if vt_model_type == "viton_hd":
                mask = get_agnostic_mask_hd(
                    model_parse, keypoints, vt_garment_type)
            elif vt_model_type == "dress_code":
                mask = get_agnostic_mask_dc(
                    model_parse, keypoints, vt_garment_type)
            mask = mask.resize((768, 1024))
            # garment_type_hd = "upper" if vt_garment_type in [
            #     "upper_body", "dresses"] else "lower"
            # mask = self.mask_predictor(src_image, garment_type_hd)["mask"]
        elif control_type == "pose_transfer":
            mask = Image.fromarray(np.ones_like(src_image_array) * 255)

        # DensePose
        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                src_image_seg_array = self.densepose_predictor.predict_seg(
                    src_image_array)[:, :, ::-1]
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
            elif vt_model_type == "dress_code":
                src_image_iuv_array = self.densepose_predictor.predict_iuv(
                    src_image_array)
                src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                src_image_seg_array = np.concatenate(
                    [src_image_seg_array] * 3, axis=-1)
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
        elif control_type == "pose_transfer":
            src_image_iuv_array = self.densepose_predictor.predict_iuv(
                src_image_array)[:, :, ::-1]
            src_image_iuv = Image.fromarray(src_image_iuv_array)
            densepose = src_image_iuv

        # Leffa
        transform = LeffaTransform()

        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                inference = self.vt_inference_hd
            elif vt_model_type == "dress_code":
                inference = self.vt_inference_dc
        elif control_type == "pose_transfer":
            inference = self.pt_inference
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,)
        gen_image = output["generated_image"][0]
        # gen_image.save("gen_image.png")
        return np.array(gen_image), np.array(mask), np.array(densepose)

    def leffa_predict_vt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint):
        return self.leffa_predict(src_image_path, ref_image_path, "virtual_tryon", ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint)

    def leffa_predict_pt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed):
        return self.leffa_predict(src_image_path, ref_image_path, "pose_transfer", ref_acceleration, step, scale, seed)


app = Flask(__name__)
leffa_predictor = LeffaPredictor()


@app.route("/")
def health_check():
    return jsonify({"status": "ok", "message": "Leffa inference API running."})


@app.route("/virtual_tryon", methods=["POST"])
def virtual_tryon():
    """
    Endpoint to perform virtual try-on.
    Expects form-data or JSON with:
      - 'src_image' file
      - 'ref_image' file
      - optional parameters: ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint
    """
    # 1) Extract files
    if "src_image" not in request.files or "ref_image" not in request.files:
        return jsonify({"error": "Missing src_image or ref_image in form data"}), 400

    src_image = request.files["src_image"]
    ref_image = request.files["ref_image"]

    # 2) Extract optional parameters
    ref_acceleration = request.form.get("ref_acceleration", "False") == "True"
    step = int(request.form.get("step", 50))
    scale = float(request.form.get("scale", 2.5))
    seed = int(request.form.get("seed", 42))
    vt_model_type = request.form.get("vt_model_type", "viton_hd")
    vt_garment_type = request.form.get("vt_garment_type", "upper_body")
    vt_repaint = request.form.get("vt_repaint", "False") == "True"

    # 3) Run inference
    try:
        gen_image_np, mask, densepose = leffa_predictor.leffa_predict_vt(
            src_image, ref_image,
            ref_acceleration=ref_acceleration,
            step=step,
            scale=scale,
            seed=seed,
            vt_model_type=vt_model_type,
            vt_garment_type=vt_garment_type,
            vt_repaint=vt_repaint
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 4) Convert numpy array to PIL Image
    try:
        gen_image_pil = Image.fromarray(gen_image_np)
    except Exception as e:
        return jsonify({"error": f"Failed to convert generated image: {str(e)}"}), 500

    # 5) Return the generated image (and optionally mask/densepose)
    img_io = BytesIO()
    gen_image_pil.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")


@app.route("/pose_transfer", methods=["POST"])
def pose_transfer():
    """
    Endpoint to perform pose transfer.
    Expects form-data or JSON with:
      - 'src_image' file (the target-pose image)
      - 'ref_image' file (the original person image)
      - optional parameters: ref_acceleration, step, scale, seed
    """
    # 1) Extract files
    if "src_image" not in request.files or "ref_image" not in request.files:
        return jsonify({"error": "Missing src_image or ref_image in form data"}), 400

    src_image = request.files["src_image"]
    ref_image = request.files["ref_image"]

    # 2) Extract optional parameters
    ref_acceleration = request.form.get("ref_acceleration", "False") == "True"
    step = int(request.form.get("step", 50))
    scale = float(request.form.get("scale", 2.5))
    seed = int(request.form.get("seed", 42))

    # 3) Run inference
    try:
        gen_image_np, mask, densepose = leffa_predictor.leffa_predict_pt(
            src_image, ref_image,
            ref_acceleration=ref_acceleration,
            step=step,
            scale=scale,
            seed=seed
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 4) Convert numpy array to PIL Image
    try:
        gen_image_pil = Image.fromarray(gen_image_np)
    except Exception as e:
        return jsonify({"error": f"Failed to convert generated image: {str(e)}"}), 500

    # 5) Return the generated image
    img_io = BytesIO()
    gen_image_pil.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
