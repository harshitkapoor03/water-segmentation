import io
import os
import base64
import logging
import tempfile
import yaml
from flask import Flask, request, jsonify, send_file, render_template_string
from predictor import WaterSegmentationPredictor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

logger.info("Loading model at startup...")
predictor = WaterSegmentationPredictor(config)
logger.info("Model ready.")

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Water Segmentation</title>
<style>
  body { font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
  h1 { font-size: 22px; margin-bottom: 24px; }
  input[type=file] { display: block; margin-bottom: 12px; }
  button { padding: 10px 24px; background: #1a8fff; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 24px; }
  .panel { background: white; border-radius: 8px; padding: 12px; border: 1px solid #ddd; }
  .panel p { font-size: 13px; color: #666; margin-bottom: 8px; }
  .panel img { width: 100%; border-radius: 4px; display: block; }
  #status { margin-top: 12px; font-size: 14px; color: #666; }
  #error { margin-top: 12px; color: red; font-size: 14px; }
</style>
</head>
<body>
<h1>Water Body Segmentation</h1>
<input type="file" id="fileInput" accept="image/*">
<button id="btn" onclick="run()" disabled>Run Segmentation</button>
<div id="status"></div>
<div id="error"></div>
<div class="grid" id="grid" style="display:none">
  <div class="panel"><p>Input Image</p><img id="inputImg"></div>
  <div class="panel"><p>Water Mask</p><img id="maskImg"></div>
</div>

<script>
const fileInput = document.getElementById('fileInput');
const btn = document.getElementById('btn');

fileInput.addEventListener('change', () => {
  btn.disabled = !fileInput.files.length;
  if (fileInput.files.length) {
    const reader = new FileReader();
    reader.onload = e => document.getElementById('inputImg').src = e.target.result;
    reader.readAsDataURL(fileInput.files[0]);
  }
});

async function run() {
  btn.disabled = true;
  document.getElementById('status').textContent = 'Running...';
  document.getElementById('error').textContent = '';
  document.getElementById('grid').style.display = 'none';

  const form = new FormData();
  form.append('image', fileInput.files[0]);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: form });
    if (!resp.ok) throw new Error('Prediction failed — check server logs');
    const blob = await resp.blob();
    document.getElementById('maskImg').src = URL.createObjectURL(blob);
    document.getElementById('grid').style.display = 'grid';
    document.getElementById('status').textContent = 'Done.';
  } catch (e) {
    document.getElementById('error').textContent = e.message;
    document.getElementById('status').textContent = '';
  }
  btn.disabled = false;
}
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    file = request.files["image"]
    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)

    try:
        from PIL import Image

        mask = predictor.predict(temp_path)
        img = Image.fromarray(mask)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


