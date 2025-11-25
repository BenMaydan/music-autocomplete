import os
import json
import io
import base64
from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt

# Import our existing scripts
import plot_ablation_grid
# We will need to slightly modify inference.py to be importable without running main
import inference

app = Flask(__name__)

# Configuration
TRAINED_MODELS_DIR = "trained_models"
INFERENCE_OUTPUT_DIR = "inference"

@app.route('/')
def index():
    return render_template('index.html')

# --- Ablation Study Routes ---

@app.route('/ablation')
def ablation_list():
    studies = []
    if os.path.exists(TRAINED_MODELS_DIR):
        for name in os.listdir(TRAINED_MODELS_DIR):
            path = os.path.join(TRAINED_MODELS_DIR, name)
            if os.path.isdir(path):
                studies.append(name)
    return render_template('ablation_list.html', studies=studies)

@app.route('/ablation/<study_name>')
def ablation_view(study_name):
    study_dir = os.path.join(TRAINED_MODELS_DIR, study_name)
    if not os.path.exists(study_dir):
        return "Study not found", 404
    
    # Get all available parameters from the first run found
    params = set()
    for run_name in os.listdir(study_dir):
        run_path = os.path.join(study_dir, run_name)
        if os.path.isdir(run_path):
            history_path = os.path.join(run_path, "loss_history.json")
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        data = json.load(f)
                        if 'ablation_params' in data:
                            params.update(data['ablation_params'].keys())
                            break # Found one, that's enough to know the keys
                except:
                    continue
    
    return render_template('ablation.html', study_name=study_name, params=sorted(list(params)))

@app.route('/ablation/<study_name>/plot')
def ablation_plot(study_name):
    x_param = request.args.get('x_param')
    y_param = request.args.get('y_param')
    
    if not x_param or not y_param:
        return "Missing parameters", 400
        
    study_dir = os.path.join(TRAINED_MODELS_DIR, study_name)
    
    # Use the existing logic from plot_ablation_grid.py
    # We need to refactor plot_ablation_grid.py slightly to return the figure 
    # or we can save to a buffer here if we modify it.
    # For now, let's assume we modify it to return a matplotlib figure.
    
    runs_data = plot_ablation_grid.load_study_data(study_dir, x_param, y_param)
    
    # Create the plot
    fig = plot_ablation_grid.create_grid_figure(runs_data, x_param, y_param)
    
    if fig is None:
        return "Could not create plot (no data?)", 404
        
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return send_file(buf, mimetype='image/png')

# --- Inference Routes ---

@app.route('/inference')
def inference_page():
    # List available runs (potential models)
    # We can look in trained_models. 
    # A valid model run should have a model.pt and config.json
    models = []
    if os.path.exists(TRAINED_MODELS_DIR):
        # We might have nested folders due to ablation studies
        # Let's just walk the directory
        for root, dirs, files in os.walk(TRAINED_MODELS_DIR):
            if 'model.pt' in files and 'config.json' in files:
                # Create a relative path as the ID
                rel_path = os.path.relpath(root, TRAINED_MODELS_DIR)
                models.append(rel_path)
                
    return render_template('inference.html', models=sorted(models))

@app.route('/inference/generate', methods=['POST'])
def inference_generate():
    data = request.json
    model_path = data.get('model_path')
    prompt = data.get('prompt', '60,62,64') # Default C major chord-ish
    max_new_tokens = int(data.get('max_new_tokens', 256))
    temperature = float(data.get('temperature', 1.0))
    
    if not model_path:
        return jsonify({'error': 'No model selected'}), 400
        
    full_run_dir = os.path.join(TRAINED_MODELS_DIR, model_path)
    
    # Prepare arguments for inference
    # We'll mock the argparse namespace or modify inference.py to accept a dict/class
    class Config:
        pass
    
    config = Config()
    config.run_dir = full_run_dir
    config.prompt_tokens = prompt
    config.max_new_tokens = max_new_tokens
    config.temperature = temperature
    config.top_k = 50 # Default
    config.tokenizer_file = "lmd_matched_tokenizer.json" # Default
    
    try:
        # We need to modify inference.py to return the output path instead of just printing
        output_path = inference.run_inference_api(config)
        
        # Return the URL to the generated file
        # The output path is likely in 'inference/<run_name>/...'
        # We need to convert this to a web-accessible URL
        
        # Assuming inference.py saves to 'inference/...' relative to CWD
        # We can serve files from there.
        
        filename = os.path.basename(output_path)
        run_name = os.path.basename(os.path.dirname(output_path))
        
        # We'll create a route to serve these
        file_url = f"/outputs/{run_name}/{filename}"
        
        return jsonify({'success': True, 'file_url': file_url})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<run_name>/<filename>')
def serve_output(run_name, filename):
    directory = os.path.join(INFERENCE_OUTPUT_DIR, run_name)
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
