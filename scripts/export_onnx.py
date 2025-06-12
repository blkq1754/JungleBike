import torch
import onnx
import onnxruntime
import numpy as np
import time
import json
from pathlib import Path

# Importer nos classes et configurations
from src.model import ProductClassifier

CONFIG = {
    "model_path": Path("models/best_model.bin"),
    "target_map_path": Path("models/dataset_target_map.json"),
    "onnx_model_path": Path("models/best_modelONNX.onnx"),
    "max_len": 128,
    "batch_size": 1, 
    "benchmark_iterations": 100
}

def export_to_onnx():
    """Charge le modèle PyTorch et l'exporte au format ONNX."""
    print("--- Démarrage de l'exportation vers ONNX ---")
    
    device = "cpu" 
    
    # le mapping
    with open(CONFIG["target_map_path"], 'r') as f:
        target_map = json.load(f)
    n_classes = len(target_map)
    
    # Initialiser et charger le modèle 
    model = ProductClassifier(n_classes=n_classes)
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=torch.device(device)))
    model.to(device)
    model.eval()

    #une entrée dummy  avec la bonne forme
    dummy_input_ids = torch.randint(0, 1000, (CONFIG["batch_size"], CONFIG["max_len"]), dtype=torch.long)
    dummy_attention_mask = torch.ones(CONFIG["batch_size"], CONFIG["max_len"], dtype=torch.long)
    
    print(f"Exportation du modèle vers {CONFIG['onnx_model_path']}...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        CONFIG["onnx_model_path"],
        input_names=['input_ids', 'attention_mask'], 
        output_names=['logits'], 
        dynamic_axes={ # pour accepter des batchs de tailles différentes
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    # Vérifier que le modèle exporté est valide
    onnx_model = onnx.load(CONFIG["onnx_model_path"])
    onnx.checker.check_model(onnx_model)
    
    print("Exportation réussie et modèle ONNX validé.")

def benchmark(model_type: str):
    """Mesure la performance d'inférence d'un modèle donné."""
    print(f"\n--- Benchmark du modèle {model_type.upper()} ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation de l'appareil : {device.upper()}")

    # Préparer une entrée dummy
    input_ids_np = np.random.randint(0, 1000, (CONFIG["batch_size"], CONFIG["max_len"]), dtype=np.int64)
    attention_mask_np = np.ones((CONFIG["batch_size"], CONFIG["max_len"]), dtype=np.int64)

    # Charger le modèle 
    if model_type == "pytorch":
        with open(CONFIG["target_map_path"], 'r') as f:
            n_classes = len(json.load(f))
        model = ProductClassifier(n_classes=n_classes)
        model.load_state_dict(torch.load(CONFIG["model_path"]))
        model.to(device)
        model.eval()
        
        # Convertion des entrées en tenseurs
        input_ids = torch.from_numpy(input_ids_np).to(device)
        attention_mask = torch.from_numpy(attention_mask_np).to(device)

    elif model_type == "onnx":
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(str(CONFIG["onnx_model_path"]), providers=providers)
        # ONNX Runtime utilise des dictionnaires d'entrées avec des arrays NumPy
        onnx_inputs = {
            'input_ids': input_ids_np,
            'attention_mask': attention_mask_np
        }

    # warm-up pour initialiser les caches
    print("Échauffement...")
    for _ in range(10):
        if model_type == "pytorch":
            with torch.no_grad():
                _ = model(input_ids, attention_mask)
        elif model_type == "onnx":
            _ = session.run(None, onnx_inputs)
    
    # Mesure du temps
    print(f"Lancement de {CONFIG['benchmark_iterations']} itérations de benchmark...")
    start_time = time.perf_counter()
    for _ in range(CONFIG["benchmark_iterations"]):
        if model_type == "pytorch":
            with torch.no_grad():
                _ = model(input_ids, attention_mask)
        elif model_type == "onnx":
            _ = session.run(None, onnx_inputs)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_per_inference = total_time / CONFIG["benchmark_iterations"] * 1000 # en ms
    inferences_per_second = CONFIG["benchmark_iterations"] / total_time

    print(f"Temps moyen par prédiction: {avg_time_per_inference:.4f} ms")
    print(f"Prédictions par seconde: {inferences_per_second:.2f} IPS")
    
    return inferences_per_second

if __name__ == "__main__":
    export_to_onnx()
    
    #benchmarks
    pytorch_ips = benchmark("pytorch")
    onnx_ips = benchmark("onnx")

    print(f"Vitesse PyTorch: {pytorch_ips:.2f} IPS")
    print(f"Vitesse ONNX:    {onnx_ips:.2f} IPS")
