# app.py - Fixed Flask Application for macOS
from flask import Flask, render_template, request, jsonify, send_file
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pennylane as qml
from pennylane import numpy as pnp
import io
import base64
import warnings
import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Configure multiprocessing for macOS
if sys.platform == 'darwin':  # macOS
    mp.set_start_method('spawn', force=True)

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'quantum-drug-discovery-2024'

# Global variables to store trained model
trained_qdd = None
training_status = {"status": "not_started", "progress": 0, "message": "Ready to start"}
training_lock = threading.Lock()

class MolecularDataGenerator:
    """Generate synthetic molecular data for drug discovery simulation"""
    
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        
    def generate_molecular_features(self):
        # Use numpy's random state for reproducibility
        rng = np.random.RandomState(42)
        
        mol_weight = rng.normal(350, 100, self.n_samples)
        mol_weight = np.clip(mol_weight, 150, 800)
        
        logp = rng.normal(2.5, 1.5, self.n_samples)
        logp = np.clip(logp, -2, 6)
        
        psa = rng.normal(60, 30, self.n_samples)
        psa = np.clip(psa, 20, 140)
        
        h_donors = rng.poisson(1.5, self.n_samples)
        h_donors = np.clip(h_donors, 0, 5)
        
        h_acceptors = rng.poisson(3, self.n_samples)
        h_acceptors = np.clip(h_acceptors, 0, 10)
        
        rot_bonds = rng.poisson(4, self.n_samples)
        rot_bonds = np.clip(rot_bonds, 0, 10)
        
        aromatic_rings = rng.poisson(1, self.n_samples)
        aromatic_rings = np.clip(aromatic_rings, 0, 4)
        
        dipole_moment = rng.exponential(2, self.n_samples)
        electronegativity = rng.normal(2.5, 0.5, self.n_samples)
        
        features = np.column_stack([
            mol_weight, logp, psa, h_donors, h_acceptors, 
            rot_bonds, aromatic_rings, dipole_moment, electronegativity
        ])
        
        return features
    
    def generate_drug_labels(self, features):
        mol_weight = features[:, 0]
        logp = features[:, 1]
        psa = features[:, 2]
        h_donors = features[:, 3]
        h_acceptors = features[:, 4]
        
        lipinski_score = (
            (mol_weight <= 500).astype(int) +
            (logp <= 5).astype(int) +
            (h_donors <= 5).astype(int) +
            (h_acceptors <= 10).astype(int)
        )
        
        psa_score = (psa <= 140).astype(int)
        
        # Use numpy's random state
        rng = np.random.RandomState(42)
        total_score = lipinski_score + psa_score + rng.normal(0, 0.5, len(features))
        probabilities = 1 / (1 + np.exp(-(total_score - 3)))
        labels = rng.binomial(1, probabilities)
        
        return labels
    
    def create_dataset(self):
        features = self.generate_molecular_features()
        labels = self.generate_drug_labels(features)
        
        feature_names = [
            'molecular_weight', 'logp', 'polar_surface_area', 
            'h_donors', 'h_acceptors', 'rotatable_bonds', 
            'aromatic_rings', 'dipole_moment', 'electronegativity'
        ]
        
        df = pd.DataFrame(features, columns=feature_names)
        df['is_effective'] = labels
        
        return df, features, labels

class QuantumNeuralNetwork:
    """Quantum Neural Network for molecular classification"""
    
    def __init__(self, n_features=4, n_layers=2, n_qubits=None):
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_qubits = n_qubits or n_features
        
        # Use default.qubit with fewer shots for stability
        self.dev = qml.device('default.qubit', wires=self.n_qubits, shots=None)
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface="autograd")
        self.params = self.initialize_parameters()
    
    def initialize_parameters(self):
        # Use PennyLane's random number generator
        pnp.random.seed(42)
        n_params_per_layer = self.n_qubits * 3
        total_params = self.n_features + (self.n_layers * n_params_per_layer)
        return pnp.random.normal(0, 0.1, total_params, requires_grad=True)
    
    def data_encoding(self, x, encoding_param):
        """Encode classical data into quantum states"""
        for i in range(min(len(x), self.n_qubits)):
            qml.RY(x[i] * encoding_param, wires=i)
    
    def variational_layer(self, params):
        """Single variational layer of the quantum neural network"""
        for i in range(self.n_qubits):
            qml.RX(params[i * 3], wires=i)
            qml.RY(params[i * 3 + 1], wires=i)
            qml.RZ(params[i * 3 + 2], wires=i)
        
        # Entangling gates
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # Circular entanglement for better connectivity
        if self.n_qubits > 2:
            qml.CNOT(wires=[self.n_qubits - 1, 0])
    
    def quantum_circuit(self, x, params):
        """Complete quantum circuit"""
        # Data encoding with first parameter as scaling factor
        self.data_encoding(x, params[0])
        
        # Variational layers
        param_idx = 1  # Skip the encoding parameter
        for layer in range(self.n_layers):
            layer_params = params[param_idx:param_idx + self.n_qubits * 3]
            self.variational_layer(layer_params)
            param_idx += self.n_qubits * 3
        
        # Measurement
        return qml.expval(qml.PauliZ(0))
    
    def predict_single(self, x):
        """Predict for a single sample"""
        try:
            return float(self.qnode(x, self.params))
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0
    
    def predict(self, X):
        """Predict for multiple samples"""
        predictions = []
        for x in X:
            pred = self.predict_single(x)
            predictions.append(pred)
        return pnp.array(predictions)
    
    def cost_function(self, params, X, y):
        """Cost function for training"""
        try:
            predictions = []
            for x in X:
                pred = self.quantum_circuit(x, params)
                predictions.append(pred)
            
            predictions = pnp.array(predictions)
            # Convert predictions to probabilities [0,1]
            predictions = (predictions + 1) / 2
            
            # Binary cross-entropy loss with numerical stability
            epsilon = 1e-7
            predictions = pnp.clip(predictions, epsilon, 1 - epsilon)
            loss = -pnp.mean(y * pnp.log(predictions) + (1 - y) * pnp.log(1 - predictions))
            
            return loss
        except Exception as e:
            print(f"Cost function error: {e}")
            return 1.0  # Return high cost on error

class QuantumDrugDiscovery:
    """Main class for quantum machine learning drug discovery"""
    
    def __init__(self, n_samples=500):
        self.n_samples = min(n_samples, 800)  # Limit for stability
        self.data_generator = MolecularDataGenerator(self.n_samples)
        self.scaler = StandardScaler()
        self.costs = []
        self.accuracy = 0
        self.qnn = None
        
    def prepare_data(self):
        """Prepare and preprocess molecular data"""
        try:
            self.df, features, labels = self.data_generator.create_dataset()
            
            # Select most important features for quantum processing
            feature_importance = ['molecular_weight', 'logp', 'polar_surface_area', 'h_donors']
            X = self.df[feature_importance].values
            y = labels
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            return True
        except Exception as e:
            print(f"Data preparation error: {e}")
            return False
    
    def train_quantum_model(self, n_epochs=50, learning_rate=0.1):
        """Train the quantum neural network"""
        global training_status
        
        try:
            with training_lock:
                training_status["status"] = "training"
                training_status["progress"] = 0
                training_status["message"] = "Initializing quantum neural network..."
            
            # Initialize QNN with smaller configuration for stability
            self.qnn = QuantumNeuralNetwork(n_features=4, n_layers=2, n_qubits=4)
            
            # Use Adam optimizer with smaller learning rate
            optimizer = qml.AdamOptimizer(stepsize=min(learning_rate, 0.1))
            
            self.costs = []
            batch_size = min(20, len(self.X_train))  # Smaller batch size
            
            for epoch in range(min(n_epochs, 80)):  # Limit epochs for stability
                try:
                    with training_lock:
                        training_status["progress"] = int((epoch / n_epochs) * 100)
                        training_status["message"] = f"Training epoch {epoch+1}/{n_epochs}"
                    
                    # Random mini-batch
                    rng = np.random.RandomState(epoch)
                    batch_indices = rng.choice(len(self.X_train), batch_size, replace=False)
                    X_batch = self.X_train[batch_indices]
                    y_batch = self.y_train[batch_indices]
                    
                    # Update parameters
                    self.qnn.params, cost = optimizer.step_and_cost(
                        lambda params: self.qnn.cost_function(params, X_batch, y_batch),
                        self.qnn.params
                    )
                    
                    self.costs.append(float(cost))
                    
                    # Early stopping if cost becomes too high
                    if cost > 10 or np.isnan(cost):
                        print(f"Training stopped early at epoch {epoch} due to high cost: {cost}")
                        break
                        
                except Exception as e:
                    print(f"Training epoch {epoch} error: {e}")
                    continue
            
            with training_lock:
                training_status["status"] = "completed"
                training_status["progress"] = 100
                training_status["message"] = "Training completed successfully!"
            
            return self.costs
            
        except Exception as e:
            with training_lock:
                training_status["status"] = "error"
                training_status["message"] = f"Training failed: {str(e)}"
            print(f"Training error: {e}")
            return []
    
    def evaluate_model(self):
        """Evaluate the quantum model performance"""
        try:
            if self.qnn is None:
                return 0.5, [], []
                
            predictions_raw = self.qnn.predict(self.X_test)
            predictions = (predictions_raw + 1) / 2
            predictions_binary = (predictions > 0.5).astype(int)
            
            self.accuracy = accuracy_score(self.y_test, predictions_binary)
            
            return self.accuracy, predictions, predictions_binary
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.5, [], []
    
    def predict_new_molecule(self, molecular_properties):
        """Predict effectiveness of a new molecule"""
        try:
            if self.qnn is None:
                return 0.5, False
                
            properties_scaled = self.scaler.transform([molecular_properties])
            prediction_raw = self.qnn.predict_single(properties_scaled[0])
            probability = (prediction_raw + 1) / 2
            
            # Ensure probability is in valid range
            probability = max(0.0, min(1.0, probability))
            is_effective = probability > 0.5
            
            return float(probability), bool(is_effective)
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5, False
    
    def generate_plots(self):
        """Generate plots and return as base64 encoded strings"""
        plots = {}
        
        try:
            # Training cost plot
            plt.figure(figsize=(10, 6))
            if self.costs:
                plt.plot(self.costs, 'b-', linewidth=2)
                plt.title('Quantum Neural Network Training Progress', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch')
                plt.ylabel('Cost (Loss)')
                plt.grid(True, alpha=0.3)
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
                img_buffer.seek(0)
                plots['training'] = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()
            
            # Confusion matrix
            if self.qnn is not None:
                predictions_raw = self.qnn.predict(self.X_test)
                predictions = (predictions_raw + 1) / 2
                predictions_binary = (predictions > 0.5).astype(int)
                
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(self.y_test, predictions_binary)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Non-Effective', 'Effective'],
                           yticklabels=['Non-Effective', 'Effective'])
                plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
                img_buffer.seek(0)
                plots['confusion'] = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()
                
        except Exception as e:
            print(f"Plot generation error: {e}")
        
        return plots

# Flask Routes with better error handling
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    global trained_qdd, training_status
    
    try:
        # Reset training status
        with training_lock:
            training_status = {"status": "starting", "progress": 0, "message": "Initializing..."}
        
        # Get parameters from request with validation
        try:
            data = request.get_json()
            if data is None:
                raise ValueError("No JSON data received")
                
            n_samples = min(max(int(data.get('n_samples', 500)), 100), 800)
            n_epochs = min(max(int(data.get('n_epochs', 50)), 10), 80)
            learning_rate = min(max(float(data.get('learning_rate', 0.1)), 0.01), 0.3)
        except (ValueError, TypeError) as e:
            return jsonify({'success': False, 'error': f'Invalid parameters: {str(e)}'})
        
        # Initialize and train model
        trained_qdd = QuantumDrugDiscovery(n_samples=n_samples)
        
        if not trained_qdd.prepare_data():
            return jsonify({'success': False, 'error': 'Failed to prepare data'})
        
        costs = trained_qdd.train_quantum_model(n_epochs=n_epochs, learning_rate=learning_rate)
        
        if not costs:
            return jsonify({'success': False, 'error': 'Training failed - no costs recorded'})
        
        accuracy, _, _ = trained_qdd.evaluate_model()
        
        return jsonify({
            'success': True,
            'accuracy': float(accuracy),
            'final_cost': float(costs[-1]) if costs else 0,
            'message': f'Model trained successfully! Accuracy: {accuracy:.1%}'
        })
        
    except Exception as e:
        with training_lock:
            training_status["status"] = "error"
            training_status["message"] = f"Training failed: {str(e)}"
        return jsonify({'success': False, 'error': str(e)})

@app.route('/training_status')
def get_training_status():
    with training_lock:
        return jsonify(training_status)

@app.route('/predict', methods=['POST'])
def predict_molecule():
    global trained_qdd
    
    if trained_qdd is None or trained_qdd.qnn is None:
        return jsonify({'success': False, 'error': 'Model not trained yet'})
    
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'success': False, 'error': 'No JSON data received'})
        
        # Validate input parameters
        try:
            molecular_weight = float(data['molecular_weight'])
            logp = float(data['logp'])
            psa = float(data['psa'])
            h_donors = int(data['h_donors'])
            
            # Basic validation
            if not (100 <= molecular_weight <= 1000):
                raise ValueError("Molecular weight must be between 100-1000 Da")
            if not (-5 <= logp <= 10):
                raise ValueError("LogP must be between -5 and 10")
            if not (0 <= psa <= 300):
                raise ValueError("PSA must be between 0-300 Ų")
            if not (0 <= h_donors <= 20):
                raise ValueError("H-donors must be between 0-20")
                
        except (ValueError, KeyError, TypeError) as e:
            return jsonify({'success': False, 'error': f'Invalid molecular parameters: {str(e)}'})
        
        probability, is_effective = trained_qdd.predict_new_molecule([
            molecular_weight, logp, psa, h_donors
        ])
        
        confidence = abs(probability - 0.5) * 2  # Confidence measure
        
        return jsonify({
            'success': True,
            'probability': float(probability),
            'is_effective': bool(is_effective),
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'})

@app.route('/results')
def get_results():
    global trained_qdd
    
    if trained_qdd is None:
        return jsonify({'success': False, 'error': 'Model not trained yet'})
    
    try:
        plots = trained_qdd.generate_plots()
        
        return jsonify({
            'success': True,
            'accuracy': float(trained_qdd.accuracy),
            'plots': plots,
            'dataset_info': {
                'total_samples': len(trained_qdd.X_train) + len(trained_qdd.X_test),
                'training_samples': len(trained_qdd.X_train),
                'test_samples': len(trained_qdd.X_test),
                'effective_ratio': float(np.mean(trained_qdd.y_test))
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to load results: {str(e)}'})

@app.route('/sample_molecules')
def get_sample_molecules():
    """Get sample molecules for testing"""
    samples = [
        {
            'name': 'Aspirin-like',
            'molecular_weight': 180,
            'logp': 1.2,
            'psa': 63,
            'h_donors': 1,
            'description': 'Small, drug-like molecule'
        },
        {
            'name': 'Lipinski Compliant',
            'molecular_weight': 350,
            'logp': 2.5,
            'psa': 75,
            'h_donors': 2,
            'description': 'Follows Lipinski Rule of Five'
        },
        {
            'name': 'Large Molecule',
            'molecular_weight': 550,
            'logp': 4.5,
            'psa': 120,
            'h_donors': 4,
            'description': 'Challenging drug candidate'
        },
        {
            'name': 'Optimal Drug',
            'molecular_weight': 300,
            'logp': 2.0,
            'psa': 60,
            'h_donors': 1,
            'description': 'Ideal drug-like properties'
        }
    ]
    
    return jsonify({'success': True, 'samples': samples})

if __name__ == '__main__':
    print("🚀 Starting Quantum Drug Discovery Application...")
    print("📊 Navigate to: http://localhost:5000")
    print("⚛️  Quantum computing meets machine learning!")
    
    # Configure Flask for macOS stability
    app.run(
        debug=True, 
        port=5000, 
        threaded=True,
        use_reloader=False  # Disable reloader to prevent multiprocessing issues
    ) 