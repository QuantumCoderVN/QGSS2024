import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# Configure parameters
n_qubits = 3  # Total number of qubits / N
n_a_qubits = 1  # Number of ancillary qubits / N_A
q_depth = 3  # Depth of the parameterised quantum circuit / D
n_generators = 4  # Number of subgenerators for the patch method / N_G
image_size = 8  
batch_size = 1
lrG = 0.3  # Learning rate for the generator
lrD = 0.01  # Learning rate for the discriminator
num_iter = 500  # Number of training iterations  


class DigitsDataset(Dataset):
    def __init__(self, csv_file, label=0, transform=None):
        self.csv_file = csv_file
        self.transform = transform
        self.df = self.filter_by_label(label)

    def filter_by_label(self, label):
        df = pd.read_csv(self.csv_file)
        df = df.loc[df.iloc[:, -1] == label]
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.df.iloc[idx, :-1] / 16
        image = np.array(image).astype(np.float32).reshape(8, 8)
        if self.transform:
            image = self.transform(image)
        return image, 0


dataset = DigitsDataset(csv_file="/home/hpc.ducdn/gan/optdigits.tra")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Display some sample images
plt.figure(figsize=(8, 2))
for i in range(8):
    image = dataset[i][0]
    plt.subplot(1, 8, i + 1)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
plt.show()

#Generator
class QuantumGenerator:
    def __init__(self, n_qubits, q_depth, n_generators, q_delta=1.0):
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.n_generators = n_generators
        self.q_params = [
            np.random.uniform(-q_delta, q_delta, (q_depth, n_qubits))
            for _ in range(n_generators)
        ]
        self.simulator = AerSimulator()

    def quantum_circuit(self, noise, weights):
        circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circuit.ry(float(noise[i]), i)
        circuit.barrier()
        for layer in weights:
            for i, theta in enumerate(layer):
                circuit.ry(float(theta), i)
            circuit.barrier ()
            for i in range(self.n_qubits - 1):
                circuit.cz(i, i + 1)
        circuit.measure_all()
        return circuit

    def partial_measure(self, noise, weights, shots=1024):
    
        circuit = self.quantum_circuit(noise, weights)
    
        
        job = self.simulator.run(circuit, shots=shots)
        result = job.result()
    
        
        counts = result.get_counts(circuit)
    
        
        probs = np.array(
            [counts.get(format(i, f'0{self.n_qubits}b'), 0) for i in range(2 ** self.n_qubits)],
            dtype=float  
        )
        probs /= np.sum(probs)  
    
        
        probsgiven0 = probs[: 2 ** (self.n_qubits - n_a_qubits)]
    
        
        if np.max(probsgiven0) > 0:
            probsgiven0 /= np.max(probsgiven0)
    
        return probsgiven0


    def generate(self, noise_batch):
        
        patch_size = 2 ** (self.n_qubits - n_a_qubits)
        generated_images = []

        for noise in noise_batch:
            patches = []
            for weights in self.q_params:
                patch = self.partial_measure(noise, weights)
                patches.append(patch)
            
            generated_image = np.hstack(patches)
            generated_image = np.resize(generated_image, (image_size * image_size))  
            generated_images.append(generated_image)

        return np.array(generated_images)

    def compute_gradient(self, noise, weights, epsilon=np.pi / 2):
        grad = np.zeros_like(weights)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                weights[i, j] += epsilon
                probs_plus = self.partial_measure(noise, weights)
                weights[i, j] -= 2 * epsilon
                probs_minus = self.partial_measure(noise, weights)
                weights[i, j] += epsilon
                grad[i, j] = (probs_plus.sum() - probs_minus.sum()) / (2 * np.sin(epsilon))
        return grad

    def update_params(self, noise_batch, gradients, learning_rate):
        for i, weights in enumerate(self.q_params):
            grad = np.mean([self.compute_gradient(noise, weights) for noise in noise_batch], axis=0)
            self.q_params[i] -= learning_rate * grad


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# Initialize generator and discriminator
generator = QuantumGenerator(n_qubits, q_depth, n_generators)
discriminator = Discriminator().to(torch.device("cpu"))
"""
# Quantum circuit
noise_sample = np.random.uniform(0, np.pi / 2, (n_qubits,))  
weights_sample = generator.q_params[0]  
quantum_circuit = generator.quantum_circuit(noise_sample, weights_sample)  
quantum_circuit.draw()
"""
# Training
generator = QuantumGenerator(n_qubits, q_depth, n_generators)
discriminator = Discriminator()
criterion = nn.BCELoss()
optD = optim.SGD(discriminator.parameters(), lr=lrD)
generator_losses = []
discriminator_losses = []
for epoch in range(num_iter):
    for data, _ in dataloader:
        real_data = data.reshape(-1, image_size * image_size).numpy()
        noise_batch = np.random.uniform(0, np.pi / 2, (batch_size, n_qubits))
        fake_data = generator.generate(noise_batch)  
        # Train Discriminator
        discriminator.zero_grad()
        real_output = discriminator(torch.tensor(real_data, dtype=torch.float32)).view(-1)
        fake_output = discriminator(torch.tensor(fake_data, dtype=torch.float32)).view(-1)
        errD = criterion(real_output, torch.ones_like(real_output)) + \
               criterion(fake_output, torch.zeros_like(fake_output))
        errD.backward()
        optD.step()
        discriminator_losses.append(errD.item())

        # Train Generator
        gradients = [generator.compute_gradient(noise, generator.q_params[0]) for noise in noise_batch]
        generator.update_params(noise_batch, gradients, lrG)

        # Calculate Generator loss
        fake_output = discriminator(torch.tensor(fake_data, dtype=torch.float32)).view(-1)
        errG = criterion(fake_output, torch.ones_like(fake_output))
        generator_losses.append(errG.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{num_iter}, Loss D: {errD.item():.5f}, Loss G: {errG.item():.5f}")

plt.close()
plt.plot(generator_losses, label="Generator Loss")
plt.plot(discriminator_losses, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.title("Training")
plt.savefig("Traning")



for epoch in range (1,num_iter+1):
    if epoch % 100 == 0:    
        print(f"Epoch {epoch}/{num_iter}")
        fixed_noise = np.random.uniform(0, np.pi / 2, (8, n_qubits)) 
        generated_images = generator.generate(fixed_noise)  # Generate image from generator
        num_images_to_show = 8  
        plt.figure(figsize=(8, 2)) 
        for i in range(num_images_to_show):
            plt.subplot(1, 8, i + 1)
            plt.imshow(generated_images[i].reshape(image_size, image_size), cmap="gray")
            plt.axis("off")   
        plt.title ("Image")
        plt.savefig("Image")
