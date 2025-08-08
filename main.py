import torch
import matplotlib.pyplot as plt

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image resolution
width, height = 1000, 1000

# Complex plane boundaries
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.5, 1.5

# Generate complex grid
x = torch.linspace(x_min, x_max, width, device=device)
y = torch.linspace(y_min, y_max, height, device=device)
X, Y = torch.meshgrid(x, y, indexing="ij")
Z = X + 1j * Y
C = torch.tensor(1j, device=device)

# Initialize escape time count
max_iters = 300
escape_radius = 10.0
escaped_at = torch.zeros_like(Z.real, dtype=torch.float32)

# Iterative fractal generation
z = Z.clone()
mask = torch.ones_like(Z, dtype=torch.bool)

for i in range(max_iters):
    z[mask] = z[mask] ** 2 + C
    escaped = (z.abs() > escape_radius) & mask
    escaped_at[escaped] = i
    mask &= ~escaped

# Normalize for coloring
escaped_at /= max_iters

# Move to CPU for display
image = escaped_at.cpu().numpy()

# Display
plt.figure(figsize=(8, 8))
plt.imshow(image.T, cmap="inferno", extent=[x_min, x_max, y_min, y_max])
plt.title("Dendrite Julia Set")
plt.xlabel("Re")
plt.ylabel("Im")
plt.axis("off")
plt.show()

