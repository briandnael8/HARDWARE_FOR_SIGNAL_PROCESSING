import pandas as pd
import matplotlib.pyplot as plt

# Charger les résultats
data = pd.read_csv("results.csv")

# Tracer les courbes
plt.plot(data["Size"], data["CPU"], label="CPU", marker='o')
plt.plot(data["Size"], data["GPU"], label="GPU", marker='o')

# Ajouter des légendes et des titres
plt.xlabel("Matrix size (rows, columns)")
plt.ylabel("Time (seconds)")
plt.title("GPU vs CPU on Matrix Multiplication")
plt.legend()
plt.grid()

# Afficher le graphique
plt.show()
