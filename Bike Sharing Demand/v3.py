import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pymc as pm
import arviz as az

# Cargar datos
df = pd.read_excel("Incendios_UTM.xlsx")

# Limpieza de columnas
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^\w\s]", "", regex=True)
)
coord_cols = ["lat_decimal", "lon_decimal"]
features = ["temperatura_c", "humedad_", "superficie_total_has"]

# Filtrar datos válidos
df = df.dropna(subset=coord_cols + features)
imputer = SimpleImputer(strategy="mean")
df[features] = imputer.fit_transform(df[features])

# Escalar variables
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(
    df_scaled, columns=[f"{f}_scaled" for f in features], index=df.index
)
df = pd.concat([df, df_scaled], axis=1)

# Asignación de macrozonas artificiales
df["macrozona"] = pd.cut(
    df["lat_decimal"],
    bins=[-np.inf, -35, -33, np.inf],
    labels=["Sur", "Centro", "Norte"],
)

# Clustering DBSCAN adaptativo
df["cluster"] = -1  # Inicializar todos como ruido
eps_dict = {}

for zona in df["macrozona"].unique():
    zona_df = df[df["macrozona"] == zona]
    if len(zona_df) < 5:  # Skip if too few samples
        continue
    neighbors = NearestNeighbors(n_neighbors=5).fit(zona_df[coord_cols])
    dists, _ = neighbors.kneighbors(zona_df[coord_cols])
    eps_zona = np.percentile(
        dists[:, 4], 90
    )  # Ajustar percentil para evitar clusters demasiado grandes
    eps_dict[zona] = eps_zona
    model = DBSCAN(eps=eps_zona, min_samples=5).fit(zona_df[coord_cols])
    df.loc[zona_df.index, "cluster"] = model.labels_

# Visualizar clustering
plt.figure(figsize=(10, 6))
for c in df["cluster"].unique():
    d = df[df["cluster"] == c]
    plt.scatter(d["lon_decimal"], d["lat_decimal"], s=10, label=f"Cluster {c}")
plt.legend()
plt.title("Clustering espacial por DBSCAN con ε adaptativo")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.grid(True)
plt.show()

# Modelo predictivo (solo si hay suficientes clusters)
if len(df["cluster"].unique()) > 1:
    X = df[features + coord_cols]
    y = df["cluster"].astype(
        int
    )  # Convertir a entero para evitar problemas con etiquetas
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
else:
    print("No hay suficientes clusters para entrenar un modelo predictivo.")
