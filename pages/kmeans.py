import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        st.write(df.head())
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def encode_column(df, column_idx):
    """Encodes a categorical column."""
    encoder = LabelEncoder()
    df.iloc[:, column_idx] = encoder.fit_transform(df.iloc[:, column_idx])
    return encoder

def apply_kmeans_on_column(df, column_idx, encoder=None, n_clusters=0):
    """Applies KMeans clustering on a specified column of the DataFrame."""
    X = df.iloc[:, column_idx].values.reshape(-1, 1)
    
    # Scale the data if it's continuous and not in special columns
    if df.dtypes[column_idx] in [np.int64, np.float64] and column_idx not in [8, 9]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    labels = kmeans.labels_

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, labels)
    st.text(f"Silhouette Score for column {headers[column_idx]}: {silhouette_avg:.2f}")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X, np.arange(len(X)), c=kmeans.labels_, cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_, np.arange(n_clusters), s=200, c=range(n_clusters), marker='X', cmap='rainbow')

    # Add legend based on encoder
    if encoder:
        original_values = encoder.inverse_transform(np.round(kmeans.cluster_centers_).astype(int).flatten())
        legend_handles = [plt.Line2D([0], [0], marker='X', color='w', label=value, markersize=10, markerfacecolor=plt.cm.rainbow(i/n_clusters)) for i, value in enumerate(original_values)]
        plt.legend(handles=legend_handles, title=headers[column_idx])
    else:
        plt.legend([plt.Line2D([0], [0], marker='X', color='w', label=f"Centroid {i+1}", markersize=10, markerfacecolor=plt.cm.rainbow(i/n_clusters)) for i in range(n_clusters)], title=headers[column_idx])

    if column_idx == 9:  # Date column
        plt.xlabel("Date")
        flattened_X = X.flatten()
        plt.xticks(ticks=np.linspace(min(flattened_X), max(flattened_X), n_clusters), labels=[encoder.inverse_transform([int(tick)])[0] for tick in np.linspace(min(flattened_X), max(flattened_X), n_clusters)])
    else:
        plt.xlabel(headers[column_idx])

    plt.ylabel('Data Points Index')
    plt.title(f'KMeans Clustering on {headers[column_idx]}')
    st.pyplot(plt.gcf())

    return kmeans.cluster_centers_

# Load data
file_path = "hf://datasets/lllaurenceee/Shopee_Bicycle_Reviews/Dataset_D_Duplicate.csv"
df = load_data(file_path)

if df is not None:
    headers = df.columns.tolist()

    with st.expander("KMEANS SINGLE COLUMN"):
        st.write(""" ## KMEANS FOR SHOP """)
        shop_encoder = encode_column(df, 1)
        n_clusters_shop = 3
        apply_kmeans_on_column(df, column_idx=1, encoder=shop_encoder, n_clusters=n_clusters_shop)

        st.write(""" ## KMEANS FOR BRAND """)
        brand_encoder = encode_column(df, 4)
        n_clusters_date = 5
        apply_kmeans_on_column(df, column_idx=4, encoder=brand_encoder, n_clusters=n_clusters_date)

        st.write(""" ## KMEANS FOR DATE """)
        date_encoder = encode_column(df, 9)
        n_clusters_date = 5
        apply_kmeans_on_column(df, column_idx=9, encoder=date_encoder, n_clusters=n_clusters_date)
