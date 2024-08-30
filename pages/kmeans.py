import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datasets import load_dataset
import csv

def load_data(file_path):
    try:
        # Read the CSV file
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)  # Get the headers
            data = [row for row in csv_reader]  # Get the rest of the data
        
        # Display the first few rows of data
        st.write("Headers:", headers)
        st.write("First few rows of data:", data[:5])
        
        return headers, data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Define the file path
file_path = "hf://datasets/lllaurenceee/Shopee_Bicycle_Reviews/Dataset_D_Duplicate.csv"

# Load and process the data
headers, data = load_data(file_path)


# Define the KMeans clustering function
def apply_kmeans_on_single_column(data, column_idx, encoder=None, n_clusters=0):
    X = np.array([[row[column_idx]] for row in data])

    # Scale the data if it's continuous and not in special columns
    if isinstance(X[0][0], (int, float)) and column_idx not in [5, 6]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    labels = kmeans.labels_

    # Calculate and print the silhouette score
    silhouette_avg = silhouette_score(X_scaled, labels)
    st.text(f"Silhouette Score for column {headers[column_idx]}: {silhouette_avg:.2f}")

    # Extract cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X, [i for i in range(len(X))], c=kmeans.labels_, cmap='rainbow')

    # Plot centroids
    centroid_colors = [plt.cm.rainbow(label/n_clusters) for label in range(n_clusters)]
    plt.scatter(kmeans.cluster_centers_, [i for i in range(n_clusters)], s=200, c=centroid_colors, marker='X')

    # Add legend based on encoder
    if encoder:
        original_values = [encoder.inverse_transform([int(center)])[0] for center in np.round(kmeans.cluster_centers_).flatten()]
        legend_handles = [plt.Line2D([0], [0], marker='X', color='w', label=value, markersize=10, markerfacecolor=centroid_colors[i]) for i, value in enumerate(original_values)]
        plt.legend(handles=legend_handles, title=headers[column_idx])
    else:
        legend_handles = [plt.Line2D([0], [0], marker='X', color='w', label=cluster_names[i], markersize=10, markerfacecolor=centroid_colors[i]) for i in range(n_clusters)]
        plt.legend(handles=legend_handles, title=headers[column_idx])

    if column_idx == 9:  # Date column
        plt.xlabel("Date")
        flattened_X = X.flatten()
        plt.xticks(ticks=np.linspace(min(flattened_X), max(flattened_X), n_clusters), labels=[encoder.inverse_transform([int(tick)])[0] for tick in np.linspace(min(flattened_X), max(flattened_X), n_clusters)])
    else:
        plt.xlabel(headers[column_idx])

    plt.ylabel('Data Points Index')
    plt.title(f'KMeans Clustering on {headers[column_idx]}')
    plt.show()
    st.pyplot(plt.gcf())

    return kmeans.cluster_centers_


#this is it
# Using label encoders to convert string data into numeric for clustering
date_encoder = LabelEncoder()

date = [row[9] for row in data]

encoded_date = date_encoder.fit_transform(date)

for idx, row in enumerate(data):
    row[9] = encoded_date[idx]  # Encoded brand

# Example usage of the function
column_idx = 9  # Replace with the index of the column you want to analyze
n_clusters = 5

# Call the function with the specified column index
cluster_centers = apply_kmeans_on_single_column(
    data,
    column_idx=column_idx,
    encoder=date_encoder,  # Use encoder if applicable
    n_clusters=n_clusters
)

quantity_centroid_meanings = [
  "1",
  "2",
  "3",
  "4",
  "5"
]

for i, center in enumerate((cluster_centers)):
    st.text(f"Centroid {i+1}: {quantity_centroid_meanings[i]} (Value: {center[0]:.2f})")
