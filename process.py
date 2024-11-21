import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the dataset
file_path = 'spotify dataset.csv'
df = pd.read_csv(file_path)

# Step 2: Handle Missing Values
# Fill missing values in numerical columns with their mean
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Fill missing values in non-numerical columns with a placeholder or mode
non_numeric_columns = df.select_dtypes(include=['object']).columns
df[non_numeric_columns] = df[non_numeric_columns].fillna('Unknown')

# Step 3: Normalize Numerical Features
numerical_features = ['tempo', 'energy', 'danceability', 'loudness', 
                      'valence', 'speechiness', 'instrumentalness', 'liveness', 'duration_ms']
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Step 4: Handle Duplicates
df.drop_duplicates(inplace=True)

# Step 5: Save Cleaned Data
cleaned_file_path = 'spotify_cleaned_dataset.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned data saved to {cleaned_file_path}")

# Helper function to save plots
def save_plot(plot, filename):
    plot.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

# Step 6: Visualization - Top 10 Playlist Genres
genre_counts = df['playlist_genre'].value_counts().head(10).reset_index()
genre_counts.columns = ['Genre', 'Count']  # Renaming for clarity

plt.figure(figsize=(10, 6))
sns.barplot(data=genre_counts, x='Count', y='Genre', palette='viridis')  # Using 'data' fixes the issue
plt.title('Top 10 Playlist Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
save_plot(plt, 'top_10_genres.png')

# Step 7: Correlation Matrix
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numerical_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Features')
save_plot(plt, 'correlation_matrix.png')

# Step 8: Visualization - Popularity Distribution with Legends
plt.figure(figsize=(10, 6))

# Plot the histogram with KDE
sns.histplot(df['track_popularity'], bins=20, kde=True, color='blue', label='Track Popularity')

# Add vertical lines for mean and median
mean_popularity = df['track_popularity'].mean()
median_popularity = df['track_popularity'].median()

plt.axvline(mean_popularity, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_popularity:.2f}')
plt.axvline(median_popularity, color='green', linestyle='--', linewidth=1.5, label=f'Median: {median_popularity:.2f}')

# Title, labels, and legend
plt.title('Track Popularity Distribution', fontsize=14)
plt.xlabel('Track Popularity', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(title='Key', loc='upper left')

# Save the plot
save_plot(plt, 'popularity_distribution_with_legends.png')

# Step 9: Visualization - Danceability vs Energy using Hexbin Plot with Custom Colors
plt.figure(figsize=(10, 6))
hb = plt.hexbin(df['danceability'], df['energy'], gridsize=30, cmap='Blues', mincnt=1)
cb = plt.colorbar(hb)
cb.set_label('Frequency')
plt.title('Danceability vs Energy Hexbin Plot')
plt.xlabel('Danceability')
plt.ylabel('Energy')
save_plot(plt, 'danceability_vs_energy_hexbin.png')

# Step 10: Visualization - Average Features by Playlist Genre
feature_means = df.groupby('playlist_genre')[numerical_features].mean().reset_index()
feature_means = pd.melt(feature_means, id_vars='playlist_genre', var_name='Feature', value_name='Average Value')

plt.figure(figsize=(14, 8))
sns.barplot(data=feature_means, x='Feature', y='Average Value', hue='playlist_genre', palette='tab20')
plt.title('Average Features by Playlist Genre')
plt.xticks(rotation=45)
save_plot(plt, 'average_features_by_genre.png')

# Step 11: Visualization - Duration Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['duration_ms'], bins=30, kde=True, color='green')
plt.title('Track Duration Distribution')
plt.xlabel('Duration (scaled)')
plt.ylabel('Frequency')
save_plot(plt, 'duration_distribution.png')

print("\nDuration distribution plot saved as 'duration_distribution.png'")
print("\nAnalysis and visualizations completed. Plots have been saved.")



