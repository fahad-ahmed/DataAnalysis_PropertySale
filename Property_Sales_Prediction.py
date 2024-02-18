import pandas as pd
import os
import sys
import numpy as np
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import precision_score, recall_score, f1_score
import webbrowser
from folium.plugins import MarkerCluster
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.geocoders import GoogleV3
from geopy.geocoders import Nominatim



def analyze_dataset(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print("File not found. Please make sure the dataset.csv file is in the correct location.")
        return

    # Load Dataset
    dataset = pd.read_csv(file_path, low_memory=False)
    #print(dataset.columns.tolist())
    dataset.reset_index()
    print(dataset.head())

    # Analysis Calculation
    attributes = ['Assessed.Value', 'Sale.Amount', 'Sales.Ratio']
    analysis_results = {
        'Range': [],
        'Mean': [],
        'Mode': [],
        'Variance': [],
        'Median': []
    }

    for attr in attributes:
        analysis_results['Range'].append((dataset[attr].min(), dataset[attr].max()))
        analysis_results['Mean'].append(round(dataset[attr].mean(),2))
        analysis_results['Mode'].append(round(dataset[attr].mode()[0],2))
        analysis_results['Variance'].append(round(np.var(dataset[attr]),2))
        analysis_results['Median'].append(round(np.median(dataset[attr]),2))

    # Improved Analysis Display
    print("\nAnalysis Results:")
    print(pd.DataFrame(analysis_results, index=attributes))



    # line Chart
    linechartDataset = dataset.sort_values(by='List.Year', ascending=True)['List.Year'].value_counts().sort_index();
    time = linechartDataset.index;
    values = linechartDataset.values
    plt.xticks(list(time), rotation=45)
    plt.title('Total House Sales year by year')
    plt.plot(time, values)
    plt.savefig('Line Chart.png')
    plt.show()


    # Bar Chart
    plt.figure(figsize=(10, 10))
    barDataset = dataset['Property.Type'].value_counts(normalize=False)
    plt.xlabel("Property Type")
    plt.ylabel("Sales Count")
    plt.title("Sales Count on Various property Type")
    plt.xticks(rotation=30)
    plt.bar(barDataset.index, barDataset.values, bottom=None, color='maroon', width=.7, align='center')
    plt.savefig('Bar Chart.png')
    plt.show()

    # Scatter plot
    plt.figure(figsize=(10, 8))
    sns.stripplot(x='List.Year', y='Sale.Amount', data=dataset, hue='Property.Type')
    plt.xlabel('List Year')
    plt.ylabel('Sale Amount')
    plt.title('Scatter plot of List Year vs Sale Amount')
    plt.xticks(rotation=45)
    plt.savefig('Scatter_plot2.png')
    plt.show()

    # Boxplots for Sale Amount, side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
    sns.boxplot(ax=axes[0], y=dataset['Assessed.Value'])
    axes[0].set_title('Assessed Value')
    axes[0].set_xlabel('')
    sns.boxplot(ax=axes[1], y=dataset['Sale.Amount'])
    axes[1].set_title('Sale Amount')
    axes[1].set_xlabel('')
    sns.boxplot(ax=axes[2], y=dataset['Sales.Ratio'])
    axes[2].set_title('Sales Ratio')
    axes[2].set_xlabel('')
    plt.tight_layout()
    plt.savefig('Boxplots_Side_by_Side.png')
    plt.show()

#load CSV
def load_csv(file_path):
    """9-
    Load a CSV file into a pandas DataFrame.
    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The DataFrame containing the CSV data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: Unable to parse the CSV file '{file_path}'. Check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


#construct filepath in code directory
def ConstructPathFromScriptDirectory(file_name):
    # Get the absolute path of the current script
    script_path = os.path.abspath(__file__)
    # Construct the full path by joining the directory of the script and the file name
    file_path = os.path.join(os.path.dirname(script_path), file_name)
    return file_path

def CleanData(dataFrame):
    missing_values = [None, 'NA', 'NAN', 'NaN', 0]
    # Replace the specified values with NaN
    dataFrame.replace(missing_values, pd.NA, inplace=True)
    # Drop rows with NaN values
    dataFrame.dropna(inplace=True)
    # Reset the index after dropping rows
    dataFrame.reset_index(drop=True, inplace=True)
    return dataFrame

def drop_columns(dataframe, columns_to_drop):
    # Use the drop method to remove the specified columns
    dataframe = dataframe.drop(columns=columns_to_drop, errors='ignore')
    return dataframe

def Calculate_mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error.

    Parameters:
    - y_true: Actual values
    - y_pred: Predicted values

    Returns:
    - MSE: Mean Squared Error
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")

    n = len(y_true)
    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
    return mse


# Example usage:
# Assuming your DataFrame is named 'df' and you want to cluster based on 'Latitude' and 'Longitude'


# Now, use the optimal_clusters value in the K-means clustering code
# ...


def add_cluster_column(dataframe, feature_columns, n_clusters):
    # Extract features for clustering
    features = dataframe[feature_columns]

    # Standardize the data (important for K-means)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply K-means clustering with explicit setting for n_init
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    dataframe['ClusterNo'] = kmeans.fit_predict(scaled_features)
    #print(dataframe.head())
    return dataframe



def remove_outliers_iqr(df, column_name, lower_bound_factor=1.5, upper_bound_factor=1.5):
    # Calculate the IQR (Interquartile Range)
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds
    lower_bound = Q1 - lower_bound_factor * IQR
    upper_bound = Q3 + upper_bound_factor * IQR

    # Create a mask to filter out rows outside the bounds
    mask = (df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)

    # Apply the mask to the DataFrame
    df_filtered = df[mask]

    return df_filtered




def calculate_classification_metrics(y_true, y_pred, threshold=500000):
    # Convert predictions to binary based on the threshold
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true_binary = (y_true > threshold).astype(int)

    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)

    return precision, recall, f1




def BuildModel(regressor):
    # Define the model
    model = Pipeline([
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('Assessed', 'passthrough', ['Assessed.Value']),
                ('date', 'passthrough', ['Date.Recorded']),              
                #('town', 'passthrough', ['Town']),
                #('County', 'passthrough', ['County']),
                ('property_type', 'passthrough', ['Property.Type']),
                #('Cluster_No', 'passthrough', ['Cluster_No']),
                ('Proximity_To_Centroid', 'passthrough', ['Proximity_To_Centroid']),
                ('List.Year', 'passthrough', ['List.Year']),
                ('geo', 'passthrough', ['Longitude', 'Latitude']),
            ]
        )),
        ('scaler', StandardScaler()),  # Standardize numerical features
        ('regressor', regressor)  # Random Forest Regressor
      
    ])
    
    return model

def TrainModel(model, dataframe):
    # Convert 'Date.Recorded' to datetime format
    dataframe['Date.Recorded'] = pd.to_datetime(dataframe['Date.Recorded'])
    #dataframe['Date.Recorded'] = dataframe['Date.Recorded'].apply(lambda x: int(x.timestamp() * 1000))

    # Drop any rows with missing values in the target variable 'Sale.Amount'
    dataframe = dataframe.dropna(subset=['Sale.Amount'])

    # Extract features and target variable
    X = dataframe[['List.Year', 'Date.Recorded', 'Property.Type','Assessed.Value', 'Longitude', 'Latitude',"Proximity_To_Centroid"]]
    y = dataframe['Sale.Amount']

    # Encode categorical features using LabelEncoder
    label_encoder = LabelEncoder()

    # Use .loc to avoid SettingWithCopyWarning
    #X.loc[:, 'Town'] = label_encoder.fit_transform(X['Town'])
    #X.loc[:, 'County'] = label_encoder.fit_transform(X['County'])
    X.loc[:, 'Property.Type'] = label_encoder.fit_transform(X['Property.Type'])
    #X.loc[:, 'HouseSize'] = label_encoder.fit_transform(X['HouseSize'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def TestModel(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    print(f'R-squared (R2) score: {r2}')
    print("\n----------------------------")


def ApplyMachineLearning(dataframe):
    data = dataframe

    random_forest_model = BuildModel(RandomForestRegressor(n_estimators=100, random_state=5))
    random_forest_model, X_test, y_test = TrainModel(random_forest_model, data)

    

    linear_regression_model = BuildModel(LinearRegression())
    linear_regression_model, X_test, y_test = TrainModel(linear_regression_model, data)


    gradient_Boosting_Regressor_model = BuildModel(GradientBoostingRegressor(n_estimators=100, random_state=0))
    gradient_Boosting_Regressor_model, X_test, y_test = TrainModel(gradient_Boosting_Regressor_model, data)


    print("\n----------------------------")
    print("\nRandom Forest Regression:")
    TestModel(random_forest_model, X_test, y_test)

    print("\nLinear Regression:")
    TestModel(linear_regression_model, X_test, y_test)

    print("\nGradient Boosting Regression:")
    TestModel(gradient_Boosting_Regressor_model, X_test, y_test)


def Remove_Outliers(loaded_data):
    
    loaded_data = remove_outliers_iqr(loaded_data, 'Sales.Ratio')
    loaded_data = remove_outliers_iqr(loaded_data, 'Latitude')
    loaded_data = remove_outliers_iqr(loaded_data, 'Longitude')
    loaded_data = remove_outliers_iqr(loaded_data, 'Assessed.Value')
    return loaded_data




def Preprocessing(dataFrame):
    dataFrame = CleanData(dataFrame)
    dataFrame = Remove_Outliers(dataFrame)
    dataFrame = drop_columns(dataFrame,'County')
    dataFrame = drop_columns(dataFrame,'Sales.Ratio')
    dataFrame = kmeans_clustering(dataFrame)
    return dataFrame



def kmeans_clustering(df, max_clusters=50):
    coordinates = df[['Latitude', 'Longitude']]

    # Standardize the features (mean=0 and variance=1)
    scaler = StandardScaler()
    coordinates_scaled = scaler.fit_transform(coordinates)

    # Using the elbow method to find the optimal number of clusters
    wcss = []  # Within-Cluster-Sum-of-Squares

    # Trying k from 1 to max_clusters
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(coordinates_scaled)
        wcss.append(kmeans.inertia_)

    # Plotting the elbow graph
    plt.plot(range(1, max_clusters + 1), wcss)
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Within-Cluster-Sum-of-Squares)')
    #plt.show()

    # Find the minimum WCSS value
    min_wcss_percentage = 20
    min_wcss = min(wcss)

    # Find the index where WCSS is at least 10% of the minimum value
    optimal_k = next((i for i, value in enumerate(wcss, start=1) if value <= min_wcss * (1 + min_wcss_percentage / 100)), None)

    if optimal_k is None:
        optimal_k = max_clusters  # If no value above the threshold, use max_clusters


    if optimal_k is None:
        optimal_k = max_clusters  # If no value above the threshold, use max_clusters


    #print(optimal_k)
    # Perform k-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    df['Cluster_No'] = kmeans.fit_predict(coordinates)

    # Calculate and add distance from centroid for each data point
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(coordinates - centroids[df['Cluster_No']], axis=1)
    df['Proximity_To_Centroid'] = distances

    return df



def create_map_with_markers(dataframe):
    """
    Creates a folium map with markers for geolocations in the given DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame with 'Longitude' and 'Latitude' columns.

    Returns:
    - folium.Map: Folium map object.
    """

    # Create a folium map centered at the mean of the coordinates
    map_center = [dataframe['Latitude'].mean(), dataframe['Longitude'].mean()]
    my_map = folium.Map(location=map_center, zoom_start=5)

    marker_cluster = MarkerCluster().add_to(my_map)

    # Iterate through the dataframe and add each point to the MarkerCluster
    for index, row in dataframe.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,  # Adjust the radius as needed
            color='blue',
            fill=True,
            fill_color='blue',
            popup=f"({row['Latitude']}, {row['Longitude']})"
        ).add_to(marker_cluster)

     # Save the map as an HTML file
    temp_html = 'temp_map.html'
    my_map.save(temp_html)

    # Open the map in the default web browser
    webbrowser.open(temp_html)

    return my_map

import colorsys

def generate_cluster_colors(max_cluster):
    # Generate distinct colors using the HSV color space starting from 0
    hsv_colors = [(i / max_cluster, 1.0, 1.0) for i in range(max_cluster)]
    rgb_colors = [colorsys.hsv_to_rgb(h, s, v) for h, s, v in hsv_colors]
    hex_colors = ['#%02x%02x%02x' % tuple(int(c * 255) for c in rgb) for rgb in rgb_colors]
    return hex_colors

def plot_house_map(df):
    # Create a folium map centered around the average latitude and longitude
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    my_map = folium.Map(location=map_center, zoom_start=10)

    # Get the maximum cluster number from the DataFrame
    max_cluster = df['Cluster_No'].max()
    #print(max_cluster)
    # Generate colors based on the maximum cluster number
    cluster_colors = generate_cluster_colors(max_cluster+1)
    #print(cluster_colors)

    # Add circle markers for each house location
    for index, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,  # You can adjust the radius as needed
            color=cluster_colors[row['Cluster_No']],  # Adjust index since colors list is 0-indexed
            #color = "Blue",
            fill=True,
            #fill_color=cluster_colors[row['Cluster_No']],
            fill_opacity=0.7,
            popup=f"Cluster: {row['Cluster_No']}"
        ).add_to(my_map)

    temp_html = 'temp_map.html'
    my_map.save(temp_html)

    # Open the map in the default web browser
    webbrowser.open(temp_html)

    return my_map


def generate_heatmap(df, features):
    """
    Generate a correlation heatmap for the specified features in the DataFrame.

    Parameters:
    - df: pandas DataFrame
    - features: list of feature names to include in the heatmap

    Returns:
    - None (displays the heatmap)
    """
    # Make a copy of the DataFrame to avoid modifying the original
    data = df[features].copy()

    label_encoder = LabelEncoder()
    data['Town']  = label_encoder.fit_transform(data['Town'])

    data['Date.Recorded'] = pd.to_datetime(data['Date.Recorded'])
    data['Date.Recorded'] = data['Date.Recorded'].apply(lambda x: int(x.timestamp() * 1000))


    correlation_matrix = data.astype(float).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

def generate_sales_coordinate_heatmap(df):
    """
    Generate a heatmap for Sales.Amount and geographical coordinates (Latitude, Longitude).

    Parameters:
    - df: pandas DataFrame containing 'Sales.Amount', 'Latitude', 'Longitude'

    Returns:
    - None (displays the plot)
    """
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df, x='Longitude', y='Latitude', fill=True, cmap='viridis', levels=20, thresh=0.05, cbar=True, cbar_kws={'label': 'Density'})
    plt.scatter(df['Longitude'], df['Latitude'], c=df['Sale.Amount'], cmap='coolwarm', s=50, edgecolors='black', linewidths=0.5)
    plt.title('Sales Amount and Geographical Coordinates Heatmap')
    plt.show()



def add_geo_location(data):
    # Assuming 'Address' and 'Town' columns exist in your DataFrame
    addresses = data['Address']
    towns = data['Town']
    # countys = data['County']
    
    merged_list = [f"{x}, {y}" for x, y in zip(addresses, towns)]
    # merged_list = [f"{x}, {y}" for x, y, z in zip(merged_list, countys)]
    
    common_string = "CT, USA"
    detail_addresses = [f"{address}, {common_string}" for address in merged_list]
    
    #print(detail_addresses)
    google_api_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    geolocator = GoogleV3(api_key=google_api_key)
    #geolocator = Nominatim(user_agent="my_geocoder")
    
    # Geocode each address
    geolocation = [geolocator.geocode(address) for address in detail_addresses]
    
    # Extract latitude and longitude
    longitude = [location.longitude if location else None for location in geolocation]
    latitude = [location.latitude if location else None for location in geolocation]
    
    # Create new columns in the DataFrame for latitude and longitude
    data['Longitude'] = longitude
    data['Latitude'] = latitude

    return data


def generate_sales_coordinate_scatter(df):
    """
    Generate a scatter plot for Sales.Amount and geographical coordinates (Latitude, Longitude).

    Parameters:
    - df: pandas DataFrame containing 'Sales.Amount', 'Latitude', 'Longitude'

    Returns:
    - None (displays the plot)
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='Sale.Amount', palette='coolwarm', size='Sale.Amount', sizes=(20, 200))
    plt.title('Sales Amount and Location Scatter Plot (Sample Size = 1000)')
    plt.show()

def RunVisualizationPart():
    file_name = 'Preporcessed.csv'
    analyze_dataset(ConstructPathFromScriptDirectory(file_name))

# main
def RunMachineLearningPart(filePath):
    loaded_data = load_csv(ConstructPathFromScriptDirectory(filePath))
    loaded_data = Preprocessing(loaded_data)
    print('\n\n')
    loaded_data = drop_columns(loaded_data, 'Cluster_No')
    loaded_data = drop_columns(loaded_data, 'Town')
    print(loaded_data.head())
    #print(len(loaded_data))
    ApplyMachineLearning(loaded_data)




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python report_2_task.py <location_of_data_file>")
    else:
        datasetPath = sys.argv[1]
        if not os.path.isfile(datasetPath):
            print("The specified file does not exist. Please check the filename and try again.")
            print("Usage: python report_2_task.py <location_of_data_file>")
        else:
            RunMachineLearningPart(datasetPath)
