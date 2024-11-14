import os
import glob
import xarray as xr
from tqdm import tqdm
import numpy as np

from utils.plot_utils import plt
from utils.plot_utils import TextColors 

green_color = '#81C784'

def read_tiff_files_with_same_crs(directory, crs):
    """
    Read GeoTIFF files from a directory and set the same CRS for each of them.

    Parameters:
        directory (str): Path to the directory containing GeoTIFF files.
        crs (str): CRS string to set for each GeoTIFF file.

    Returns:
        list: List of xarray DataArrays, each representing a GeoTIFF file with the same CRS.
    """
    tiff_files = glob.glob(os.path.join(directory, "*.tif"))
    data_arrays = []

    for tiff_file in tqdm(tiff_files, desc="Reading TIFF files", unit="files"):
        try:
            # Read the TIFF file
            data_array = xr.open_rasterio(tiff_file)
            
            # Set the CRS
            data_array.attrs['crs'] = crs
            
            # Get the name of the TIFF file (without extension) and set it as the name of the data array
            tiff_name = os.path.splitext(os.path.basename(tiff_file))[0]
            data_array.name = tiff_name
            
            # Append the data array to the list
            data_arrays.append(data_array)
        except Exception as e:
            print(f"Error reading file {tiff_file}: {e}")

    return data_arrays



import pandas as pd
import xarray as xr
from shapely.geometry import Point

from tqdm import tqdm

def overlay_points_on_data(data_arrays, points_csv_path, output_csv_path):
    """
    Overlay points from a CSV file onto GeoTIFF data arrays and save the results to another CSV file.

    Parameters:
        data_arrays (list): List of xarray DataArrays representing GeoTIFF data.
        points_csv_path (str): Path to the CSV file containing points with latitudes and longitudes.
        output_csv_path (str): Path to save the resulting CSV file with overlaid points.

    Returns:
        None
    """
    # Your import statement for pandas is missing. I'll add it here.
    import pandas as pd  

    # Read points from CSV file
    points_df = pd.read_csv(points_csv_path)

    # Create a DataFrame to store overlaid points data
    overlaid_points_df = pd.DataFrame()

    # Set colors for waiting bar and text within []
    bar_color = TextColors.OKGREEN  # Change this to your desired color for the progress bar
    text_color = TextColors.OKBLUE  # Change this to your desired color for the text within []

    # Iterate over data arrays with tqdm progress bar
    for data_array in tqdm(data_arrays, desc=bar_color + "Overlaying points", unit="array"):
        # Extract CRS information
        crs = data_array.attrs.get('crs')

        # Get the name of the .tif file
        tif_name = data_array.name
        print(TextColors.OKCYAN + tif_name + TextColors.ENDC)

        # Create a DataFrame to store overlaid points data for this data_array
        tif_points_df = pd.DataFrame()

        # Iterate over points with tqdm progress bar
        for index, point_row in tqdm(points_df.iterrows(), desc=text_color + "Overlaying points", unit="point", leave=False):
            # Create a Point object from lat and lon
            point = Point(point_row['X'], point_row['Y'])

            # Convert point to the same CRS as the data array if CRS is available
            if crs:
                point = point

            # Extract data value at the nearest grid cell to the point location
            data_value = data_array.sel(x=point.x, y=point.y, method='nearest').values.item()

            # Add data value to the point data with column name as tif_name
            point_data = {tif_name: data_value}
            tif_points_df = pd.concat([tif_points_df, pd.DataFrame([point_data])], ignore_index=True)

        # Rename the column to tif_name
        tif_points_df = tif_points_df.rename(columns={0: tif_name})

        # Concatenate tif_points_df with overlaid_points_df
        overlaid_points_df = pd.concat([overlaid_points_df, tif_points_df], axis=1)

    # Concatenate points_df with overlaid_points_df
    overlaid_points_df = pd.concat([points_df, overlaid_points_df], axis=1)

    # Save overlaid points data to CSV file
    overlaid_points_df.to_csv(output_csv_path, index=False)


import os
import rasterio
import matplotlib.pyplot as plt

def plot_raster_histogram(directory):
    """
    Read raster files from a directory and plot the histogram of each image.
    
    Parameters:
    -----------
    directory : str
        Path to the directory containing raster files.
    """
    # Get list of files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Loop through each file in the directory
    for file in files:
        if file.endswith('.tif') or file.endswith('.tiff'):
            filepath = os.path.join(directory, file)
            with rasterio.open(filepath) as src:
                # Read raster image
                img = src.read(1)

                # Remove NaN values
                img = np.nan_to_num(img)
                
                # Normalize pixel values between 0 and 1
                img_min = np.min(img)
                img_max = np.max(img)
                img_normalized = (img - img_min) / (img_max - img_min)
                
                # Plot histogram
                plt.figure(figsize=(8, 5))
                plt.hist(img_normalized.flatten(), bins=100, color= green_color, edgecolor='black', alpha=0.7)
                plt.xlabel('Pixel Value')
                plt.ylabel('Frequency')
                plt.title('Histogram of {}'.format(file))
                plt.grid(True)
                plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def plot_dataframe_histogram(csv_file):
    """
    Read a CSV file into a Pandas DataFrame and plot the histogram of each column.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file.
    """
    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Plot histogram for each column
    for column in df.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(df[column], bins=100, color=green_color, edgecolor='gray', alpha=0.7)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of {}'.format(column))
        plt.grid(True)
        plt.show()

import pandas as pd













