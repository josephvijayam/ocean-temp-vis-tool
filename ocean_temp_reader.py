import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

class OceanTempReader:
    """
    A class to read and process ocean temperature data from NetCDF files.
    """
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the OceanTempReader

        Args:
            data_dir (str): Directory where NetCDF files are stored
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def read_netcdf(self, filename: str) -> xr.Dataset:
        """
        Read a NetCDF file and return as xarray dataset

        Args:
            filename (str): name of NetCDF file
        
        Returns:
            xr.Dataset: dataset with ocean temperature data
        """
        try:
            filepath = self.data_dir / filename
            self.logger.info(f"Reading NetCDF File: {filepath}")
            dataset = xr.open_dataset(filepath)

            # Print available variables for debugging
            self.logger.info(f"Available variables in dataset: {list(dataset.variables)}")
            
            return dataset
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
        except Exception as e:
            self.logger.error(f"Error reading NetCDF File: {str(e)}")
            raise

    def extract_temperature_data(self, dataset: xr.Dataset, 
                               temp_var: str = 'sst',
                               time_slice: str = None) -> pd.DataFrame:
        """
        Extract temperature data from dataset into a pandas DataFrame.
        
        Args:
            dataset (xr.Dataset): Input dataset
            temp_var (str): Name of temperature variable in dataset
            time_slice (str): Optional time slice to extract (e.g., '2020-01')
            
        Returns:
            pd.DataFrame: DataFrame with temperature data
        """
        try:
            self.logger.info(f"Extracting temperature data from variable: {temp_var}")
            
            # Get the temperature data
            if temp_var not in dataset:
                available_vars = list(dataset.variables)
                self.logger.error(f"Temperature variable '{temp_var}' not found. Available variables: {available_vars}")
                raise KeyError(f"Variable '{temp_var}' not found in dataset")
            
            # Extract a single time slice if specified
            if time_slice:
                data = dataset[temp_var].sel(time=time_slice)
            else:
                # Take the first time slice if multiple times exist
                if 'time' in dataset[temp_var].dims:
                    data = dataset[temp_var].isel(time=0)
                else:
                    data = dataset[temp_var]
            
            self.logger.info(f"Data shape before conversion: {data.shape}")
            
            # Convert to DataFrame
            df = data.to_dataframe()
            
            # Reset index to make coordinate variables into columns
            df = df.reset_index()
            
            # Basic data validation
            if df[temp_var].isnull().all():
                raise ValueError("Temperature data contains all null values")
            
            self.logger.info(f"Successfully extracted {len(df)} temperature readings")
            self.logger.info(f"DataFrame columns: {df.columns.tolist()}")
            
            return df
            
        except KeyError as e:
            self.logger.error(f"Error accessing data: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error extracting temperature data: {str(e)}")
            raise

    
    def get_basic_stats(self, df: pd.DataFrame, 
                       temp_var: str = 'sst') -> dict:
        """
        Calculate basic statistics for temperature data.
        
        Args:
            df (pd.DataFrame): DataFrame containing temperature data
            temp_var (str): Name of temperature column
            
        Returns:
            dict: Dictionary containing basic statistics
        """
        try:
            if df is None:
                raise ValueError("Input DataFrame is None")
            
            if temp_var not in df.columns:
                raise ValueError(f"Temperature variable '{temp_var}' not found in DataFrame. Available columns: {df.columns.tolist()}")
            
            # Remove any NaN values for statistics
            clean_data = df[temp_var].dropna()
            
            stats = {
                'mean': clean_data.mean(),
                'std': clean_data.std(),
                'min': clean_data.min(),
                'max': clean_data.max(),
                'missing_values': df[temp_var].isnull().sum(),
                'total_points': len(df)
            }
            self.logger.info("Calculated basic temperature statistics")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            raise

    def visualize_sst(self, df):
        """
        Create a heatmap of sea surface temperatures.
        """
        # Pivot the data to create a 2D matrix of temperatures
        temp_matrix = df.pivot(index='lat', columns='lon', values='sst')
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Create heatmap
        sns.heatmap(temp_matrix, 
                    cmap='RdYlBu_r',  # Red-Yellow-Blue colormap (reversed)
                    center=0,         # Center the colormap at 0°C
                    cbar_kws={'label': 'Temperature (°C)'})
        
        plt.title('Global Sea Surface Temperature')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Save the plot
        plt.savefig('sst_heatmap.png')
        plt.close()

        return 'sst_heatmap.png'
        
if __name__ == "__main__":

    reader = OceanTempReader()

    try:
        dataset = reader.read_netcdf('sst_monthly.nc')
        temp_df = reader.extract_temperature_data(dataset, temp_var='sst')
        
        # Look at the DataFrame
        print("\nDataFrame Info:")
        print(temp_df.info())

        print("\nFirst few rows:")
        print(temp_df.head())

        print("\nColumns:")
        print(temp_df.columns.tolist())

        stats = reader.get_basic_stats(temp_df, temp_var='sst')
        
        print("Temperature statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Create visualization
        output_file = reader.visualize_sst(temp_df)
        print(f"Visualization saved as: {output_file}")

    except Exception as e:
        print(f"Error processing data: {str(e)}")