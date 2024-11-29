import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import logging

class OceanReader:
    """
    A class to read and process ocean temperature and current data from NetCDF files.
    """
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the OceanReader

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
            xr.Dataset: dataset with ocean data
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
        # Code for extract_temperature_data remains the same as before
        return self._extract_data(dataset, temp_var, time_slice)

    def extract_current_data(self, dataset: xr.Dataset,
                            u_var: str = 'uo',
                            v_var: str = 'vo',
                            time_slice: str = None) -> pd.DataFrame:
        """
        Extract current data (u and v components) from dataset into a pandas DataFrame.
        
        Args:
            dataset (xr.Dataset): Input dataset
            u_var (str): Name of eastward current variable in dataset
            v_var (str): Name of northward current variable in dataset
            time_slice (str): Optional time slice to extract (e.g., '2020-01')
            
        Returns:
            pd.DataFrame: DataFrame with current data
        """
        return self._extract_data(dataset, u_var, time_slice, v_var=v_var)

    def _extract_data(self, dataset: xr.Dataset, 
                    var1: str, time_slice: str = None, 
                    var2: str = None) -> pd.DataFrame:
        """
        Helper function to extract data from a dataset.
        
        Args:
            dataset (xr.Dataset): Input dataset
            var1 (str): Name of the first variable to extract
            time_slice (str): Optional time slice to extract (e.g., '2020-01')
            var2 (str): Name of the second variable to extract (optional)
            
        Returns:
            pd.DataFrame: DataFrame with the extracted data
        """
        try:
            self.logger.info(f"Extracting data from variable(s): {var1}, {var2 if var2 else ''}")
            
            # Get the data
            if var1 not in dataset:
                available_vars = list(dataset.variables)
                self.logger.error(f"Variable '{var1}' not found. Available variables: {available_vars}")
                raise KeyError(f"Variable '{var1}' not found in dataset")
            
            # Extract a single time slice if specified
            if time_slice:
                data1 = dataset[var1].sel(time=time_slice)
                if var2:
                    data2 = dataset[var2].sel(time=time_slice)
            else:
                # Take the first time slice if multiple times exist
                if 'time' in dataset[var1].dims:
                    data1 = dataset[var1].isel(time=0)
                    if var2:
                        data2 = dataset[var2].isel(time=0)
                else:
                    data1 = dataset[var1]
                    if var2:
                        data2 = dataset[var2]
            
            self.logger.info(f"Data1 shape: {data1.shape}")
            if var2:
                self.logger.info(f"Data2 shape: {data2.shape}")
            
            # Convert to DataFrame
            df = data1.to_dataframe()
            if var2:
                df[var2] = data2.values
            
            # Reset index to make coordinate variables into columns
            df = df.reset_index()
            
            # Basic data validation
            if df[var1].isnull().all():
                raise ValueError(f"Data for variable '{var1}' contains all null values")
            if var2 and df[var2].isnull().all():
                raise ValueError(f"Data for variable '{var2}' contains all null values")
            
            self.logger.info(f"Successfully extracted {len(df)} data points")
            self.logger.info(f"DataFrame columns: {df.columns.tolist()}")
            
            return df
            
        except KeyError as e:
            self.logger.error(f"Error accessing data: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error extracting data: {str(e)}")
            raise

    def get_basic_stats(self, df: pd.DataFrame, 
                       var: str) -> dict:
        """
        Calculate basic statistics for a data variable.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            var (str): Name of the data variable
            
        Returns:
            dict: Dictionary containing basic statistics
        """
        try:
            if df is None:
                raise ValueError("Input DataFrame is None")
            
            if var not in df.columns:
                raise ValueError(f"Variable '{var}' not found in DataFrame. Available columns: {df.columns.tolist()}")
            
            # Remove any NaN values for statistics
            clean_data = df[var].dropna()
            
            stats = {
                'mean': clean_data.mean(),
                'std': clean_data.std(),
                'min': clean_data.min(),
                'max': clean_data.max(),
                'missing_values': df[var].isnull().sum(),
                'total_points': len(df)
            }
            self.logger.info(f"Calculated basic statistics for variable '{var}'")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            raise

    def visualize_data(self, temp_df: pd.DataFrame, current_df: pd.DataFrame = None, save_path: str = 'ocean_data_visualization.png'):
        """
        Create a map visualization showing both sea surface temperature and ocean currents.
        
        Args:
            temp_df (pd.DataFrame): DataFrame with sea surface temperature data
            current_df (pd.DataFrame): DataFrame with ocean current data
            save_path (str): Path to save the output visualization
        """
        plt.figure(figsize=(15, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.gridlines(draw_labels=True)

        # Plot temperature data
        temp_pivot = temp_df.pivot(index='lat', columns='lon', values='sst')
        temp_mesh = ax.pcolormesh(temp_pivot.columns, temp_pivot.index, temp_pivot,
                                transform=ccrs.PlateCarree(),
                                cmap='RdYlBu_r',
                                shading='auto')
        plt.colorbar(temp_mesh, label='Sea Surface Temperature (°C)', orientation='horizontal', pad=0.05)

        # Plot current data
        # ax.quiver(current_df['lon'], current_df['lat'], current_df['u'], current_df['v'],
        #         transform=ccrs.PlateCarree(),
        #         color='k',
        #         scale=30)

        plt.title('Sea Surface Temperature and Ocean Currents')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        self.logger.info(f"Visualization saved as {save_path}")
        return save_path

if __name__ == "__main__":
    reader = OceanReader()

    # Read temperature and current data
    temp_dataset = reader.read_netcdf('sst_monthly.nc')
    # current_dataset = reader.read_netcdf('ocean_currents.nc')

    # Extract temperature and current data
    temp_df = reader.extract_temperature_data(temp_dataset, temp_var='sst')
    # current_df = reader.extract_current_data(current_dataset, u_var='uo', v_var='vo')

    # Visualize both temperature and current data
    reader.visualize_data(temp_df, save_path='ocean_data_visualization.png')