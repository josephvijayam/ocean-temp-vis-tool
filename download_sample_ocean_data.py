import requests
from pathlib import Path
import logging
from tqdm import tqdm

def download_sample_sst():
    """
    Downloads a sample SST NetCDF file from NOAA's PSL data repository.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # NOAA PSL sample SST data URL (small subset of monthly data)
    url = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/sst.mnmean.nc"
    
    output_file = data_dir / "sst_monthly.nc"
    
    try:
        logger.info(f"Downloading SST data from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(output_file, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        logger.info(f"Download complete. File saved as: {output_file}")
        return output_file
        
    except requests.RequestException as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Install required packages if needed
        import pkg_resources
        required = {'requests', 'tqdm'}
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = required - installed
        
        if missing:
            print("Installing required packages...")
            import subprocess
            subprocess.check_call(['pip', 'install', *missing])
        
        # Download the data
        file_path = download_sample_sst()
        
        # Print some information about how to use the file
        print("\nSuccess! You can now use this file with the OceanTempReader:")
        print("\nExample usage:")
        print("-------------------")
        print("from ocean_temp_reader import OceanTempReader")
        print("reader = OceanTempReader()")
        print(f"dataset = reader.read_netcdf('{file_path.name}')")
        print("temp_df = reader.extract_temperature_data(dataset, temp_var='sst')")
        print("stats = reader.get_basic_stats(temp_df, temp_var='sst')")
        
    except Exception as e:
        print(f"Error: {str(e)}")