import pytest
import os
import numpy as np
import xarray as xr
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from ad99py._data import (
    get_cache_dir, ensure_data, download_loon_data,
    get_loon_nc_mask_path, get_loon_np_mask_path,
    save_loon_basins, get_loon_basin_data
)




class TestCacheDir:
    """Test cases for cache directory functions."""
    
    def test_get_cache_dir_returns_path(self):
        """Test that get_cache_dir returns a Path object."""
        cache_dir = get_cache_dir()
        
        # Should return a Path object
        assert isinstance(cache_dir, Path)
        
        # Should be an absolute path
        assert cache_dir.is_absolute()
        
        # Should contain the package name
        assert "ad99py" in str(cache_dir)
    
    def test_get_cache_dir_creates_directory(self):
        """Test that get_cache_dir creates the directory if it doesn't exist."""
        # Get the cache directory
        cache_dir = get_cache_dir()
        
        # Directory should exist after calling the function
        assert cache_dir.exists()
        assert cache_dir.is_dir()
    
    def test_get_cache_dir_consistency(self):
        """Test that get_cache_dir returns the same path consistently."""
        cache_dir1 = get_cache_dir()
        cache_dir2 = get_cache_dir()
        
        # Should return the same path
        assert cache_dir1 == cache_dir2
    
    @patch('ad99py._data.user_cache_dir')
    def test_get_cache_dir_uses_platformdirs(self, mock_user_cache_dir):
        """Test that get_cache_dir uses platformdirs correctly."""
        mock_user_cache_dir.return_value = "/mock/cache/dir"
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            with patch.dict(os.environ, {}, clear=True):  # Clear env vars
                get_cache_dir()
                
                # Should call user_cache_dir with package name
                mock_user_cache_dir.assert_called_once_with("ad99py")
                
                # Should call mkdir with proper arguments
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    def test_get_cache_dir_uses_env_var(self):
        """Test that get_cache_dir uses AD99PY_CACHE_DIR environment variable."""
        test_cache_path = "/custom/cache/path"
        
        with patch.dict(os.environ, {'AD99PY_CACHE_DIR': test_cache_path}):
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                cache_dir = get_cache_dir()
                
                # Should use the environment variable path
                assert str(cache_dir) == test_cache_path
                
                # Should still call mkdir
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    def test_get_cache_dir_expands_user_path(self):
        """Test that get_cache_dir expands ~ in environment variable."""
        test_cache_path = "~/custom/cache"
        
        with patch.dict(os.environ, {'AD99PY_CACHE_DIR': test_cache_path}):
            with patch('pathlib.Path.mkdir'):
                cache_dir = get_cache_dir()
                
                # Should expand the ~ to full path
                assert not str(cache_dir).startswith("~")
                assert cache_dir.is_absolute()


class TestEnsureData:
    """Test cases for the ensure_data function."""
    
    def test_ensure_data_skips_existing_file(self):
        """Test that ensure_data doesn't download if file already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_filename = "loon_GW_momentum_fluxes.csv"
            test_content = "time,lat,lon,flux\n2020-01-01,0,0,1.5"
            
            # Create the file first
            cache_path = Path(temp_dir) / test_filename
            cache_path.write_text(test_content)
            original_content = cache_path.read_text()
            
            with patch('ad99py._data.get_cache_dir') as mock_get_cache_dir:
                mock_get_cache_dir.return_value = Path(temp_dir)
                
                with patch('requests.get') as mock_get:
                    # Call ensure_data with Stanford URL
                    result_path = ensure_data(
                        test_filename, 
                        'https://stacks.stanford.edu/file/zh044ts5443/loon_GW_momentum_fluxes.csv'
                    )
                    
                    # Should return the correct path
                    assert result_path == cache_path
                    
                    # Should not have made a request since file exists
                    mock_get.assert_not_called()
                    
                    # File should still have original content
                    assert cache_path.read_text() == original_content
    
    @patch('requests.get')
    def test_ensure_data_downloads_missing_file(self, mock_get):
        """Test that ensure_data downloads when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_filename = "loon_GW_momentum_fluxes.csv"
            test_content = b"time,lat,lon,flux\n2020-01-01,0,0,1.5"
            
            # Mock the response
            mock_response = MagicMock()
            mock_response.content = test_content
            mock_get.return_value = mock_response
            
            with patch('ad99py._data.get_cache_dir') as mock_get_cache_dir:
                mock_get_cache_dir.return_value = Path(temp_dir)
                
                cache_path = Path(temp_dir) / test_filename
                assert not cache_path.exists()
                
                result_path = ensure_data(
                    test_filename,
                    'https://stacks.stanford.edu/file/zh044ts5443/loon_GW_momentum_fluxes.csv'
                )
                
                # Should have made the request
                mock_get.assert_called_once_with(
                    'https://stacks.stanford.edu/file/zh044ts5443/loon_GW_momentum_fluxes.csv'
                )
                mock_response.raise_for_status.assert_called_once()
                
                # Should return correct path and file should exist
                assert result_path == cache_path
                assert cache_path.exists()
                assert cache_path.read_bytes() == test_content


class TestDownloadLoonData:
    """Test cases for the download_loon_data function."""
    
    @patch('ad99py._data.ensure_data')
    def test_download_loon_data_calls_ensure_data(self, mock_ensure_data):
        """Test that download_loon_data calls ensure_data with correct parameters."""
        expected_path = Path("/mock/cache/loon_GW_momentum_fluxes.csv")
        mock_ensure_data.return_value = expected_path
        
        result = download_loon_data()
        
        # Should call ensure_data with correct parameters
        mock_ensure_data.assert_called_once_with(
            'loon_GW_momentum_fluxes.csv',
            'https://stacks.stanford.edu/file/zh044ts5443/loon_GW_momentum_fluxes.csv'
        )
        
        # Should return the path from ensure_data
        assert result == expected_path
    
    def test_download_loon_data_prints_info(self, capsys):
        """Test that download_loon_data prints informational messages."""
        with patch('ad99py._data.ensure_data') as mock_ensure_data:
            mock_ensure_data.return_value = Path("/mock/path")
            
            download_loon_data()
            
            # Capture printed output
            captured = capsys.readouterr()
            
            # Should print download info
            assert "[INFO] Downloading Loon data" in captured.out
            assert "https://stacks.stanford.edu/file/zh044ts5443/loon_GW_momentum_fluxes.csv" in captured.out
            
            # Should print license info
            assert "CC BY 4.0" in captured.out
            assert "Rhodes and Candido, 2021" in captured.out
    
    def test_download_loon_data_integration(self):
        """Integration test for download_loon_data (may take time, can be skipped)."""
        # This test actually downloads the file - mark as slow/integration test
        pytest.skip("Integration test - skipping to avoid network dependency")
        
        # Uncomment to run actual download test:
        # result_path = download_loon_data()
        # assert result_path.exists()
        # assert result_path.name == 'loon_GW_momentum_fluxes.csv'
        # assert result_path.stat().st_size > 0


class TestLoonMaskPaths:
    """Test cases for Loon mask path functions."""
    
    @patch('ad99py._data.get_xarray_mask')
    def test_get_loon_nc_mask_path_creates_file(self, mock_get_xarray_mask):
        """Test that get_loon_nc_mask_path creates NetCDF file when it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the mask dataset
            mock_ds = MagicMock()
            mock_get_xarray_mask.return_value = mock_ds
            
            with patch('ad99py._data.get_cache_dir') as mock_get_cache_dir:
                mock_get_cache_dir.return_value = Path(temp_dir)
                
                cache_path = Path(temp_dir) / 'loon_masks.nc'
                assert not cache_path.exists()
                
                result_path = get_loon_nc_mask_path()
                
                # Should call get_xarray_mask and save to NetCDF
                mock_get_xarray_mask.assert_called_once()
                mock_ds.to_netcdf.assert_called_once_with(cache_path)
                
                # Should return the cache path
                assert result_path == cache_path
    
    @patch('ad99py._data.get_xarray_mask')
    def test_get_loon_nc_mask_path_skips_existing_file(self, mock_get_xarray_mask):
        """Test that get_loon_nc_mask_path doesn't recreate existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / 'loon_masks.nc'
            cache_path.touch()  # Create empty file
            
            with patch('ad99py._data.get_cache_dir') as mock_get_cache_dir:
                mock_get_cache_dir.return_value = Path(temp_dir)
                
                result_path = get_loon_nc_mask_path()
                
                # Should not call get_xarray_mask since file exists
                mock_get_xarray_mask.assert_not_called()
                
                # Should return the existing cache path
                assert result_path == cache_path
    
    @patch('ad99py._data.get_numpy_mask')
    @patch('numpy.save')
    def test_get_loon_np_mask_path_creates_file(self, mock_np_save, mock_get_numpy_mask):
        """Test that get_loon_np_mask_path creates numpy file when it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the mask array
            mock_mask = np.array([[1, 0], [0, 1]])
            mock_get_numpy_mask.return_value = mock_mask
            
            with patch('ad99py._data.get_cache_dir') as mock_get_cache_dir:
                mock_get_cache_dir.return_value = Path(temp_dir)
                
                cache_path = Path(temp_dir) / 'loon_masks.npy'
                assert not cache_path.exists()
                
                result_path = get_loon_np_mask_path()
                
                # Should call get_numpy_mask and save with numpy
                mock_get_numpy_mask.assert_called_once()
                mock_np_save.assert_called_once_with(cache_path, mock_mask)
                
                # Should return the cache path
                assert result_path == cache_path
    
    @patch('ad99py._data.get_numpy_mask')
    def test_get_loon_np_mask_path_skips_existing_file(self, mock_get_numpy_mask):
        """Test that get_loon_np_mask_path doesn't recreate existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / 'loon_masks.npy'
            cache_path.touch()  # Create empty file
            
            with patch('ad99py._data.get_cache_dir') as mock_get_cache_dir:
                mock_get_cache_dir.return_value = Path(temp_dir)
                
                result_path = get_loon_np_mask_path()
                
                # Should not call get_numpy_mask since file exists
                mock_get_numpy_mask.assert_not_called()
                
                # Should return the existing cache path
                assert result_path == cache_path


class TestSaveLoonBasins:
    """Test cases for save_loon_basins function."""
    
    @patch('ad99py._data.process_flights')
    @patch('ad99py._data.get_numpy_mask')
    @patch('ad99py._data.download_loon_data')
    @patch('numpy.save')
    def test_save_loon_basins_processes_and_saves_all_basins(
        self, mock_np_save, mock_download_loon_data, mock_get_numpy_mask, mock_process_flights
    ):
        """Test that save_loon_basins processes and saves all basin data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock inputs
            loon_data_path = Path("/mock/loon_data.csv")
            mock_download_loon_data.return_value = loon_data_path
            
            mock_masks = np.array([[1, 0], [0, 1]])
            mock_get_numpy_mask.return_value = mock_masks
            
            # Mock flight data (6 basins)
            mock_flights = [
                np.array([1, 2, 3]),  # trop_atl
                np.array([4, 5, 6]),  # extra_atl
                np.array([7, 8, 9]),  # extra_pac
                np.array([10, 11, 12]),  # indian
                np.array([13, 14, 15]),  # trop_pac
                np.array([16, 17, 18])   # SO
            ]
            mock_process_flights.return_value = mock_flights
            
            with patch('ad99py._data.get_cache_dir') as mock_get_cache_dir:
                mock_get_cache_dir.return_value = Path(temp_dir)
                
                result_paths = save_loon_basins()
                
                # Should call the dependencies correctly
                mock_download_loon_data.assert_called_once()
                mock_get_numpy_mask.assert_called_once()
                mock_process_flights.assert_called_once_with(loon_data_path, mock_masks)
                
                # Should save all 6 basin files
                assert mock_np_save.call_count == 6
                
                # Should return 6 paths
                assert len(result_paths) == 6
                
                # Check that all expected files are in the results
                expected_basins = ['trop_atl', 'extra_atl', 'extra_pac', 'indian', 'trop_pac', 'SO']
                for i, basin in enumerate(expected_basins):
                    expected_path = Path(temp_dir) / f'{basin}_flights_flux.npy'
                    assert result_paths[i] == expected_path


class TestGetLoonBasinData:
    """Test cases for get_loon_basin_data function."""
    
    def test_get_loon_basin_data_returns_existing_file(self):
        """Test that get_loon_basin_data returns existing basin file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            basin_name = "trop_atl"
            basin_file = Path(temp_dir) / f"{basin_name}_flights_flux.npy"
            basin_file.touch()  # Create empty file
            
            with patch('ad99py._data.get_cache_dir') as mock_get_cache_dir:
                mock_get_cache_dir.return_value = Path(temp_dir)
                
                with patch('ad99py._data.save_loon_basins') as mock_save_basins:
                    result_path = get_loon_basin_data(basin_name)
                    
                    # Should not call save_loon_basins since file exists
                    mock_save_basins.assert_not_called()
                    
                    # Should return the existing file path
                    assert result_path == basin_file
    
    @patch('ad99py._data.save_loon_basins')
    def test_get_loon_basin_data_generates_missing_file(self, mock_save_basins):
        """Test that get_loon_basin_data generates basin data when file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            basin_name = "indian"
            basin_file = Path(temp_dir) / f"{basin_name}_flights_flux.npy"
            
            with patch('ad99py._data.get_cache_dir') as mock_get_cache_dir:
                mock_get_cache_dir.return_value = Path(temp_dir)
                
                assert not basin_file.exists()
                
                result_path = get_loon_basin_data(basin_name)
                
                # Should call save_loon_basins to generate data
                mock_save_basins.assert_called_once()
                
                # Should return the basin file path
                assert result_path == basin_file
    
    def test_get_loon_basin_data_prints_info_when_generating(self, capsys):
        """Test that get_loon_basin_data prints info when generating missing data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            basin_name = "SO"
            
            with patch('ad99py._data.get_cache_dir') as mock_get_cache_dir:
                mock_get_cache_dir.return_value = Path(temp_dir)
                
                with patch('ad99py._data.save_loon_basins'):
                    get_loon_basin_data(basin_name)
                    
                    # Should print informational message
                    captured = capsys.readouterr()
                    assert "[INFO] Basin data for 'SO' not found in cache" in captured.out
                    assert "Generating basin data..." in captured.out