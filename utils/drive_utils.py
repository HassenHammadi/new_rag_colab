"""
Google Drive integration utilities for the RAG system.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

class DriveHandler:
    """Handler for Google Drive operations in Colab."""
    
    def __init__(self, base_folder: str = "RAG_vector_stores"):
        """
        Initialize the Drive handler.
        
        Args:
            base_folder: Base folder name in Google Drive
        """
        self.base_folder = base_folder
        self.is_mounted = False
        self.mount_point = "/content/drive"
        self.drive_folder = None
    
    def mount_drive(self) -> bool:
        """
        Mount Google Drive in Colab.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if we're in Colab
            import google.colab
            from google.colab import drive
            
            # Mount drive if not already mounted
            if not os.path.exists(self.mount_point):
                drive.mount(self.mount_point)
                print(f"Google Drive mounted at {self.mount_point}")
            else:
                print(f"Google Drive already mounted at {self.mount_point}")
            
            # Create base folder if it doesn't exist
            drive_path = Path(self.mount_point) / "MyDrive" / self.base_folder
            os.makedirs(drive_path, exist_ok=True)
            
            self.drive_folder = drive_path
            self.is_mounted = True
            print(f"Using Google Drive folder: {drive_path}")
            
            return True
        except ImportError:
            print("Not running in Google Colab or google.colab package not available.")
            return False
        except Exception as e:
            print(f"Error mounting Google Drive: {str(e)}")
            return False
    
    def save_to_drive(self, local_path: Union[str, Path], drive_subfolder: Optional[str] = None) -> bool:
        """
        Save files from a local path to Google Drive.
        
        Args:
            local_path: Local path to save from
            drive_subfolder: Optional subfolder within the base folder
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_mounted and not self.mount_drive():
            return False
        
        try:
            local_path = Path(local_path)
            
            # Determine the destination path
            if drive_subfolder:
                dest_path = self.drive_folder / drive_subfolder
            else:
                dest_path = self.drive_folder / local_path.name
            
            # Create destination directory if it doesn't exist
            os.makedirs(dest_path, exist_ok=True)
            
            # Copy files
            if local_path.is_dir():
                # Copy directory contents
                for item in local_path.glob('*'):
                    if item.is_file():
                        shutil.copy2(item, dest_path)
                    else:
                        shutil.copytree(item, dest_path / item.name, dirs_exist_ok=True)
                print(f"Copied directory {local_path} to Google Drive: {dest_path}")
            else:
                # Copy single file
                shutil.copy2(local_path, dest_path)
                print(f"Copied file {local_path} to Google Drive: {dest_path}")
            
            return True
        except Exception as e:
            print(f"Error saving to Google Drive: {str(e)}")
            return False
    
    def load_from_drive(self, drive_path: str, local_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Load files from Google Drive to a local path.
        
        Args:
            drive_path: Path within the base folder in Google Drive
            local_path: Optional local destination path
            
        Returns:
            Path to the local files if successful, None otherwise
        """
        if not self.is_mounted and not self.mount_drive():
            return None
        
        try:
            # Determine the source path
            source_path = self.drive_folder / drive_path
            
            if not source_path.exists():
                print(f"Path does not exist in Google Drive: {source_path}")
                return None
            
            # Determine the destination path
            if local_path is None:
                local_path = Path(tempfile.mkdtemp()) / source_path.name
            else:
                local_path = Path(local_path)
                os.makedirs(local_path, exist_ok=True)
            
            # Copy files
            if source_path.is_dir():
                # Copy directory contents
                for item in source_path.glob('*'):
                    if item.is_file():
                        shutil.copy2(item, local_path)
                    else:
                        shutil.copytree(item, local_path / item.name, dirs_exist_ok=True)
                print(f"Copied directory from Google Drive: {source_path} to {local_path}")
            else:
                # Copy single file
                shutil.copy2(source_path, local_path)
                print(f"Copied file from Google Drive: {source_path} to {local_path}")
            
            return local_path
        except Exception as e:
            print(f"Error loading from Google Drive: {str(e)}")
            return None
    
    def list_vector_stores(self) -> list:
        """
        List available vector stores in Google Drive.
        
        Returns:
            List of vector store names
        """
        if not self.is_mounted and not self.mount_drive():
            return []
        
        try:
            # Get all subdirectories in the base folder
            return [d.name for d in self.drive_folder.glob('*') if d.is_dir()]
        except Exception as e:
            print(f"Error listing vector stores: {str(e)}")
            return []
