from dataclasses import dataclass
from typing import List


@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str