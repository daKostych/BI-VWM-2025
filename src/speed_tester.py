import time
from enum import Enum

class OperationType(Enum):
    DATA_LOAD = 1
    DATA_PREPROCESS = 2
    TFIDF_MATRIX_BUILD = 3
    INVERTED_INDEX_BUILD = 4
    SEARCH_SIMILAR_DOCUMENTS = 5

class SpeedTester:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.load_speed = None
        self.preprocess_speed = None
        self.matrix_build_speed = None
        self.inverted_index_build_speed = None
        self.search_speed = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self, operation):
        self.end_time = time.perf_counter()
        if operation == OperationType.DATA_LOAD:
            self.load_speed = self.end_time - self.start_time
        elif operation == OperationType.DATA_PREPROCESS:
            self.preprocess_speed = self.end_time - self.start_time
        elif operation == OperationType.TFIDF_MATRIX_BUILD:
            self.matrix_build_speed = self.end_time - self.start_time
        elif operation == OperationType.INVERTED_INDEX_BUILD:
             self.inverted_index_build_speed = self.end_time - self.start_time
        elif operation == OperationType.SEARCH_SIMILAR_DOCUMENTS:
            self.search_speed = self.end_time - self.start_time
        else:
            print(f"Error: Unknown operation {operation}")

    def print_statistics(self):
        print(f"Data load speed: {self.load_speed:.5f} seconds\n"
              f"Data preprocessing speed: {self.preprocess_speed:.5f} seconds\n"
              f"TF-IDF matrix build speed: {self.matrix_build_speed:.5f} seconds\n"
              f"Inverted index build speed: {self.inverted_index_build_speed:.5f} seconds\n"
              f"Search process speed: {self.search_speed:.5f} seconds")