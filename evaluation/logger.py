import os
import csv
from typing import Any, Dict, List

class Logger:
    """
    logs agent actions, market states, rewards, and relevant information to a csv file
    """

    def __init__(self, output_file: str, headers: List[str]):
        self.output_file = output_file
        self.headers = headers

        # ensure directory exists
        dirname = os.path.dirname(output_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # append mode if file exists, else write mode
        file_exists = os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0
        mode = 'a' if file_exists else 'w'

        self._file = open(self.output_file, mode, newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.headers)

        # write header only if we just created the file
        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def log_step(self, **logging_data: Any):
        """
        logs a single step of information to the csv

        :param logging_data: dictionary containing data to be logged
        """
        # build a full row, filling in '' for any missing columns
        row: Dict[str, Any] = {col: logging_data.get(col, '') for col in self.headers}

        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        """
        closes csv file
        """
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
