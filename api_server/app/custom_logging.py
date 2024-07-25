import logging
from logging.handlers import TimedRotatingFileHandler
import os
import datetime

class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, *args, **kwargs):
        self.log_dir = kwargs.pop('log_dir', 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        super().__init__(*args, **kwargs)

    def get_new_filename(self):
        # Get the current date and time for the filename suffix
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.log_dir, f'asdinsight_{timestamp}.log')

    def doRollover(self):
        # Close the current log file and rename it with a timestamp
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Rename the current log file to include a timestamp
        if os.path.exists(self.baseFilename):
            new_filename = self.get_new_filename()
            os.rename(self.baseFilename, new_filename)
        
        # Create a new log file
        self.mode = 'w'
        self.stream = self._open()