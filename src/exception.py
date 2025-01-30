import sys
import logging
from src.logger import logging
from src import logger

def error_message_details(error, error_details: sys):
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error Occured in scripts: {file_name}, line: {line_number}, error: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details)

    def __str__(self):
        return self.error_message
    
if __name__=='__main__':
    try:
        logger.logging.info('Enter the try block')
        a = 1/0
        print('This is will not be printed')
    except Exception as e:
        raise CustomException(e, sys)