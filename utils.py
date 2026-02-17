import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def validate_file_exists(file_path):
    """ Check if a file exists. """
    exists = os.path.isfile(file_path)
    if not exists:
        logging.error(f"File not found: {file_path}")
    return exists


def handle_error(exception):
    """ Handle exceptions and log the error message. """
    logging.error(f"An error occurred: {str(exception)}")


def save_to_file(file_path, content):
    """ Save content to a file, creating it if it doesn't exist. """
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        logging.info(f"Successfully saved to {file_path}")
    except Exception as e:
        handle_error(e)


def read_from_file(file_path):
    """ Read content from a file. """
    if validate_file_exists(file_path):
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            logging.info(f"Successfully read from {file_path}")
            return content
        except Exception as e:
            handle_error(e)
    return None
