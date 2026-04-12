import logging
import os

def get_logger(name="project_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)

        file_handler = logging.FileHandler("logs/app.log")
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger