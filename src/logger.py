import logging
import os
from pathlib import Path

class Logger:
    def __init__(self, log_dir: str = "logs", log_filename: str = "app.log"):
        """
        Logger sınıfı, belirtilen log dizinine log kaydetmek için kullanılır.

        :param log_dir: Log dosyalarının kaydedileceği dizin
        :param log_filename: Log dosyasının adı
        """
        self.log_dir = Path(log_dir)
        self.log_filename = log_filename
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Log dizinini oluştur
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log dosyasının tam yolu
        log_path = self.log_dir / self.log_filename

        # File handler ekleme
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                      datefmt='%m/%d/%Y %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message: str):
        """ Bilgi seviyesinde log kaydı yapar. """
        self.logger.info(message)

    def warning(self, message: str):
        """ Uyarı seviyesinde log kaydı yapar. """
        self.logger.warning(message)

    def error(self, message: str):
        """ Hata seviyesinde log kaydı yapar. """
        self.logger.error(message)

    def debug(self, message: str):
        """ Debug seviyesinde log kaydı yapar. """
        self.logger.debug(message)
