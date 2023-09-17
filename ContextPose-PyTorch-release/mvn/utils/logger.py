import os
import logging


class Logger():
	def __init__(self, log_path, level="DEBUG"):
		self.logger = logging.getLogger()
		self.logger.setLevel(level)
		self.log_path = log_path
		self.add_handler()

	def console_handler(self,level="DEBUG"):
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)

		console_handler.setFormatter(self.get_formatter()[0])

		return console_handler

	def file_handler(self, level="DEBUG"):
			file_handler = logging.FileHandler(os.path.join(self.log_path, "log.txt"),mode="a",encoding="utf-8")
			file_handler.setLevel(level)

			file_handler.setFormatter(self.get_formatter()[1])

			return file_handler

	def get_formatter(self):
		console_fmt = logging.Formatter(fmt="%(asctime)s: %(message)s")
		file_fmt = logging.Formatter(fmt="%(asctime)s: %(message)s")

		return console_fmt,file_fmt

	def add_handler(self):
		self.logger.addHandler(self.console_handler())
		self.logger.addHandler(self.file_handler())
