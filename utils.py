"""This module contains utility functions for the bot."""
import logging
import os
import signal
import sys
import time


def logger(save_path: str = "logs", max_log_files: int = 9) -> None:
  """This function creates a logger object and sets up the logging configuration.

  Args:
    save_path: The path to save the log files.
    max_log_files: The maximum number of log files to keep.
  """
  os.makedirs(save_path, exist_ok=True)

  log_files = [
      os.path.join(save_path, f)
      for f in os.listdir(save_path)
      if f.endswith('_yzuCourseBot_log.txt')
  ]
  log_files.sort(key=os.path.getctime, reverse=True)

  for file in log_files[max_log_files:]:
    os.remove(file)

  log_filename = os.path.join(
      save_path,
      time.strftime("%Y-%m-%d_%H-%M-%S") + "_yzuCourseBot_log.txt",
  )
  log_format = f"%(asctime)-30s%(levelname)-10s%(message)s"
  logging.basicConfig(
      level=logging.INFO,
      format=log_format,
      handlers=[logging.FileHandler(log_filename),
                logging.StreamHandler()],
  )
  return logging.getLogger(__name__)


def enable_signal_handler() -> None:
  """This function enables the signal handler for the program."""

  def signal_handler(sig, frame):
    """This function handles the signal interrupt.
    
    Args:
      sig: The signal number.
      frame: The current stack frame.
    """
    while True:
      answer = input("繼續執行? [Y/n]: ").lower()
      if answer == 'y' or answer == "":
        return
      elif answer == 'n':
        print("正在退出...")
        sys.exit(0)
      else:
        print("無效輸入！")

  signal.signal(signal.SIGINT, signal_handler)
