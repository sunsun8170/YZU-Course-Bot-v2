"""This is the main script for the course bot."""
import getpass
import json
import os
import sys
import time

from functools import wraps
from typing import Any, Callable, Dict, Final, Tuple

import io
from PIL import Image
import re

from bs4 import BeautifulSoup
import requests

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import utils

# url configs
LOGIN_URL_REFERER: Final = "https://isdna1.yzu.edu.tw/Cnstdsel/default.aspx"
LOGIN_URL: Final = "https://isdna1.yzu.edu.tw/CnStdSel/Index.aspx"
CAPTCHA_URL: Final = "https://isdna1.yzu.edu.tw/CnStdSel/SelRandomImage.aspx"
COURSE_LIST_URL: Final = "https://isdna1.yzu.edu.tw/CnStdSel/SelCurr/CosList.aspx"
COURSE_SELECT_URL: Final = "https://isdna1.yzu.edu.tw/CnStdSel/SelCurr/CurrMainTrans.aspx?mSelType=SelCos&mUrl="
# model configs
PRETRAINED_MODEL_NAME: Final = "microsoft/trocr-small-printed"
PRETRAINED_MODEL_PATH: Final = "./model"
# general configs
REQUEST_TIMEOUT: Final = 5
DEBUG_MODE: Final = False  # set to True to enable detailed error messages
LOG_DIR: Final = "./logs"


def handle_exceptions(wait: int = 0.5) -> Callable[[Callable], Callable]:
  """A decorator that handles exceptions and retries the function.
  
  Args:
    wait: The waiting time in seconds before retrying the function.
  """

  def decorator(func: Callable) -> Callable:

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
      while True:
        try:
          return func(self, *args, **kwargs)
        except requests.RequestException as e:
          self.logger.error(f"[ 網路異常 ] 嘗試連線中! 詳細資訊: {e}", exc_info=DEBUG_MODE)
          time.sleep(wait)
        except Exception as e:
          self.logger.critical(f"[ 未知的錯誤 ] 詳細資訊: {e}", exc_info=True)
          sys.exit(0)

    return wrapper

  return decorator


class CourseBot:
  """A course bot for automatically selecting courses in Yuan Ze University.
  
  Attributes:
    account: The account used to log in.
    password: The password used to log in.
    logger: A logger to record the bot's actions.
    processor: A TrOCRProcessor object.
    model: A VisionEncoderDecoderModel object.
    session: A requests session object.
    select_payload: A dictionary containing the select payload.
    usr_course_list: A list of courses to be selected.
    courses_db: A dictionary containing course information.
    boosted: A boolean indicating whether selecting courses more aggressively.
  """

  def __init__(
      self,
      usr_course_list: list,
  ):
    """Initialize the course bot.

    Args:
      usr_course_list: A list of courses to be selected.
    """
    self.account = ""
    self.password = ""
    self.logger = self._init_logger(LOG_DIR)
    self.processor, self.model = self._load_model(
        PRETRAINED_MODEL_NAME,
        PRETRAINED_MODEL_PATH,
    )
    self.session = self._init_session()
    self.select_payload: Dict[str, Any] = {}
    self.usr_course_list = usr_course_list
    self.courses_db: Dict[str, Any] = {}
    self.boosted = True  # for more explanation, see the dynamic_delay method

  def _load_model(
      self, pretrained_model_name, pretrained_model_path
  ) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load the TrOCR model and processor.
    
    Args:
      pretrained_model_name: 
        The name of the pretrained TrOCR model.
      pretrained_model_path: 
        The path to the pretrained TrOCR model.
    Returns:
      A tuple containing the processor and the model.
    """
    try:
      device = "cuda" if torch.cuda.is_available() else "cpu"
      processor = TrOCRProcessor.from_pretrained(
          pretrained_model_name,
          clean_up_tokenization_spaces=True,
          char_level=True,
      )
      model = VisionEncoderDecoderModel.from_pretrained(
          pretrained_model_path).to(device)
      self.logger.info(f"[ 模型載入成功 ] 使用 {pretrained_model_name} 模型!")
      return processor, model

    except Exception as e:
      self.logger.critical(f"[ 模型載入失敗 ] 詳細資訊: {e}", exc_info=DEBUG_MODE)
      sys.exit(0)

  def _get_captcha_text(self, image: Image.Image) -> str:
    """Perform OCR on the captcha image.

    Args:
      image: A PIL image containing the captcha.
    Returns:
      The TrOCR result, which is the text in the captcha, 4 characters long.
    """
    try:
      device = "cuda" if torch.cuda.is_available() else "cpu"
      pixel_values = self.processor(
          image,
          return_tensors="pt",
      ).pixel_values.to(device)
      generated_ids = self.model.generate(pixel_values)
      return self.processor.batch_decode(
          generated_ids,
          skip_special_tokens=True,
      )[0]
    except Exception as e:
      self.logger.critical(
          f"[ 模型辨識錯誤 ] 在辨識驗證碼文字時發生了未知的錯誤! 詳細資訊: {e}",
          exc_info=DEBUG_MODE,
      )
      sys.exit(0)

  def _init_logger(self, path: str) -> object:
    """Initialize the logger object.
    
    Returns:
      A logger object.
    """
    logger = utils.logger(path)
    return logger

  @staticmethod
  def _init_session() -> requests.Session:
    """Initialize the requests session.
    
    Returns:
      A requests session object.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent":
            "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
        "Accept":
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8",
        "Accept-Language":
            "en-US,en;q=0.5",
        "Referer":
            LOGIN_URL_REFERER,
        "Accept-Encoding":
            "gzip, deflate, br, zstd",
        "Upgrade-Insecure-Requests":
            "1",
    })
    return session

  def _init_login_payload(self, **kwargs) -> Dict[str, str]:
    """Initialize the login payload with necessary values.
    
    Args:
      kwargs: Additional key-value pairs to be included in the payload.
    Returns:
      A dictionary containing the login payload.
    """
    payload = {
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "Txt_User": self.account,
        "Txt_Password": self.password,
        "btnOK": "確定"
    }
    payload.update(kwargs)
    return payload

  def _init_select_payload(self, dept_id, **kwargs) -> Dict[str, str]:
    """Initialize the select payload with necessary values.
    
    Args:
      dept_id: Department ID.
      kwargs: Additional key-value pairs to be included in the payload.
    """
    payload = {
        "__EVENTARGUMENT": "",
        "__LASTFOCUS": "",
        "__VIEWSTATEENCRYPTED": "",
        "Hidden1": "",
        "Hid_SchTime": "",
        "DPL_DeptName": dept_id,
        "DPL_Degree": "6"
    }
    payload.update(kwargs)
    return payload

  @staticmethod
  def _clean_alert_msg(response_text: str) -> str:
    """Clean the alert message from the response text.

    The javascript alert message of the course selection system often contains 
    characters like '(r)', '(c)', '(c.)', '\\n', etc. 
    This function removes these characters for better readability.

    Args:
      response_text: The response text containing the alert message.
    Returns:
      The cleaned alert message.
    """
    alert_msg = re.search(r"alert\(['\"](.*?)['\"]\)", response_text).group(1)
    alert_msg = re.sub(r"[()\.\r\\nrc]", "", alert_msg)
    return alert_msg

  @staticmethod
  def _clear_terminal() -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")

  @handle_exceptions()
  def _login(self) -> bool:
    """Log into the course selection system with account, password, and captcha.

    Returns:
      True if login is successful, False otherwise.
    """
    while True:
      self.session.cookies.clear()

      captcha_response = self.session.get(
          CAPTCHA_URL,
          stream=True,
          timeout=REQUEST_TIMEOUT,
      )
      captcha_response.raise_for_status()

      captcha_data = io.BytesIO(captcha_response.content)
      captcha_img = Image.open(captcha_data).convert("RGB")
      captcha_text = self._get_captcha_text(captcha_img)

      login_response = self.session.get(LOGIN_URL)
      login_response.raise_for_status()

      if "選課系統尚未開放" in login_response.text:
        self.logger.info("[ 選課系統尚未開放 ]")
        continue

      parser = BeautifulSoup(login_response.text, "lxml")

      login_payload = self._init_login_payload(
          __VIEWSTATE=parser.select_one("#__VIEWSTATE")["value"],
          __VIEWSTATEGENERATOR=parser.select_one("#__VIEWSTATEGENERATOR")
          ["value"],
          __EVENTVALIDATION=parser.select_one("#__EVENTVALIDATION")["value"],
          DPL_SelCosType=parser.select_one(
              "#DPL_SelCosType option:not([value='00'])")["value"],
          Txt_CheckCode=captcha_text,
      )

      result = self.session.post(
          LOGIN_URL,
          data=login_payload,
          timeout=REQUEST_TIMEOUT,
      )
      result.raise_for_status()

      is_logged_in = self._handle_login_result(result.text)
      if isinstance(is_logged_in, bool):
        return is_logged_in

  def _handle_login_result(self, response_text: str) -> bool:
    """Handle the login result and display messages accordingly.
    
    Args:
      response_text: The response text from the login request.
    Returns:
      True if login is successful, False otherwise.
    """
    succeeded_message = "parent.location ='SelCurr.aspx?Culture=zh-tw'"
    retry_message = "驗證碼錯誤"

    if succeeded_message in response_text:
      self.logger.info("[ 登入成功 ]")
      return True
    elif retry_message in response_text:
      self.logger.warning("[ 登入失敗 ] 驗證碼錯誤， 重新嘗試中...")
      return None
    else:
      alert_msg = self._clean_alert_msg(response_text)
      self.logger.error(f"[ 登入失敗 ] {alert_msg}")
      input("請按任意鍵繼續...")
      self._clear_terminal()
      return False

  @handle_exceptions()
  def _verify_usr_course_list(self) -> list:
    """Verify the user's course list.

    If user type the wrong department ID or course ID, the course will be ignored
    and the program will be continued on.

    Returns:
      A list of verified courses.
    """
    self.logger.info("[ 開始檢查選課清單 ]")
    verified_courses = []

    for option in self.usr_course_list:
      option = option.replace(" ", "")
      usr_dept_id, *rest = option.split(",")
      usr_course_id = ",".join(rest)

      html = self.session.get(
          COURSE_LIST_URL,
          timeout=REQUEST_TIMEOUT,
      )
      html.raise_for_status()
      parser = BeautifulSoup(html.text, "lxml")

      if "異常登入" in html.text:
        self.logger.critical("[ 帳號被阻擋 ] 已被相關單位偵測到頻繁搶課!")
        sys.exit(0)

      sys_dept_id = parser.select(
          f"#DPL_DeptName option[value='{usr_dept_id}']")
      if not sys_dept_id:
        self.logger.warning(
            f"[ 已忽略選項 ] 選項 {option} 錯誤, 系所代號 {usr_dept_id} 不存在!")
        continue
      sys_dept_name = sys_dept_id[0].text

      select_payload = self._init_select_payload(
          usr_dept_id,
          __EVENTTARGET="DPL_Degree",
          __VIEWSTATE=parser.select_one("#__VIEWSTATE")["value"],
          __VIEWSTATEGENERATOR=parser.select_one("#__VIEWSTATEGENERATOR")
          ["value"],
          __EVENTVALIDATION=parser.select_one("#__EVENTVALIDATION")["value"],
      )

      html = self.session.post(
          COURSE_LIST_URL,
          data=select_payload,
          timeout=REQUEST_TIMEOUT,
      )
      html.raise_for_status()
      parser = BeautifulSoup(html.text, "lxml")

      sys_course_id = parser.select(
          f"#CosListTable input[name*='{usr_course_id}']")

      if not sys_course_id:
        self.logger.warning(
            f"[ 已忽略選項 ] 選項 {option} 錯誤, 系所 {sys_dept_name} 查無 {usr_course_id} 課程!",
        )
        continue

      self.select_payload[usr_dept_id] = select_payload

      sys_course_id = sys_course_id[0]
      sys_course_name = sys_course_id.attrs["name"].split(" ")[-1]
      sys_course_info = f"{usr_course_id} {sys_course_name}"
      self.courses_db[usr_course_id] = {
          "info": sys_course_info,
          "mUrl": sys_course_id.attrs["name"]
      }

      verified_courses.append(option)

    if not verified_courses:
      self.logger.error(
          "[ 選課清單為空 ] 在忽略所有錯誤的選項後, 選課清單為空! 請重新檢查 course_list.json!",)
      sys.exit(0)

    self.logger.info("[ 選課清單檢查完成 ]")

    return verified_courses

  @handle_exceptions()
  def _select_courses(self, verified_usr_course_list: list) -> None:
    """Automatically select courses from the verified course list.

    Args:
      verified_usr_course_list: A list of verified courses.
    """
    time_stamp = time.time()
    self.logger.info("[ 開始搶課 ]")
    while verified_usr_course_list:
      for option in verified_usr_course_list.copy():
        usr_dept_id, *rest = option.split(",")
        usr_course_id = ",".join(rest)

        html = self.session.post(
            COURSE_LIST_URL,
            data=self.select_payload[usr_dept_id],
            timeout=REQUEST_TIMEOUT,
        )
        html.raise_for_status()

        parser = BeautifulSoup(html.text, "lxml")
        select_payload = self._init_select_payload(
            usr_dept_id,
            __EVENTTARGET="",
            __VIEWSTATE=parser.select_one("#__VIEWSTATE")["value"],
            __VIEWSTATEGENERATOR=parser.select_one("#__VIEWSTATEGENERATOR")
            ["value"],
            __EVENTVALIDATION=parser.select_one("#__EVENTVALIDATION")["value"],
        )
        m_url = self.courses_db[usr_course_id]["mUrl"]
        select_payload.update({f"{m_url}.x": "0", f"{m_url}.y": "0"})

        response = self.session.post(
            COURSE_LIST_URL,
            data=select_payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        html = self.session.get(
            f"{COURSE_SELECT_URL}{m_url} ,B,",
            timeout=REQUEST_TIMEOUT,
        )
        html.raise_for_status()

        self._handle_select_courses_result(
            html.text,
            verified_usr_course_list,
            option,
            usr_course_id,
        )

        time.sleep(self.dynamic_delay(time_stamp))
    self.logger.info("[ 結束搶課 ] 選課清單中所有可搶課程已搶畢!")

  def _handle_select_courses_result(
      self,
      response_text: str,
      verified_usr_course_list: list,
      option: str,
      usr_course_id: str,
  ) -> None:
    """Handle the result of selecting courses and display messages accordingly.
    
    Args:
      response_text: The response text from the course selection request.
      verified_usr_course_list: A list of verified courses.
      option: The course option.
      usr_course_id: The course ID.
    """
    alert_msg = self._clean_alert_msg(response_text)
    detailed_info = f"{self.courses_db[usr_course_id]['info']} \t{alert_msg}"

    succeeded_messages = ["加選訊息", "已選過", "完成加選"]
    retry_messages = ["開放外系生可加選", "人數已達上限"]
    failed_messages = ["異常查詢課程資訊", "斷線", "逾時", "logged off"]
    critical_messages = ["異常登入"]

    if any(msg in response_text for msg in retry_messages):
      self.logger.info(f"[ 持續搶課中 ] {detailed_info}")
      return
    elif any(msg in response_text for msg in succeeded_messages):
      self.logger.info(f"[ 已成功加選 ] {detailed_info}")
      verified_usr_course_list.remove(option)
      return
    elif any(msg in response_text for msg in failed_messages):
      self.logger.error(f"[ 重新連線中 ] 已由選課系統登出!")
      self._login()
      return
    elif any(msg in response_text for msg in critical_messages):
      self.logger.critical(f"[ 帳號被阻擋 ] {detailed_info}")
      sys.exit(0)
    else:
      self.logger.warning(f"[ 已忽略選項 ] {detailed_info}")
      verified_usr_course_list.remove(option)

  def dynamic_delay(self, time_stamp: float) -> int:
    """Set the frequency of the course selection dynamically.
      
      Args:
        time_stamp: The time stamp when starting to select courses.
    """

    if not self.boosted:
      # not recommended to reduce the delay time, you might get banned lol
      return 3

    elapsed_time = (time.time() - time_stamp) % 60
    if elapsed_time < 5:
      # greedily select courses in the first 5 seconds, no delay time set
      # you can select around 50 to nearly 60 times in the first 5 seconds
      # not recommended to extend more then 5 seconds, you might get banned
      return 0
    else:
      # after being greedy, set the boosted flag to False,
      # then set the delay time to 3 seconds to avoid being banned
      self.boosted = False
      return 3

  def run(self) -> None:
    """Start the course selection process."""
    self._clear_terminal()

    # login
    while True:
      print("YZU Course Bot v2.0")
      print("Please enter your YZU Portal account and password. "
            "Your password will be hidden while typing.")
      self.account = input("Account: ")

      if self.account == "bark":
        print(""" 
            |\_/|                  
            | @ @   Woof! Woof!
            |   <>              _  
            |  _/\------____ ((| |))
            |               `--' |   
        ____|_       ___|   |___.' 
        /_/_____/____/_______|
        """)
        input("\"Stupid Humans!\", doggy said ...")
        self._clear_terminal()
        continue
      elif self.account == "exit":
        sys.exit(0)

      self.password = getpass.getpass(prompt="Password: ")

      if self._login():
        break

    # verify user's course list
    verified_courses = self._verify_usr_course_list()

    # automatically select courses
    self._select_courses(verified_courses)


if __name__ == "__main__":

  try:
    with open("course_list.json", "r", encoding="utf-8") as f:
      usr_course_list = json.load(f)
      USR_COURSE_LIST = usr_course_list["course_list"]
  except FileNotFoundError as e:
    print(f"[ 找不到檔案 ] 請確認 course_list.json 是否存在! 詳細資訊: {e}")
    sys.exit(0)
  except json.JSONDecodeError as e:
    print(f"[ JSON 解析錯誤 ] 請確認 course_list.json 中語法是否正確! 詳細資訊: {e}")
    sys.exit(0)
  except Exception as e:
    print(f"[ 未知的錯誤 ] 詳細資訊: {e}")
    sys.exit(0)

  utils.enable_signal_handler()

  bot = CourseBot(USR_COURSE_LIST)
  bot.run()
