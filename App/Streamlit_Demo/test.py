from dotenv import load_dotenv
import os

load_dotenv()
URL = os.getenv("URL")
print(f"URL='{URL}'")  # kiểm tra xem có ký tự lạ hoặc trống không
