import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv
load_dotenv()

# Lấy biến môi trường
db_server = os.getenv("DB_SERVER")
db_name = os.getenv("DB_NAME")
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")

params = quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={db_server};"
    f"DATABASE={db_name};"
    f"UID={username};"
    f"PWD={password};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
)

connection_url = f"mssql+pyodbc:///?odbc_connect={params}"

engine = create_engine(connection_url)

print("SERVER:", db_server)
print("USERNAME:", username)