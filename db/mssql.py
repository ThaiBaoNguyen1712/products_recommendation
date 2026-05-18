import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

db_server = os.getenv("DB_SERVER")
db_name = os.getenv("DB_NAME")
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
trusted_connection = os.getenv("DB_TRUSTED_CONNECTION", "").lower() in {"1", "true", "yes"}
encrypt = os.getenv("DB_ENCRYPT", "yes")
trust_server_certificate = os.getenv("DB_TRUST_SERVER_CERTIFICATE", "no")
connect_timeout = os.getenv("DB_CONNECT_TIMEOUT", "5")

parts = [
    f"DRIVER={{{driver}}}",
    f"SERVER={db_server}",
    f"DATABASE={db_name}",
    f"Encrypt={encrypt}",
    f"TrustServerCertificate={trust_server_certificate}",
    f"Connection Timeout={connect_timeout}",
]

if trusted_connection:
    parts.append("Trusted_Connection=yes")
elif username and password:
    parts.append(f"UID={username}")
    parts.append(f"PWD={password}")
else:
    raise RuntimeError(
        "Database credentials are not configured. Set DB_TRUSTED_CONNECTION=true "
        "for local SQL Server or provide DB_USERNAME/DB_PASSWORD."
    )

params = quote_plus(";".join(parts) + ";")
connection_url = f"mssql+pyodbc:///?odbc_connect={params}"

engine = create_engine(connection_url)
