import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def _to_odbc_bool(value: str) -> str:
    return "yes" if value.strip().lower() in {"1", "true", "yes"} else "no"


def _normalize_connection_string(raw_connection_string: str) -> str:
    key_mapping = {
        "driver": "DRIVER",
        "server": "SERVER",
        "data source": "SERVER",
        "initial catalog": "DATABASE",
        "database": "DATABASE",
        "user id": "UID",
        "uid": "UID",
        "password": "PWD",
        "pwd": "PWD",
        "trusted_connection": "Trusted_Connection",
        "integrated security": "Trusted_Connection",
        "encrypt": "Encrypt",
        "trustservercertificate": "TrustServerCertificate",
        "connection timeout": "Connection Timeout",
        "connect timeout": "Connection Timeout",
        "multipleactiveresultsets": "MARS_Connection",
    }
    bool_keys = {
        "Trusted_Connection",
        "Encrypt",
        "TrustServerCertificate",
        "MARS_Connection",
    }
    ignored_keys = {"persist security info"}
    normalized_parts: list[str] = []

    for raw_part in raw_connection_string.split(";"):
        part = raw_part.strip()
        if not part or "=" not in part:
            continue

        raw_key, raw_value = part.split("=", 1)
        raw_key_normalized = raw_key.strip().lower()
        if raw_key_normalized in ignored_keys:
            continue

        normalized_key = key_mapping.get(raw_key_normalized, raw_key.strip())
        normalized_value = raw_value.strip()

        if normalized_key == "DRIVER" and not normalized_value.startswith("{"):
            normalized_value = f"{{{normalized_value}}}"
        elif normalized_key in bool_keys:
            normalized_value = _to_odbc_bool(normalized_value)

        normalized_parts.append(f"{normalized_key}={normalized_value}")

    return ";".join(normalized_parts) + ";"


connection_string = os.getenv("DB_CONNECTION_STRING")
db_server = os.getenv("DB_SERVER")
db_name = os.getenv("DB_NAME")
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
trusted_connection = os.getenv("DB_TRUSTED_CONNECTION", "").lower() in {"1", "true", "yes"}
encrypt = os.getenv("DB_ENCRYPT", "yes")
trust_server_certificate = os.getenv("DB_TRUST_SERVER_CERTIFICATE", "no")
connect_timeout = os.getenv("DB_CONNECT_TIMEOUT", "5")

if connection_string:
    normalized_connection_string = _normalize_connection_string(connection_string.strip())
    params = quote_plus(normalized_connection_string)
else:
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
            "Database credentials are not configured. Set DB_CONNECTION_STRING, "
            "enable DB_TRUSTED_CONNECTION=true, or provide DB_USERNAME/DB_PASSWORD."
        )

    params = quote_plus(";".join(parts) + ";")

connection_url = f"mssql+pyodbc:///?odbc_connect={params}"

engine = create_engine(connection_url)
