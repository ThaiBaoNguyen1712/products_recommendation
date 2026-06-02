import os
from urllib.parse import quote_plus

import certifi
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

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


def _get_engine_options() -> dict[str, object]:
    options: dict[str, object] = {"pool_pre_ping": True}
    if os.getenv("VERCEL", "").strip() == "1":
        # Serverless functions should not hold open SQL connections between invocations.
        options["poolclass"] = NullPool
    return options


def _split_server_and_port(server_value: str, fallback_port: str) -> tuple[str, str]:
    normalized_server = server_value.strip()
    normalized_port = fallback_port.strip()

    if "," in normalized_server:
        host, port = normalized_server.rsplit(",", 1)
        host = host.strip()
        port = port.strip()
        if host:
            normalized_server = host
        if port:
            normalized_port = port

    return normalized_server, normalized_port


def _build_pyodbc_engine() -> Engine:
    connection_string = os.getenv("DB_CONNECTION_STRING", "").strip()
    db_server = os.getenv("DB_SERVER", "").strip()
    db_name = os.getenv("DB_NAME", "").strip()
    username = os.getenv("DB_USERNAME", "").strip()
    password = os.getenv("DB_PASSWORD", "").strip()
    driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server").strip()
    trusted_connection = os.getenv("DB_TRUSTED_CONNECTION", "").lower() in {"1", "true", "yes"}
    encrypt = os.getenv("DB_ENCRYPT", "yes").strip()
    trust_server_certificate = os.getenv("DB_TRUST_SERVER_CERTIFICATE", "no").strip()
    connect_timeout = os.getenv("DB_CONNECT_TIMEOUT", "5").strip()

    if connection_string:
        normalized_connection_string = _normalize_connection_string(connection_string)
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
                "Database credentials are not configured for pyodbc. Set DB_CONNECTION_STRING, "
                "enable DB_TRUSTED_CONNECTION=true, or provide DB_USERNAME/DB_PASSWORD."
            )

        params = quote_plus(";".join(parts) + ";")

    connection_url = f"mssql+pyodbc:///?odbc_connect={params}"
    return create_engine(connection_url, **_get_engine_options())


def _build_pytds_engine() -> Engine:
    sqlalchemy_url = os.getenv("DB_SQLALCHEMY_URL", "").strip()
    if sqlalchemy_url:
        return create_engine(sqlalchemy_url, **_get_engine_options())

    db_server, db_port = _split_server_and_port(
        server_value=os.getenv("DB_SERVER", ""),
        fallback_port=os.getenv("DB_PORT", "1433"),
    )
    db_name = os.getenv("DB_NAME", "").strip()
    username = os.getenv("DB_USERNAME", "").strip()
    password = os.getenv("DB_PASSWORD", "").strip()
    encrypt = os.getenv("DB_ENCRYPT", "yes").strip().lower() in {"1", "true", "yes"}
    trust_server_certificate = (
        os.getenv("DB_TRUST_SERVER_CERTIFICATE", "no").strip().lower() in {"1", "true", "yes"}
    )
    connect_timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "5").strip())

    if not all([db_server, db_name, username, password]):
        raise RuntimeError(
            "Database credentials are not configured for pytds. Set DB_SQLALCHEMY_URL or provide "
            "DB_SERVER, DB_PORT, DB_NAME, DB_USERNAME, and DB_PASSWORD."
        )

    connection_url = (
        f"mssql+pytds://{quote_plus(username)}:{quote_plus(password)}"
        f"@{db_server}:{db_port}/{db_name}"
    )

    engine_options = _get_engine_options()
    engine_options["connect_args"] = {
        "login_timeout": connect_timeout,
        "timeout": connect_timeout,
    }

    if encrypt:
        engine_options["connect_args"]["cafile"] = certifi.where()
        engine_options["connect_args"]["validate_host"] = not trust_server_certificate
        engine_options["connect_args"]["enc_login_only"] = False

    return create_engine(connection_url, **engine_options)


def _resolve_driver_mode() -> str:
    explicit_mode = os.getenv("DB_RUNTIME_DRIVER", "").strip().lower()
    if explicit_mode in {"pyodbc", "pytds"}:
        return explicit_mode

    if os.getenv("VERCEL", "").strip() == "1":
        return "pytds"

    return "pyodbc"


def create_mssql_engine() -> Engine:
    driver_mode = _resolve_driver_mode()
    if driver_mode == "pytds":
        return _build_pytds_engine()
    return _build_pyodbc_engine()


engine = create_mssql_engine()
