#db/mssql.py

from sqlalchemy import create_engine
from urllib.parse import quote_plus
import env
db_server = env.db_server
db_name = env.db_name
username = env.username
password = quote_plus(env.password)

# ===== 1. CONNECT DATABASE =====
engine = create_engine(
    f"mssql+pyodbc://{username}:{password}@{db_server}/{db_name}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)
