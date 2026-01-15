"""
Database manager for PostgreSQL/TimescaleDB operations.
Handles connection pooling, query execution, and transactions.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg
from dotenv import load_dotenv
from loguru import logger
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

# Load environment variables
load_dotenv()


class DatabaseManager:
    """
    Manages database connections and operations.

    Uses connection pooling for efficient resource management.
    """

    def __init__(
        self,
        min_conn: int = 1,
        max_conn: int = 10,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize database manager with connection pool.

        Args:
            min_conn: Minimum connections in pool
            max_conn: Maximum connections in pool
            host: Database host (default from env)
            port: Database port (default from env)
            database: Database name (default from env)
            user: Database user (default from env)
            password: Database password (default from env)
        """
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", "5432"))
        self.database = database or os.getenv("DB_NAME", "velib")
        self.user = user or os.getenv("DB_USER", "velib_user")
        self.password = password or os.getenv("DB_PASSWORD", "")

        self.min_conn = min_conn
        self.max_conn = max_conn
        self.pool = None

        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the connection pool."""
        try:
            conninfo = (
                f"host={self.host} port={self.port} dbname={self.database} "
                f"user={self.user} password={self.password}"
            )
            self.pool = ConnectionPool(
                conninfo,
                min_size=self.min_conn,
                max_size=self.max_conn,
                kwargs={"row_factory": dict_row},
            )
            logger.info(f"Database pool initialized: {self.database}@{self.host}:{self.port}")
        except psycopg.Error as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a connection from the pool.

        Yields:
            Database connection

        Example:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except psycopg.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self, commit: bool = True):
        """
        Context manager for getting a cursor.

        Args:
            commit: Whether to commit after cursor operations

        Yields:
            Database cursor

        Example:
            with db.get_cursor() as cursor:
                cursor.execute("INSERT INTO table VALUES (%s)", (value,))
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except psycopg.Error as e:
                conn.rollback()
                logger.error(f"Cursor operation failed: {e}")
                raise
            finally:
                cursor.close()

    def execute(self, query: str, params: Optional[tuple] = None, commit: bool = True) -> int:
        """
        Execute a single query (INSERT, UPDATE, DELETE).

        Args:
            query: SQL query string
            params: Query parameters
            commit: Whether to commit transaction

        Returns:
            Number of rows affected

        Example:
            db.execute(
                "INSERT INTO stations (id, name) VALUES (%s, %s)",
                (1, "Station A")
            )
        """
        with self.get_cursor(commit=commit) as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    def execute_many(
        self,
        query: str,
        params_list: List[Dict[str, Any]],
        commit: bool = True,
        page_size: int = 1000,
    ) -> int:
        """
        Execute a query multiple times with different parameters (batch insert).

        Args:
            query: SQL query string
            params_list: List of parameter dictionaries
            commit: Whether to commit transaction
            page_size: Number of records per batch

        Returns:
            Total number of rows affected

        Example:
            db.execute_many(
                "INSERT INTO stations (id, name) VALUES (%(id)s, %(name)s)",
                [{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}]
            )
        """
        if not params_list:
            return 0

        total_rows = 0
        with self.get_cursor(commit=commit) as cursor:
            # Process in batches to avoid memory issues
            for i in range(0, len(params_list), page_size):
                batch = params_list[i : i + page_size]
                for params in batch:
                    cursor.execute(query, params)
                    total_rows += cursor.rowcount

        return total_rows

    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row from database.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Dictionary of column: value or None

        Example:
            result = db.fetch_one(
                "SELECT * FROM stations WHERE id = %s",
                (1,)
            )
        """
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

    def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Fetch all rows from database.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of dictionaries

        Example:
            results = db.fetch_all("SELECT * FROM stations")
        """
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    def bulk_insert(self, table: str, columns: List[str], values: List[tuple]) -> int:
        """
        Fast bulk insert using COPY.

        Args:
            table: Table name
            columns: List of column names
            values: List of value tuples

        Returns:
            Number of rows inserted

        Example:
            db.bulk_insert(
                'station_status',
                ['time', 'station_id', 'num_bikes'],
                [(datetime.now(), 1, 5), (datetime.now(), 2, 10)]
            )
        """
        if not values:
            return 0

        columns_str = ", ".join(columns)
        total_rows = 0

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Use COPY for efficient bulk insert
                with cursor.copy(f"COPY {table} ({columns_str}) FROM STDIN") as copy:
                    for row in values:
                        copy.write_row(row)
                total_rows = len(values)
                conn.commit()

        return total_rows

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table

        Returns:
            True if table exists, False otherwise
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_name = %s
            )
        """
        result = self.fetch_one(query, (table_name,))
        return result["exists"] if result else False

    def get_table_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table.

        Args:
            table_name: Name of the table

        Returns:
            Number of rows
        """
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.fetch_one(query)
        return result["count"] if result else 0

    def close(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.close()
            logger.info("Database pool closed")


# Singleton instance (optional, for convenience)
_db_instance: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """
    Get or create singleton database manager instance.

    Returns:
        DatabaseManager instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance


if __name__ == "__main__":
    # Test database connection
    db = DatabaseManager()

    try:
        # Test query
        result = db.fetch_one("SELECT version()")
        logger.info(f"Connected to: {result['version']}")

        # Check if tables exist
        tables = ["station_information", "station_status"]
        for table in tables:
            exists = db.table_exists(table)
            logger.info(f"Table '{table}' exists: {exists}")

            if exists:
                count = db.get_table_row_count(table)
                logger.info(f"Table '{table}' has {count} rows")

    finally:
        db.close()
