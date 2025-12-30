"""
Weather data collector using Open-Meteo API.
Fetches current weather data for Paris and stores in database.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from loguru import logger
from pydantic import BaseModel

from src.data.database import DatabaseManager


class WeatherData(BaseModel):
    """Weather data model."""

    time: datetime
    temperature: float
    apparent_temperature: Optional[float] = None
    precipitation: float
    rain: float
    snowfall: float
    wind_speed: Optional[float] = None
    wind_direction: Optional[int] = None
    wind_gusts: Optional[float] = None
    pressure: Optional[float] = None
    humidity: Optional[int] = None
    cloud_cover: Optional[int] = None
    weather_code: Optional[int] = None


class OpenMeteoClient:
    """Client for Open-Meteo API."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # Paris coordinates
    PARIS_LAT = 48.8566
    PARIS_LON = 2.3522

    def __init__(self, timeout: int = 30):
        """
        Initialize Open-Meteo client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "VelibPredictor/1.0"})

    def fetch_current_weather(self) -> WeatherData:
        """
        Fetch current weather for Paris.

        Returns:
            WeatherData object

        Raises:
            requests.RequestException: If API request fails
        """
        logger.info("Fetching current weather from Open-Meteo...")

        params = {
            "latitude": self.PARIS_LAT,
            "longitude": self.PARIS_LON,
            "current": [
                "temperature_2m",
                "apparent_temperature",
                "precipitation",
                "rain",
                "snowfall",
                "weather_code",
                "cloud_cover",
                "pressure_msl",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "relative_humidity_2m",
            ],
            "timezone": "Europe/Paris",
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            current = data["current"]

            # Parse time (ISO 8601 format)
            weather_time = datetime.fromisoformat(current["time"].replace("Z", "+00:00"))

            weather = WeatherData(
                time=weather_time,
                temperature=current["temperature_2m"],
                apparent_temperature=current.get("apparent_temperature"),
                precipitation=current.get("precipitation", 0),
                rain=current.get("rain", 0),
                snowfall=current.get("snowfall", 0),
                wind_speed=current.get("wind_speed_10m"),
                wind_direction=current.get("wind_direction_10m"),
                wind_gusts=current.get("wind_gusts_10m"),
                pressure=current.get("pressure_msl"),
                humidity=current.get("relative_humidity_2m"),
                cloud_cover=current.get("cloud_cover"),
                weather_code=current.get("weather_code"),
            )

            logger.info(
                f"Fetched weather: {weather.temperature}°C, "
                f"{weather.precipitation}mm precipitation"
            )

            return weather

        except requests.RequestException as e:
            logger.error(f"Failed to fetch weather: {e}")
            raise

    def fetch_historical_weather(
        self, start_date: datetime, end_date: datetime
    ) -> List[WeatherData]:
        """
        Fetch historical weather data.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of WeatherData objects

        Raises:
            requests.RequestException: If API request fails
        """
        logger.info(f"Fetching historical weather from {start_date} to {end_date}...")

        params = {
            "latitude": self.PARIS_LAT,
            "longitude": self.PARIS_LON,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": [
                "temperature_2m",
                "apparent_temperature",
                "precipitation",
                "rain",
                "snowfall",
                "weather_code",
                "cloud_cover",
                "pressure_msl",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "relative_humidity_2m",
            ],
            "timezone": "Europe/Paris",
        }

        # Use historical endpoint for past data
        url = "https://archive-api.open-meteo.com/v1/archive"

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            hourly = data["hourly"]
            weather_records = []

            # Parse all hourly records
            for i in range(len(hourly["time"])):
                weather_time = datetime.fromisoformat(hourly["time"][i].replace("Z", "+00:00"))

                weather = WeatherData(
                    time=weather_time,
                    temperature=hourly["temperature_2m"][i],
                    apparent_temperature=hourly.get("apparent_temperature", [None])[i],
                    precipitation=hourly.get("precipitation", [0])[i] or 0,
                    rain=hourly.get("rain", [0])[i] or 0,
                    snowfall=hourly.get("snowfall", [0])[i] or 0,
                    wind_speed=hourly.get("wind_speed_10m", [None])[i],
                    wind_direction=hourly.get("wind_direction_10m", [None])[i],
                    wind_gusts=hourly.get("wind_gusts_10m", [None])[i],
                    pressure=hourly.get("pressure_msl", [None])[i],
                    humidity=hourly.get("relative_humidity_2m", [None])[i],
                    cloud_cover=hourly.get("cloud_cover", [None])[i],
                    weather_code=hourly.get("weather_code", [None])[i],
                )
                weather_records.append(weather)

            logger.info(f"Fetched {len(weather_records)} historical weather records")
            return weather_records

        except requests.RequestException as e:
            logger.error(f"Failed to fetch historical weather: {e}")
            raise


class WeatherCollector:
    """Weather data collector orchestrator."""

    def __init__(self, db_manager: DatabaseManager, api_client: OpenMeteoClient):
        """
        Initialize weather collector.

        Args:
            db_manager: Database manager instance
            api_client: Open-Meteo API client instance
        """
        self.db = db_manager
        self.api = api_client

    def collect_current_weather(self) -> int:
        """
        Fetch and store current weather.

        Returns:
            Number of records inserted (1 or 0)
        """
        logger.info("Collecting current weather...")

        try:
            weather = self.api.fetch_current_weather()

            # Insert into database
            query = """
                INSERT INTO weather_data (
                    time, temperature, apparent_temperature, precipitation,
                    rain, snowfall, wind_speed, wind_direction, wind_gusts,
                    pressure, humidity, cloud_cover, weather_code
                )
                VALUES (
                    %(time)s, %(temperature)s, %(apparent_temperature)s,
                    %(precipitation)s, %(rain)s, %(snowfall)s,
                    %(wind_speed)s, %(wind_direction)s, %(wind_gusts)s,
                    %(pressure)s, %(humidity)s, %(cloud_cover)s, %(weather_code)s
                )
                ON CONFLICT (time) DO UPDATE SET
                    temperature = EXCLUDED.temperature,
                    apparent_temperature = EXCLUDED.apparent_temperature,
                    precipitation = EXCLUDED.precipitation,
                    rain = EXCLUDED.rain,
                    snowfall = EXCLUDED.snowfall,
                    wind_speed = EXCLUDED.wind_speed,
                    wind_direction = EXCLUDED.wind_direction,
                    wind_gusts = EXCLUDED.wind_gusts,
                    pressure = EXCLUDED.pressure,
                    humidity = EXCLUDED.humidity,
                    cloud_cover = EXCLUDED.cloud_cover,
                    weather_code = EXCLUDED.weather_code
            """

            record = weather.model_dump()

            rows = self.db.execute(query, record)
            logger.success(f"Stored weather data for {weather.time}")

            return rows

        except Exception as e:
            logger.error(f"Failed to collect weather: {e}")
            raise

    def backfill_historical_weather(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> int:
        """
        Backfill historical weather data.

        Args:
            start_date: Start date (default: earliest station data)
            end_date: End date (default: yesterday)

        Returns:
            Number of records inserted
        """
        logger.info("Backfilling historical weather...")

        try:
            # Get date range from station_status if not provided
            if start_date is None:
                result = self.db.fetch_one("SELECT MIN(DATE(time)) as min_date FROM station_status")
                start_date = result["min_date"] if result else datetime.now().date()

            if end_date is None:
                end_date = (datetime.now() - timedelta(days=1)).date()

            # Ensure datetime objects
            if not isinstance(start_date, datetime):
                start_date = datetime.combine(start_date, datetime.min.time())
            if not isinstance(end_date, datetime):
                end_date = datetime.combine(end_date, datetime.min.time())

            logger.info(f"Backfilling from {start_date} to {end_date}")

            # Fetch historical data
            weather_records = self.api.fetch_historical_weather(start_date, end_date)

            if not weather_records:
                logger.warning("No historical weather data found")
                return 0

            # Bulk insert
            query = """
                INSERT INTO weather_data (
                    time, temperature, apparent_temperature, precipitation,
                    rain, snowfall, wind_speed, wind_direction, wind_gusts,
                    pressure, humidity, cloud_cover, weather_code
                )
                VALUES (
                    %(time)s, %(temperature)s, %(apparent_temperature)s,
                    %(precipitation)s, %(rain)s, %(snowfall)s,
                    %(wind_speed)s, %(wind_direction)s, %(wind_gusts)s,
                    %(pressure)s, %(humidity)s, %(cloud_cover)s, %(weather_code)s
                )
                ON CONFLICT (time) DO NOTHING
            """

            records = [w.model_dump() for w in weather_records]

            rows_inserted = self.db.execute_many(query, records)
            logger.success(f"Backfilled {rows_inserted} weather records")

            return rows_inserted

        except Exception as e:
            logger.error(f"Failed to backfill weather: {e}")
            raise

    def get_weather_stats(self) -> Dict:
        """
        Get statistics about weather data collection.

        Returns:
            Dictionary with weather statistics
        """
        query = """
            SELECT 
                COUNT(*) as total_records,
                MIN(time) as oldest_record,
                MAX(time) as newest_record,
                AVG(temperature) as avg_temperature,
                SUM(CASE WHEN precipitation > 0 THEN 1 ELSE 0 END) as rainy_periods
            FROM weather_data
            WHERE time > NOW() - INTERVAL '7 days'
        """

        result = self.db.fetch_one(query)
        return dict(result) if result else {}


def main():
    """Main entry point for weather collection."""
    db_manager = DatabaseManager()
    api_client = OpenMeteoClient()
    collector = WeatherCollector(db_manager, api_client)

    try:
        # Collect current weather
        collector.collect_current_weather()

        # Print statistics
        stats = collector.get_weather_stats()
        logger.info(f"Weather stats: {stats}")

    except Exception as e:
        logger.error(f"Weather collection failed: {e}")
        raise
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
