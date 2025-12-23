"""
Data collector for Velib station information and status.
Fetches data from Velib open data API and stores in PostgreSQL/TimescaleDB.
"""

from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

import requests
from loguru import logger
from dotenv import load_dotenv

from src.data.database import DatabaseManager

# Load environment variables
load_dotenv()


@dataclass
class StationInfo:
    """Station information data model."""
    station_id: int
    station_code: str
    name: str
    lat: float
    lon: float
    capacity: int


@dataclass
class StationStatus:
    """Station status data model."""
    station_id: int
    num_bikes_available: int
    num_mechanical: int
    num_ebike: int
    num_docks_available: int
    is_installed: bool
    is_returning: bool
    is_renting: bool
    last_reported: int


class VelibAPIClient:
    """Client for interacting with Velib open data API."""
    
    BASE_URL = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole"
    STATION_INFO_URL = f"{BASE_URL}/station_information.json"
    STATION_STATUS_URL = f"{BASE_URL}/station_status.json"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize API client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'VelibPredictor/1.0'
        })
    
    def fetch_station_information(self) -> List[StationInfo]:
        """
        Fetch station information from API.
        
        Returns:
            List of StationInfo objects
            
        Raises:
            requests.RequestException: If API request fails
        """
        logger.info("Fetching station information...")
        
        try:
            response = self.session.get(
                self.STATION_INFO_URL, 
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            stations = []
            for station in data['data']['stations']:
                stations.append(StationInfo(
                    station_id=station['station_id'],
                    station_code=station['stationCode'],
                    name=station['name'],
                    lat=station['lat'],
                    lon=station['lon'],
                    capacity=station['capacity']
                ))
            
            logger.info(f"Fetched {len(stations)} station information records")
            return stations
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch station information: {e}")
            raise
    
    def fetch_station_status(self) -> List[StationStatus]:
        """
        Fetch current station status from API.
        
        Returns:
            List of StationStatus objects
            
        Raises:
            requests.RequestException: If API request fails
        """
        logger.info("Fetching station status...")
        
        try:
            response = self.session.get(
                self.STATION_STATUS_URL,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            statuses = []
            for station in data['data']['stations']:
                # Extract mechanical and ebike counts
                bike_types = station.get('num_bikes_available_types', [])
                num_mechanical = 0
                num_ebike = 0
                
                for bike_type in bike_types:
                    if 'mechanical' in bike_type:
                        num_mechanical = bike_type['mechanical']
                    elif 'ebike' in bike_type:
                        num_ebike = bike_type['ebike']
                
                statuses.append(StationStatus(
                    station_id=station['station_id'],
                    num_bikes_available=station['num_bikes_available'],
                    num_mechanical=num_mechanical,
                    num_ebike=num_ebike,
                    num_docks_available=station['num_docks_available'],
                    is_installed=bool(station['is_installed']),
                    is_returning=bool(station['is_returning']),
                    is_renting=bool(station['is_renting']),
                    last_reported=station['last_reported']
                ))
            
            logger.info(f"Fetched {len(statuses)} station status records")
            return statuses
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch station status: {e}")
            raise


class VelibDataCollector:
    """Main data collector orchestrator."""
    
    def __init__(self, db_manager: DatabaseManager, api_client: VelibAPIClient):
        """
        Initialize data collector.
        
        Args:
            db_manager: Database manager instance
            api_client: API client instance
        """
        self.db = db_manager
        self.api = api_client
    
    def update_station_information(self) -> int:
        """
        Fetch and update station information in database.
        
        This should be run less frequently (daily) as station info rarely changes.
        
        Returns:
            Number of stations updated
        """
        logger.info("Updating station information...")
        
        try:
            stations = self.api.fetch_station_information()
            
            # Upsert into database
            query = """
                INSERT INTO station_information (
                    station_id, station_code, name, lat, lon, capacity, updated_at
                )
                VALUES (%(station_id)s, %(station_code)s, %(name)s, 
                        %(lat)s, %(lon)s, %(capacity)s, CURRENT_TIMESTAMP)
                ON CONFLICT (station_id) 
                DO UPDATE SET
                    station_code = EXCLUDED.station_code,
                    name = EXCLUDED.name,
                    lat = EXCLUDED.lat,
                    lon = EXCLUDED.lon,
                    capacity = EXCLUDED.capacity,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            records = [
                {
                    'station_id': s.station_id,
                    'station_code': s.station_code,
                    'name': s.name,
                    'lat': s.lat,
                    'lon': s.lon,
                    'capacity': s.capacity
                }
                for s in stations
            ]
            
            self.db.execute_many(query, records)
            logger.success(f"Updated {len(stations)} stations in database")
            
            return len(stations)
            
        except Exception as e:
            logger.error(f"Failed to update station information: {e}")
            raise
    
    def collect_station_status(self) -> int:
        """
        Fetch and store current station status.
        
        This should be run every 15 minutes.
        
        Returns:
            Number of status records inserted
        """
        logger.info("Collecting station status...")
        
        try:
            statuses = self.api.fetch_station_status()
            collection_time = datetime.now()
            
            # Insert into database
            query = """
                INSERT INTO station_status (
                    time, station_id, num_bikes_available, num_mechanical, 
                    num_ebike, num_docks_available, is_installed, 
                    is_returning, is_renting, last_reported
                )
                VALUES (
                    %(time)s, %(station_id)s, %(num_bikes_available)s,
                    %(num_mechanical)s, %(num_ebike)s, %(num_docks_available)s,
                    %(is_installed)s, %(is_returning)s, %(is_renting)s,
                    %(last_reported)s
                )
                ON CONFLICT (time, station_id) DO NOTHING
            """
            
            records = [
                {
                    'time': collection_time,
                    'station_id': s.station_id,
                    'num_bikes_available': s.num_bikes_available,
                    'num_mechanical': s.num_mechanical,
                    'num_ebike': s.num_ebike,
                    'num_docks_available': s.num_docks_available,
                    'is_installed': s.is_installed,
                    'is_returning': s.is_returning,
                    'is_renting': s.is_renting,
                    'last_reported': s.last_reported
                }
                for s in statuses
            ]
            
            rows_inserted = self.db.execute_many(query, records)
            logger.success(
                f"Inserted {rows_inserted} status records at {collection_time}"
            )
            
            # Refresh materialized view (optional, can be scheduled separately)
            self.db.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY latest_station_status")
            
            return rows_inserted
            
        except Exception as e:
            logger.error(f"Failed to collect station status: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about recent data collection.
        
        Returns:
            Dictionary with collection statistics
        """
        query = """
            SELECT 
                COUNT(DISTINCT station_id) as stations_with_data,
                COUNT(*) as total_records,
                MIN(time) as oldest_record,
                MAX(time) as newest_record,
                (SELECT COUNT(*) FROM station_information) as total_stations
            FROM station_status
            WHERE time > NOW() - INTERVAL '24 hours'
        """
        
        result = self.db.fetch_one(query)
        return dict(result) if result else {}


def main():
    """Main entry point for data collection."""
    # Initialize components
    db_manager = DatabaseManager()
    api_client = VelibAPIClient()
    collector = VelibDataCollector(db_manager, api_client)
    
    try:
        # Update station information (run less frequently)
        collector.update_station_information()
        
        # Collect current status (run every 15 minutes)
        collector.collect_station_status()
        
        # Print statistics
        stats = collector.get_collection_stats()
        logger.info(f"Collection stats: {stats}")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()