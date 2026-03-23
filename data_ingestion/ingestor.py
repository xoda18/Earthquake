import requests
import datetime
import csv
import os
import re

def get_post_event_context(hours_back=24):
    """
    Fetches recent quakes to provide context for the Digital Twin's
    structural health monitoring agents.
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    # Calculate time window
    starttime = (datetime.datetime.now() - datetime.timedelta(hours=hours_back)).isoformat()

    params = {
        "format": "geojson",
        "starttime": starttime,
        "minmagnitude": 1.0
    }

    response = requests.get(url, params=params)
    events = response.json()['features']

    # In JASS 2026, this would be formatted as an 'Context' for an LLM Agent
    report = f"Seismic Report for last {hours_back}h:\n"
    for eq in events:
        mag = eq['properties']['mag']
        place = eq['properties']['place']
        time = datetime.datetime.fromtimestamp(eq['properties']['time']/1000)
        report += f"- M{mag} at {place} ({time})\n"

    return report


def _parse_event(feature):
    """Converts a USGS GeoJSON feature into a CSV-compatible dict."""
    props = feature['properties']
    coords = feature['geometry']['coordinates']  # [lon, lat, depth]

    lon, lat, depth = coords[0], coords[1], coords[2]
    mag = props['mag']
    timestamp = datetime.datetime.fromtimestamp(props['time'] / 1000, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    place_raw = props.get('place', '')

    # Parse "X km DIR of Place, Region" format
    distance_km, distance_miles, nearest_place, district = '', '', place_raw, ''
    match = re.match(r'^([\d.]+)\s+km\s+\w+\s+of\s+(.+)$', place_raw)
    if match:
        distance_km = float(match.group(1))
        distance_miles = round(distance_km * 0.621371, 1)
        location_parts = match.group(2).split(',')
        nearest_place = location_parts[0].strip()
        district = location_parts[1].strip() if len(location_parts) > 1 else ''

    return {
        'timestamp_utc': timestamp,
        'magnitude': mag,
        'depth_km': depth,
        'nearest_place': nearest_place,
        'district': district,
        'country': '',
        'latitude': lat,
        'longitude': lon,
        'distance_from_place_km': distance_km,
        'distance_from_place_miles': distance_miles,
    }


def fetch_events(hours_back=24):
    """Returns raw USGS GeoJSON features for the given time window."""
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    starttime = (datetime.datetime.now() - datetime.timedelta(hours=hours_back)).isoformat()
    params = {
        "format": "geojson",
        "starttime": starttime,
        "minmagnitude": 1.0
    }
    response = requests.get(url, params=params)
    return response.json()['features']


FIELDNAMES = [
    'timestamp_utc', 'magnitude', 'depth_km', 'nearest_place',
    'district', 'country', 'latitude', 'longitude',
    'distance_from_place_km', 'distance_from_place_miles'
]


LIVE_CSV = os.path.join(os.path.dirname(__file__), '..', 'earthquake_data_live.csv')


def write_to_csv(features, output_path=LIVE_CSV):
    """
    Appends new events to output_path (default: earthquake_data_live.csv).
    Skips duplicates based on timestamp_utc + latitude + longitude.
    """
    output_path = os.path.normpath(output_path)

    # Load existing keys to detect duplicates
    existing_keys = set()
    if os.path.exists(output_path):
        for row in read_csv(output_path):
            existing_keys.add((row['timestamp_utc'], row['latitude'], row['longitude']))

    new_rows = []
    for f in features:
        row = _parse_event(f)
        key = (row['timestamp_utc'], str(row['latitude']), str(row['longitude']))
        if key not in existing_keys:
            new_rows.append(row)
            existing_keys.add(key)

    if not new_rows:
        print("No new events to write.")
        return output_path

    write_header = not os.path.exists(output_path)
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"Appended {len(new_rows)} new events to {output_path}")
    return output_path


def read_csv(path):
    """Reads a earthquake CSV file and returns a list of dicts."""
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    print(get_post_event_context())

    events = fetch_events(hours_back=24)
    write_to_csv(events)
