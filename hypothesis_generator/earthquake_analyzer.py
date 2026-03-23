"""
JASS Earthquake Analysis Module

A standalone Python module that periodically analyzes earthquake data
and writes findings to the JASS ORB blackboard.

Not an API - runs as a scheduled process.
"""

import time
import os
import csv
import json
import requests
from collections import Counter
from datetime import datetime
from typing import List, Dict
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from pydantic import BaseModel


# Load .env into environment (if present)
load_dotenv(dotenv_path="/Users/satya/Desktop/JASS26/Code/JASS-ORB/JASS-Earthquake/.env")

# Hugging Face token (from environment or .env)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Warning: HF_TOKEN not found in environment. Set HF_TOKEN in your .env or environment variables to enable LLM calls.")


# ------------------------------
# Configuration
# ------------------------------

SERVICE_NAME = "jass_earthquake_analysis"
AGENT_NAME = "earthquake_agent"
CSV_FILE_PATH = "/Users/satya/Desktop/JASS26/Code/JASS-ORB/JASS-Earthquake/earthquake_data.csv"
ANALYSIS_INTERVAL_SECONDS = 60  # Run analysis every 5 minutes (adjust as needed)

# Track registration status
already_registered = False


# ------------------------------
# ORB Communication
# ------------------------------

def register_with_orb():
    """
    Register this module with JASS ORB (one-time operation).
    Instead of checking a local flag file, query ORB `/list_services`.
    If `SERVICE_NAME` already exists, consider it registered.
    """
    def orb_is_running() -> bool:
        """Return True if ORB responds to a basic status GET."""
        try:
            r = requests.get(f"{ORB_URL}/get_status", timeout=3)
            if r.status_code == 200:
                data = r.json()
                if data.get("err_code") == 0 and data.get("orb") == "running":
                    return True
                return False
        except requests.exceptions.RequestException:
            print(f"✗ Could not connect to ORB at {ORB_URL}")
            return False

    try:
        # Check ORB is running first
        if not orb_is_running():
            print(f"✗ ORB does not appear to be running at {ORB_URL}")
            return False

        # Query ORB for existing services
        resp = requests.get(f"{ORB_URL}/list_services", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            services = data.get("services") or {}
            if SERVICE_NAME in services:
                print(f"✓ Service '{SERVICE_NAME}' already registered with ORB")
                return True
        else:
            print(f"Warning: could not list services (HTTP {resp.status_code}); will attempt registration")

        # Not present — attempt registration
        response = requests.post(
            f"{ORB_URL}/register_service",
            json={
                "service_name": SERVICE_NAME,
                "agent": AGENT_NAME,
                "endpoint": "",  # Not an API, so no endpoint
                "description": "Paphos earthquake data analysis module - processes seismic CSV data and generates geological insights"
            },
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("err_code") == 0:
                print(f"✓ Successfully registered with JASS ORB as '{SERVICE_NAME}'")
                return True
            else:
                print(f"✗ ORB returned error during registration: {result}")
                return False
        else:
            print(f"✗ Failed to register with ORB: HTTP {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ Error connecting to ORB at {ORB_URL}: {e}")
        return False


def write_to_orb_blackboard(content: str, entry_type: str = "seismic_analysis", confidence: float = 0.88):
    """
    Write analysis results to ORB's blackboard
    """
    try:
        response = requests.post(
            f"{ORB_URL}/blackboard",
            json={
                "agent": SERVICE_NAME,
                "type": entry_type,
                "content": content,
                "confidence": confidence,
                "timestamp": time.time()
            },
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("err_code") == 0:
                print(f"✓ Successfully wrote to ORB blackboard")
                return True
        
        print(f"✗ Failed to write to blackboard: HTTP {response.status_code}")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error writing to blackboard: {e}")
        return False


# ------------------------------
# CSV Data Reading
# ------------------------------

def read_earthquake_csv():
    """
    Read and parse Paphos earthquake data from CSV file
    Returns list of earthquake dictionaries
    """
    if not os.path.exists(CSV_FILE_PATH):
        print(f"✗ CSV file not found: {CSV_FILE_PATH}")
        return []

    earthquakes = []
    try:
        with open(CSV_FILE_PATH, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if not row:
                    continue
                try:
                    earthquakes.append({
                        'timestamp': row.get('timestamp_utc', ''),
                        'magnitude': float(row.get('magnitude', 0)),
                        'depth_km': float(row.get('depth_km', 0)),
                        'nearest_place': row.get('nearest_place', ''),
                        'district': row.get('district', ''),
                        'country': row.get('country', ''),
                        'latitude': float(row.get('latitude', 0)),
                        'longitude': float(row.get('longitude', 0)),
                        'distance_from_place_km': float(row.get('distance_from_place_km', 0))
                    })
                except (ValueError, TypeError) as e:
                    # Skip malformed rows
                    continue
                    
    except Exception as e:
        print(f"✗ Error reading CSV file: {e}")
        return []

    print(f"✓ Loaded {len(earthquakes)} earthquake records from {CSV_FILE_PATH}")
    return earthquakes


# ------------------------------
# Data Analysis
# ------------------------------

def stats_summary_for_ai(earthquakes: List[Dict]) -> Dict:
    magnitudes = [e['magnitude'] for e in earthquakes]
    depths = [e['depth_km'] for e in earthquakes]
    locations = [e['nearest_place'] for e in earthquakes]
    countries = [e['country'] for e in earthquakes]
    
    avg_magnitude = sum(magnitudes) / len(magnitudes)
    max_magnitude = max(magnitudes)
    min_magnitude = min(magnitudes)
    avg_depth = sum(depths) / len(depths)
    
    # Location analysis
    location_counts = Counter(locations)
    top_location = location_counts.most_common(1)[0] if location_counts else ("Unknown", 0)
    
    country_counts = Counter(countries)
    cyprus_count = country_counts.get('Cyprus', 0)
    turkey_count = country_counts.get('Turkey', 0)
    
    # Temporal analysis
    years = [e['timestamp'][:4] for e in earthquakes if e.get('timestamp')]
    year_counts = Counter(years)
    most_active_year = year_counts.most_common(1)[0] if year_counts else ("Unknown", 0)
    
    # Risk assessment
    if max_magnitude >= 7.0:
        risk_level = "HIGH"
    elif max_magnitude >= 6.0:
        risk_level = "ELEVATED"
    elif max_magnitude >= 5.0:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW-MODERATE"
        
    # Just pick the 3 biggest earthquakes to give the LLM some "flavor"
    top_3_events = sorted(earthquakes, key=lambda x: x['magnitude'], reverse=True)[:3]
    
    return {
        "avg_magnitude": avg_magnitude,
        "max_magnitude": max_magnitude,
        "min_magnitude": min_magnitude,
        "avg_depth": avg_depth,
        "top_location": top_location,
        "cyprus_count": cyprus_count,
        "turkey_count": turkey_count,
        "most_active_year": most_active_year,
        "risk_level": risk_level,
        "top_3_events": top_3_events
    }
    

def get_hf_hypothesis(stats_dict):
    # Initialize the client (Get a free token from huggingface.co/settings/tokens)
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set. Please add HF_TOKEN to your environment or .env file before running LLM calls.")

    client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", 
                             token=HF_TOKEN)
    
    top_events = ", ".join(f"{e.get('magnitude')}M at {e.get('nearest_place')} on {e.get('timestamp')}" for e in stats_dict.get("top_3_events", [])
)

    # Create a cleaner data string for the prompt
    data_summary = f"""
    - Magnitude Range: {stats_dict['min_magnitude']} to {stats_dict['max_magnitude']} (Avg: {stats_dict['avg_magnitude']})
    - Average Depth: {stats_dict['avg_depth']} km
    - Geospatial Focus: {stats_dict['cyprus_count']} events in Cyprus, {stats_dict['turkey_count']} in Turkey
    - Primary Hotspot: {stats_dict['top_location']}
    - Most Active Year: {stats_dict['most_active_year']}
    - Calculated Risk: {stats_dict['risk_level']}
    - Representative Events: {top_events}
    """
    user_prompt = f"""
    Using the following earthquake statistics, generate a formal 1-paragraph report:

    DATA SUMMARY:
    {data_summary}

    REQUIREMENTS:
    1. GEOLOGICAL INTERPRETATION: Explain the specific tectonic drivers. Mention the subduction of the African Plate beneath the Anatolian Plate (Cyprus Arc) and how the average depth of {stats_dict['avg_depth']}km relates to crustal vs. subduction zone activity.
    2. HYPOTHESIS: Based on the {stats_dict['max_magnitude']} max magnitude and frequency in {stats_dict['top_location']}, hypothesize if we are seeing a "seismic swarm" or a buildup of "interseismic strain."
    3. RISK ASSESSMENT: Validate the calculated risk level of '{stats_dict['risk_level']}'. Does the data suggest an imminent threat to local infrastructure?

    Do not use any special formatting in the response - just plain text. Be concise but informative.
    """
    
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a Senior Seismologist and Tectonic Analyst specializing in Eastern Mediterranean geodynamics."},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def analyze_earthquake_data(earthquakes: List[Dict]) -> Dict:
    """
    Analyze earthquake data and generate hypothesis
    
    This is where you would integrate your LLM analysis.
    Currently uses statistical analysis - replace with LLM API calls.
    
    Args:
        earthquakes: List of earthquake event dictionaries
        
    Returns:
        Dictionary containing analysis results and hypothesis
    """
    if not earthquakes:
        return {
            "status": "no_data",
            "hypothesis": "No earthquake data available for analysis",
            "error": True
        }
    
    # Statistical Analysis

    # get statistics summary
    stats_for_ai = stats_summary_for_ai(earthquakes)
    
    # generate LLM hypothesis based on the analysis results
    hypothesis = get_hf_hypothesis(stats_for_ai) # string for now
    
    return hypothesis


# ------------------------------
# Main Analysis Loop
# ------------------------------


def run_analysis_cycle():
    """
    Run one cycle of earthquake analysis
    1. Read CSV data
    2. Analyze data
    3. Write hypothesis to ORB blackboard
    """
    print(f"\n{'='*70}")
    print(f"ANALYSIS CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # Step 1: Read earthquake data
    earthquakes = read_earthquake_csv()
    
    if not earthquakes:
        print("No data to analyze. Skipping this cycle.")
        return False
    
    # Step 2: Analyze data (use LLM)
    print(f"Step 2: Analyzing {len(earthquakes)} earthquake events...")
    hypothesis = analyze_earthquake_data(earthquakes) # returns a dictionary
    
    # Step 3: Write to ORB blackboard
    print("Step 3: Writing hypothesis to ORB blackboard...")
    success = write_to_orb_blackboard(hypothesis)
    
    if success:
        return True
    else:
        print("Failed to write to ORB blackboard")
        return False


def main():
    """
    Main function - initializes and runs the periodic analysis loop
    """
    print("="*70)
    print("JASS EARTHQUAKE ANALYZER MODULE")
    print("="*70)
    print(f"Configuration:")
    print(f"  - ORB URL: {ORB_URL}")
    print(f"  - CSV File: {CSV_FILE_PATH}")
    print(f"  - Analysis Interval: {ANALYSIS_INTERVAL_SECONDS} seconds")
    print("="*70)
    
    print("\n[INITIALIZATION]")
    print("Registering with JASS ORB...")
    registration_success = register_with_orb() # register with ORB (one-time operation)
    
    if not registration_success:
        print("\nFailed to register with ORB")
    else:
        print("\n Initialization complete. Starting analysis loop...")
        pass

        # Step 2: Run periodic analysis loop
        print(f"\n[STARTING ANALYSIS LOOP]")
        print(f"Analysis will run every {ANALYSIS_INTERVAL_SECONDS} seconds")
        
        cycle_count = 0
        
        # run_analysis_cycle() # for testing
        
        try:
            while True:
                cycle_count += 1
                
                # Run analysis
                run_analysis_cycle()
                
                # Wait for next cycle
                print(f"\nWaiting {ANALYSIS_INTERVAL_SECONDS} seconds until next analysis...")
                time.sleep(ANALYSIS_INTERVAL_SECONDS)
                
        except Exception as e:
            print(f"\n Error occurred in main loop: {e}")   


if __name__ == "__main__":
    main()
