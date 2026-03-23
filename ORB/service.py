import requests

ORB_URL = "https://crooked-jessenia-nongenerating.ngrok-free.dev"
SERVICE_NAME = "jass_earthquake_analysis"
AGENT_NAME = "earthquake_agent"

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
