"""
Fetch a comprehensive list of car makes / models / years from the
NHTSA vPIC API (free, no API key required) and write them to
car_classes.json (which config.py auto-loads).

The NHTSA (National Highway Traffic Safety Administration) maintains a
public Vehicle Product Information Catalog with every vehicle sold in
the United States.

API docs: https://vpic.nhtsa.dot.gov/api/

Usage
-----
    # Generate car classes for popular makes, years 2018-2025
    python tools/fetch_car_classes.py

    # Custom year range
    python tools/fetch_car_classes.py --years 2020 2025

    # Only specific makes
    python tools/fetch_car_classes.py --makes Toyota Honda BMW

    # Include ALL makes (warning: huge list, slow)
    python tools/fetch_car_classes.py --all-makes --years 2023 2024

    # Preview only — don't write anything
    python tools/fetch_car_classes.py --dry-run

    # Limit to "Passenger Car" vehicle type only (exclude trucks, SUVs, etc.)
    python tools/fetch_car_classes.py --car-type passenger
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ──────────────────────────────────────────────
# NHTSA API base URL
# ──────────────────────────────────────────────
BASE_URL = "https://vpic.nhtsa.dot.gov/api/vehicles"

# ──────────────────────────────────────────────
# Popular makes (curated for practical training)
# ──────────────────────────────────────────────
POPULAR_MAKES = [
    # Japanese
    "Toyota", "Honda", "Nissan", "Mazda", "Subaru",
    "Lexus", "Acura", "Infiniti", "Mitsubishi",
    # American
    "Ford", "Chevrolet", "GMC", "Dodge", "Ram",
    "Jeep", "Cadillac", "Lincoln", "Buick", "Chrysler",
    "Tesla",
    # German
    "BMW", "Mercedes-Benz", "Audi", "Volkswagen", "Porsche",
    # Korean
    "Hyundai", "Kia", "Genesis",
    # European
    "Volvo", "Land Rover", "Jaguar", "Mini", "Fiat",
    "Alfa Romeo", "Maserati", "Ferrari", "Lamborghini",
    "Bentley", "Rolls-Royce", "Aston Martin", "McLaren",
]

# Vehicle types to include (NHTSA VehicleTypeName values)
VEHICLE_TYPE_FILTERS = {
    "passenger": [
        "Passenger Car",
    ],
    "suv_truck": [
        "Multipurpose Passenger Vehicle (MPV)",
        "Truck",
    ],
    "all": None,  # no filtering
}


# ──────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET request to the NHTSA API with retry logic."""
    url = f"{BASE_URL}/{endpoint}"
    params = params or {}
    params["format"] = "json"

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [retry] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  [error] Failed after {retries} attempts: {e}")
                return {"Results": []}


def get_all_makes() -> list[str]:
    """Fetch every make known to NHTSA."""
    print("Fetching all makes from NHTSA...")
    data = _get("GetAllMakes")
    makes = [r["Make_Name"] for r in data.get("Results", [])]
    print(f"  Found {len(makes)} total makes")
    return makes


def get_models_for_make_year(make: str, year: int) -> list[dict]:
    """
    Fetch models for a given make and model year.
    Returns list of dicts with keys: Make_Name, Model_Name, VehicleTypeName, etc.
    """
    data = _get(f"GetModelsForMakeYear/make/{make}/modelyear/{year}/vehicletype/car")
    results = data.get("Results", [])

    # Also fetch MPV (SUVs/crossovers) — many popular vehicles are categorized here
    data_mpv = _get(
        f"GetModelsForMakeYear/make/{make}/modelyear/{year}"
        f"/vehicletype/multipurpose passenger vehicle (mpv)"
    )
    results.extend(data_mpv.get("Results", []))

    return results


# ──────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
    """
    Clean up model / make names from the NHTSA API.

    - Strips extra whitespace
    - Title-cases ALL-CAPS names
    - Removes problematic characters (quotes, backslashes, etc.) that
      would break JSON or Python string literals
    """
    name = " ".join(name.split())

    # Remove characters that are unsafe in JSON string values or file paths
    name = name.replace('"', '')
    name = name.replace("'", '')
    name = name.replace("\\", '')

    # Title-case ALL-CAPS names (e.g. "MUSTANG" -> "Mustang")
    if name.isupper():
        name = name.title()

    # Strip any remaining leading/trailing whitespace from cleanup
    name = name.strip()
    return name


def _is_valid_entry(make: str, model: str) -> bool:
    """Filter out junk entries from NHTSA."""
    if not make or not model:
        return False
    # Skip entries that are just numbers or very short garbage
    if len(model) < 2:
        return False
    # Skip entries with only special characters
    if re.match(r'^[\W\d]+$', model):
        return False
    return True


def fetch_car_classes(
    makes: list[str],
    year_start: int,
    year_end: int,
    car_type: str = "all",
) -> list[dict]:
    """
    Fetch car classes from the NHTSA API.

    Returns a sorted list of dicts:
        [{"label": "Toyota Camry 2023", "make": "Toyota", "model": "Camry", "year": "2023"}, ...]
    """
    type_filter = VEHICLE_TYPE_FILTERS.get(car_type)
    classes = []
    seen = set()

    years = list(range(year_start, year_end + 1))

    for make in tqdm(makes, desc="Makes"):
        for year in years:
            results = get_models_for_make_year(make, year)

            for r in results:
                model_name = _sanitize_name(r.get("Model_Name", ""))
                make_name = _sanitize_name(r.get("Make_Name", make))

                if not _is_valid_entry(make_name, model_name):
                    continue

                # Apply vehicle type filter
                if type_filter:
                    vtype = r.get("VehicleTypeName", "")
                    if vtype not in type_filter:
                        continue

                key = (make_name, model_name, str(year))
                if key in seen:
                    continue
                seen.add(key)

                label = f"{make_name} {model_name} {year}"
                classes.append({
                    "label": label,
                    "make": make_name,
                    "model": model_name,
                    "year": str(year),
                })

            # Be polite to the API
            time.sleep(0.15)

    # Sort by make, then model, then year
    classes.sort(key=lambda c: (c["make"], c["model"], c["year"]))
    return classes


# ──────────────────────────────────────────────
# Output writers
# ──────────────────────────────────────────────

def write_to_json(classes: list[dict], output_path: Path):
    """Write the car classes to a JSON file."""
    output_path.write_text(json.dumps(classes, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(classes)} car classes to {output_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch car makes/models/years from the NHTSA API and write car_classes.json"
    )
    parser.add_argument(
        "--makes", nargs="+", default=None,
        help="Specific makes to fetch (default: curated popular makes list)"
    )
    parser.add_argument(
        "--all-makes", action="store_true",
        help="Fetch ALL makes from NHTSA (warning: very large list, slow)"
    )
    parser.add_argument(
        "--years", nargs=2, type=int, default=[2018, 2025], metavar=("START", "END"),
        help="Year range to fetch (default: 2018 2025)"
    )
    parser.add_argument(
        "--car-type", choices=["passenger", "suv_truck", "all"], default="all",
        help="Filter by vehicle type (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print results without writing any files"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Custom output path for JSON file (default: car_classes.json in project root)"
    )
    args = parser.parse_args()

    # Determine which makes to fetch
    if args.all_makes:
        makes = get_all_makes()
    elif args.makes:
        makes = args.makes
    else:
        makes = POPULAR_MAKES

    year_start, year_end = args.years
    print(f"\nFetching models for {len(makes)} makes, years {year_start}-{year_end}")
    print(f"Vehicle type filter: {args.car_type}\n")

    classes = fetch_car_classes(makes, year_start, year_end, car_type=args.car_type)

    # Summary
    unique_makes = len(set(c["make"] for c in classes))
    unique_models = len(set((c["make"], c["model"]) for c in classes))
    print(f"\n{'='*60}")
    print(f"Total car classes:   {len(classes)}")
    print(f"Unique makes:        {unique_makes}")
    print(f"Unique models:       {unique_models}")
    print(f"Year range:          {year_start}-{year_end}")
    print(f"{'='*60}")

    # Preview first/last few
    print("\nSample classes:")
    for c in classes[:5]:
        print(f"  {c['label']}")
    if len(classes) > 10:
        print(f"  ... ({len(classes) - 10} more) ...")
    for c in classes[-5:]:
        print(f"  {c['label']}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # Always write to JSON — config.py auto-loads it
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).resolve().parent.parent / "car_classes.json"

    write_to_json(classes, output_path)
    print(f"\nconfig.py will auto-load from {output_path.name}")
    print("Done! You can now run: python download_data.py")


if __name__ == "__main__":
    main()
