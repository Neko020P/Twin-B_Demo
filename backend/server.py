from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# Configure static folder relative to this file (backend/)
APP_STATIC = os.path.join(os.path.dirname(__file__), '..', 'frontend')

app = Flask(__name__, static_folder=APP_STATIC, static_url_path='')
CORS(app)

# Cache for CSV and SQL data to avoid reloading
csv_cache = {}
sql_cache = {}

# Define available scenarios and datasets
SCENARIOS = ['conference', 'exam_period', 'normal_weekday', 'summer_break', 'weekend']
POLICIES = ['minimum_activation', 'setpoint_expansion_1c', 'setpoint_expansion_2c', 'setpoint_expansion_3c']

def dataframe_to_clean_dict(df):
    """Convert dataframe to dict with NaN/Inf/empty strings properly handled"""
    # Replace NaN and Inf with None
    df = df.where(pd.notna(df), None)
    df = df.replace([np.inf, -np.inf], None)

    # Convert to dict
    data = df.to_dict(orient='records')

    # Second pass: clean any remaining problematic values
    cleaned_data = []
    for row in data:
        cleaned_row = {}
        for key, value in row.items():
            # Handle various problematic values
            try:
                if pd.isna(value) or value is None:
                    cleaned_row[key] = None
                elif isinstance(value, float):
                    if np.isnan(value) or np.isinf(value):
                        cleaned_row[key] = None
                    else:
                        cleaned_row[key] = float(value)
                elif isinstance(value, (int, np.integer)):
                    cleaned_row[key] = int(value)
                elif isinstance(value, str):
                    # Convert empty strings to None
                    cleaned_row[key] = value if value.strip() else None
                else:
                    cleaned_row[key] = value
            except Exception:
                # Fallback: include raw value if checks fail
                cleaned_row[key] = value
        cleaned_data.append(cleaned_row)

    return cleaned_data

def extract_electricity_facility_from_sql(filepath):
    """
    Extract Electricity:Facility values from EnergyPlus SQL database.
    Returns a list of dicts with timestamp and electricity value.
    """
    try:
        conn = sqlite3.connect(filepath)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        print(f"[DEBUG] Opening SQL file: {filepath}")

        # Query to get Electricity:Facility values (attempt typical EnergyPlus schema)
        query = """
        SELECT 
            t.Month,
            t.Day,
            t.Hour,
            t.Minute,
            t.Interval,
            r.Value
        FROM Time t
        LEFT JOIN ReportMeterData r ON t.TimeIndex = r.TimeIndex
        WHERE r.MeterIndex = (
            SELECT MeterIndex FROM Meter WHERE MeterName LIKE '%Electricity:Facility%'
        )
        ORDER BY t.TimeIndex ASC
        """

        try:
            cursor.execute(query)
            rows = cursor.fetchall()

            data = []
            for row in rows:
                data.append({
                    'month': row['Month'],
                    'day': row['Day'],
                    'hour': row['Hour'],
                    'minute': row['Minute'],
                    'interval': row['Interval'],
                    'electricity_kwh': float(row['Value']) if row['Value'] is not None else None
                })

            conn.close()
            print(f"[DEBUG] Extracted {len(data)} records from {filepath}")
            return data

        except sqlite3.OperationalError as e:
            print(f"[WARNING] Query failed: {str(e)}. Attempting alternative query structure...")

            # Alternative approach: discover tables, then try to locate relevant meter indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"[DEBUG] Available tables: {tables}")

            data = []
            if 'ReportMeterData' in tables:
                # Get meter indices whose MeterName contains Electricity:Facility
                try:
                    cursor.execute("SELECT MeterIndex FROM Meter WHERE MeterName LIKE '%Electricity:Facility%'")
                    meter_indices = cursor.fetchall()
                    if meter_indices:
                        meter_idx = meter_indices[0][0]
                        cursor.execute("""
                            SELECT r.TimeIndex, r.Value, t.Month, t.Day, t.Hour, t.Minute
                            FROM ReportMeterData r
                            JOIN Time t ON r.TimeIndex = t.TimeIndex
                            WHERE r.MeterIndex = ?
                            ORDER BY r.TimeIndex ASC
                        """, (meter_idx,))

                        for row in cursor.fetchall():
                            data.append({
                                'month': row[2],
                                'day': row[3],
                                'hour': row[4],
                                'minute': row[5],
                                'interval': None,
                                'electricity_kwh': float(row[1]) if row[1] is not None else None
                            })
                except Exception as ex:
                    print(f"[WARNING] Alternative extraction failed: {ex}")

            conn.close()
            print(f"[DEBUG] Extracted {len(data)} records (alternative method)")
            return data

    except Exception as e:
        print(f"[ERROR] Failed to extract electricity data from {filepath}: {str(e)}")
        return []

@app.route('/api/csv-list')
def get_csv_list():
    """Get list of available CSV files"""
    try:
        data_dir = Path('./data/agent_simulation')
        csv_files = sorted([f.stem for f in data_dir.glob('*.csv')])

        agents_files = [f for f in csv_files if f.startswith('agents_')]
        zones_files = [f for f in csv_files if f.startswith('zones_')]

        return jsonify({
            "status": "success",
            "agents": agents_files,
            "zones": zones_files,
            "scenarios": SCENARIOS,
            "policies": POLICIES
        })
    except Exception as e:
        print(f"[ERROR] get_csv_list: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/energy-list')
def get_energy_list():
    """Get list of available energy SQL files"""
    try:
        energy_dir = Path('./data/energy')
        sql_files = sorted([f.stem.replace('eplusout_', '') for f in energy_dir.glob('eplusout_*.sql')])

        return jsonify({
            "status": "success",
            "files": sql_files,
            "scenarios": SCENARIOS,
            "policies": POLICIES,
            "count": len(sql_files)
        })
    except Exception as e:
        print(f"[ERROR] get_energy_list: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/energy/<scenario>/<policy>')
def get_energy_data(scenario, policy):
    """Load electricity facility data from SQL file"""
    try:
        filename = f'eplusout_{scenario}_{policy}.sql'
        filepath = os.path.join('./data/energy', filename)

        print(f"[DEBUG] get_energy_data: Loading {filename}")
        print(f"[DEBUG] File path: {filepath}")
        print(f"[DEBUG] File exists: {os.path.exists(filepath)}")

        if not os.path.exists(filepath):
            error_msg = f"File not found: {filename}"
            print(f"[ERROR] {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 404

        # Check cache first
        cache_key = filename
        if cache_key in sql_cache:
            print(f"[DEBUG] Loading {filename} from cache")
            data = sql_cache[cache_key]
        else:
            print(f"[DEBUG] Extracting electricity data from {filename}")
            data = extract_electricity_facility_from_sql(filepath)
            sql_cache[cache_key] = data
            print(f"[DEBUG] Successfully cached {len(data)} rows")

        # Return with aggregated statistics
        if data:
            df = pd.DataFrame(data)
            stats = {
                'total_kwh': float(df['electricity_kwh'].sum()) if not df.empty else 0,
                'average_kwh': float(df['electricity_kwh'].mean()) if not df.empty else 0,
                'max_kwh': float(df['electricity_kwh'].max()) if not df.empty else 0,
                'min_kwh': float(df['electricity_kwh'].min()) if not df.empty else 0,
                'count': len(data)
            }
        else:
            stats = {
                'total_kwh': 0,
                'average_kwh': 0,
                'max_kwh': 0,
                'min_kwh': 0,
                'count': 0
            }

        response_data = {
            "status": "success",
            "filename": filename,
            "scenario": scenario,
            "policy": policy,
            "data": data,
            "rows": data,
            "count": len(data),
            "statistics": stats
        }
        print(f"[DEBUG] Returning response with {len(data)} rows")
        return jsonify(response_data)

    except Exception as e:
        error_msg = f"Exception in get_energy_data: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500

@app.route('/api/energy-all')
def get_all_energy_data():
    """Get electricity facility data from all energy SQL files"""
    try:
        print("[DEBUG] get_all_energy_data: Loading all energy files")
        all_results = []

        for scenario in SCENARIOS:
            for policy in POLICIES:
                filename = f'eplusout_{scenario}_{policy}.sql'
                filepath = os.path.join('./data/energy', filename)

                if not os.path.exists(filepath):
                    print(f"[WARNING] File not found: {filename}")
                    continue

                cache_key = filename
                if cache_key in sql_cache:
                    data = sql_cache[cache_key]
                else:
                    data = extract_electricity_facility_from_sql(filepath)
                    sql_cache[cache_key] = data

                if data:
                    df = pd.DataFrame(data)
                    result = {
                        "filename": filename,
                        "scenario": scenario,
                        "policy": policy,
                        "count": len(data),
                        "total_kwh": float(df['electricity_kwh'].sum()) if not df.empty else 0,
                        "average_kwh": float(df['electricity_kwh'].mean()) if not df.empty else 0,
                        "max_kwh": float(df['electricity_kwh'].max()) if not df.empty else 0,
                        "min_kwh": float(df['electricity_kwh'].min()) if not df.empty else 0
                    }
                    all_results.append(result)

        print(f"[DEBUG] Loaded {len(all_results)} energy files")
        return jsonify({
            "status": "success",
            "data": all_results,
            "count": len(all_results)
        })

    except Exception as e:
        error_msg = f"Exception in get_all_energy_data: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500

@app.route('/api/energy-csv/<filename>')
def get_energy_csv(filename):
    """
    Serve CSV files from ./data/energy/<filename>.csv
    Usage example: /api/energy-csv/rank_optimal
    """
    try:
        # Basic safety check
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"status": "error", "message": "Invalid filename"}), 400

        if not filename.lower().endswith('.csv'):
            filename_csv = f"{filename}.csv"
        else:
            filename_csv = filename

        filepath = os.path.join('./data/energy', filename_csv)
        print(f"[DEBUG] get_energy_csv: Loading {filepath}")

        if not os.path.exists(filepath):
            error_msg = f"File not found: {filename_csv}"
            print(f"[ERROR] {error_msg}")
            return jsonify({"status": "error", "message": error_msg}), 404

        cache_key = f'energy::{filename_csv}'
        if cache_key in csv_cache:
            data = csv_cache[cache_key]
            print(f"[DEBUG] Loaded {filename_csv} from cache ({len(data)} rows)")
        else:
            try:
                df = pd.read_csv(filepath)
                # Convert to clean dict and ensure numeric conversion where possible
                if 'comfort_score' in df.columns:
                    df['comfort_score'] = pd.to_numeric(df['comfort_score'], errors='coerce')
                if 'energy_score' in df.columns:
                    df['energy_score'] = pd.to_numeric(df['energy_score'], errors='coerce')
                if 'energy_usage_mwh' in df.columns:
                    df['energy_usage_mwh'] = pd.to_numeric(df['energy_usage_mwh'], errors='coerce')

                data = dataframe_to_clean_dict(df)
                csv_cache[cache_key] = data
                print(f"[DEBUG] Read and cached {filename_csv} ({len(data)} rows)")
            except Exception as read_error:
                error_msg = f"Error reading CSV {filename_csv}: {str(read_error)}"
                print(f"[ERROR] {error_msg}")
                return jsonify({"status": "error", "message": error_msg}), 500

        response_data = {
            "status": "success",
            "filename": filename_csv,
            "count": len(data),
            "data": data,
            "rows": data,
            "columns": list(data[0].keys()) if data else []
        }
        return jsonify(response_data)
    except Exception as e:
        error_msg = f"Exception in get_energy_csv: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({"status": "error", "message": error_msg}), 500

@app.route('/api/csv/<data_type>/<scenario>/<policy>')
def get_csv_data(data_type, scenario, policy):
    """Load specific CSV file - data_type: agents or zones (all rows)"""
    try:
        # Construct filename
        filename = f'{data_type}_{scenario}_{policy}.csv'
        filepath = os.path.join('./data/agent_simulation', filename)

        print(f"[DEBUG] get_csv_data: Loading {filename}")
        print(f"[DEBUG] File path: {filepath}")
        print(f"[DEBUG] File exists: {os.path.exists(filepath)}")

        # Check if file exists
        if not os.path.exists(filepath):
            error_msg = f"File not found: {filename}"
            print(f"[ERROR] {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 404

        # Check cache first
        cache_key = filename
        if cache_key in csv_cache:
            print(f"[DEBUG] Loading {filename} from cache")
            data = csv_cache[cache_key]
        else:
            try:
                print(f"[DEBUG] Reading CSV: {filename}")
                # Read CSV with error handling (all rows)
                df = pd.read_csv(filepath, dtype=str, na_values=['', 'NaN', 'nan'])
                print(f"[DEBUG] Read {len(df)} rows, {len(df.columns)} columns")

                # Convert to clean dict
                print(f"[DEBUG] Cleaning data...")
                data = dataframe_to_clean_dict(df)

                # Load ALL rows (no 1000-row truncation)
                csv_cache[cache_key] = data

                print(f"[DEBUG] Successfully loaded and cached: {len(data)} rows")
            except Exception as read_error:
                error_msg = f"Error reading CSV: {str(read_error)}"
                print(f"[ERROR] {error_msg}")
                return jsonify({
                    "status": "error",
                    "message": error_msg
                }), 500

        try:
            response_data = {
                "status": "success",
                "filename": filename,
                "count": len(data),
                "data": data,
                "rows": data,          # alias for frontend convenience
                "columns": list(data[0].keys()) if data else []
            }
            print(f"[DEBUG] Returning response with {len(data)} rows")
            return jsonify(response_data)
        except Exception as json_error:
            error_msg = f"Error serializing JSON: {str(json_error)}"
            print(f"[ERROR] {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 500
    except Exception as e:
        error_msg = f"Exception in get_csv_data: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500

@app.route('/api/csv-file/<filename>')
def get_csv_by_filename(filename):
    """Load arbitrary CSV file from ./data/agent_simulation/<filename>.csv (filename without .csv)"""
    try:
        # Reject suspicious filenames (no path traversal)
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"status": "error", "message": "Invalid filename"}), 400

        filepath = os.path.join('./data/agent_simulation', f'{filename}.csv')
        print(f"[DEBUG] get_csv_by_filename: Loading {filepath}")
        if not os.path.exists(filepath):
            error_msg = f"File not found: {filename}.csv"
            print(f"[ERROR] {error_msg}")
            return jsonify({"status": "error", "message": error_msg}), 404

        cache_key = f'file::{filename}.csv'
        if cache_key in csv_cache:
            data = csv_cache[cache_key]
            print(f"[DEBUG] Loaded {filename}.csv from cache ({len(data)} rows)")
        else:
            try:
                df = pd.read_csv(filepath, dtype=str, na_values=['', 'NaN', 'nan'])
                data = dataframe_to_clean_dict(df)
                csv_cache[cache_key] = data
                print(f"[DEBUG] Read and cached {filename}.csv ({len(data)} rows)")
            except Exception as read_error:
                error_msg = f"Error reading CSV {filename}.csv: {str(read_error)}"
                print(f"[ERROR] {error_msg}")
                return jsonify({"status": "error", "message": error_msg}), 500

        response_data = {
            "status": "success",
            "filename": f"{filename}.csv",
            "count": len(data),
            "data": data,
            "rows": data,
            "columns": list(data[0].keys()) if data else []
        }
        return jsonify(response_data)
    except Exception as e:
        error_msg = f"Exception in get_csv_by_filename: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({"status": "error", "message": error_msg}), 500

@app.route('/api/csv-preview/<data_type>/<scenario>/<policy>')
def get_csv_preview(data_type, scenario, policy):
    """Get first 10 rows preview"""
    try:
        filename = f'{data_type}_{scenario}_{policy}.csv'
        filepath = os.path.join('./data/agent_simulation', filename)

        print(f"[DEBUG] get_csv_preview: Loading {filename}")

        if not os.path.exists(filepath):
            error_msg = f"File not found: {filename}"
            print(f"[ERROR] {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 404

        try:
            print(f"[DEBUG] Reading preview from {filename}")
            df = pd.read_csv(filepath, nrows=10, dtype=str, na_values=['', 'NaN', 'nan'])

            # Convert to clean dict
            preview_data = dataframe_to_clean_dict(df)

            print(f"[DEBUG] Preview loaded: {len(df)} rows")

            # Count total rows efficiently
            with open(filepath, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1

            response_data = {
                "status": "success",
                "filename": filename,
                "preview": preview_data,
                "rows": preview_data,
                "total_rows": total_rows,
                "columns": list(df.columns)
            }
            print(f"[DEBUG] Preview response ready")
            return jsonify(response_data)
        except Exception as read_error:
            error_msg = f"Error reading preview: {str(read_error)}"
            print(f"[ERROR] {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 500
    except Exception as e:
        error_msg = f"Exception in get_csv_preview: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500

@app.route('/api/csv-count/<filename>')
def get_csv_count(filename):
    """Get total row count of CSV file"""
    try:
        filepath = os.path.join('./data/agent_simulation', f'{filename}.csv')

        print(f"[DEBUG] get_csv_count: {filename}")

        if not os.path.exists(filepath):
            error_msg = f"File not found: {filename}"
            print(f"[ERROR] {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 404

        try:
            # Count rows efficiently
            with open(filepath, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f) - 1

            print(f"[DEBUG] Row count: {count}")

            return jsonify({
                "status": "success",
                "filename": filename,
                "count": count
            })
        except Exception as count_error:
            error_msg = f"Error counting rows: {str(count_error)}"
            print(f"[ERROR] {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 500
    except Exception as e:
        error_msg = f"Exception in get_csv_count: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500

# Serve frontend static files AFTER API endpoints so /api/* works reliably
@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_frontend(path):
    # Serve from the configured static folder
    try:
        return send_from_directory(app.static_folder, path)
    except Exception as e:
        # If file not found, return index.html (useful for SPA routing)
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("=" * 70)
    print("Starting Twin-B Flask Server")
    print("=" * 70)
    print(f"Server: http://0.0.0.0:5000")
    print(f"CSV Data directory: {os.path.abspath('./data/agent_simulation')}")
    print(f"Energy Data directory: {os.path.abspath('./data/energy')}")

    print("\n✓ Available CSV files:")
    data_dir = Path('./data/agent_simulation')
    csv_files = sorted(data_dir.glob('*.csv'))
    if csv_files:
        for csv_file in csv_files:
            file_size = os.path.getsize(csv_file) / (1024 * 1024)
            print(f"  • {csv_file.name} ({file_size:.2f} MB)")
    else:
        print("  ⚠ No CSV files found!")

    print("\n✓ Available Energy files:")
    energy_dir = Path('./data/energy')
    energy_files = sorted(energy_dir.glob('*'))
    if energy_files:
        for f in energy_files:
            file_size = os.path.getsize(f) / (1024 * 1024)
            print(f"  • {f.name} ({file_size:.2f} MB)")
    else:
        print("  ⚠ No energy files found!")

    print(f"\n✓ Scenarios: {', '.join(SCENARIOS)}")
    print(f"✓ Policies: {', '.join(POLICIES)}")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)