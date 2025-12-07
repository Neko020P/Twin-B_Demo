#backend/server.py
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import logging
import tempfile
from uuid import uuid4

# Optional Dropbox support
try:
    import dropbox
    from dropbox.exceptions import ApiError
except Exception:
    dropbox = None
    ApiError = Exception

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure static folder relative to this file (backend/)
APP_STATIC = os.path.join(os.path.dirname(__file__), '..', 'frontend')

app = Flask(__name__, static_folder=APP_STATIC, static_url_path='')
CORS(app)

# Cache for CSV and SQL data to avoid reloading
csv_cache = {}
sql_cache = {}

# Dropbox configuration (via environment variables)
DROPBOX_TOKEN = os.getenv('DROPBOX_TOKEN')  # REQUIRED for production
# Folder paths in your Dropbox where files live (relative path like "Twin-B/agent_simulation")
DROPBOX_AGENT_FOLDER = os.getenv('DROPBOX_AGENT_FOLDER', 'Twin-B/agent_simulation')
DROPBOX_ENERGY_FOLDER = os.getenv('DROPBOX_ENERGY_FOLDER', 'Twin-B/energy')

# Normalize folder paths to start with slash
def _norm_dropbox_path(p):
    if not p:
        return '/'
    return ('/' + p.strip('/')) if not p.startswith('/') else p

DROPBOX_AGENT_FOLDER = _norm_dropbox_path(DROPBOX_AGENT_FOLDER)
DROPBOX_ENERGY_FOLDER = _norm_dropbox_path(DROPBOX_ENERGY_FOLDER)

# Initialize Dropbox client — REQUIRED, no fallback
DROPBOX_ENABLED = False
dbx = None
if DROPBOX_TOKEN:
    if dropbox is None:
        logger.error("❌ FATAL: Dropbox SDK not installed. Install 'dropbox' package to enable Dropbox support.")
        logger.error("Run: pip install dropbox")
    else:
        try:
            dbx = dropbox.Dropbox(DROPBOX_TOKEN)
            # quick test call to validate token
            dbx.users_get_current_account()
            DROPBOX_ENABLED = True
            logger.info("✅ Dropbox integration ENABLED.")
            logger.info(f"   Agent folder: {DROPBOX_AGENT_FOLDER}")
            logger.info(f"   Energy folder: {DROPBOX_ENERGY_FOLDER}")
        except Exception as e:
            logger.error(f"❌ FATAL: Failed to initialize Dropbox client: {e}")
            logger.error("Please check your DROPBOX_TOKEN environment variable.")
else:
    logger.error("❌ FATAL: DROPBOX_TOKEN environment variable not set.")
    logger.error("This system requires Dropbox integration to function.")
    logger.error("Set environment variable: $env:DROPBOX_TOKEN='<your_token>'")

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

# ---------------- Dropbox helpers ----------------
def _dropbox_list_folder(folder_path):
    """Return list of file dicts for given Dropbox folder path (non-recursive)."""
    if not DROPBOX_ENABLED or dbx is None:
        logger.error("Dropbox not enabled")
        return []
    try:
        entries = []
        res = dbx.files_list_folder(folder_path, recursive=False)
        entries.extend(res.entries)
        while res.has_more:
            res = dbx.files_list_folder_continue(res.cursor)
            entries.extend(res.entries)
        # Convert to simple dicts
        files = []
        for e in entries:
            # only files, not folders
            if hasattr(e, 'name') and hasattr(e, 'path_lower'):
                files.append({'name': e.name, 'path_lower': e.path_lower})
        return files
    except ApiError as e:
        logger.warning(f"Dropbox list_folder failed for {folder_path}: {e}")
        return []
    except Exception as e:
        logger.warning(f"Dropbox unexpected error listing {folder_path}: {e}")
        return []

def _dropbox_download_file(folder_path, filename):
    """Download a file from Dropbox and return BytesIO or None on failure."""
    if not DROPBOX_ENABLED or dbx is None:
        logger.error(f"Dropbox not enabled, cannot download {filename}")
        return None
    try:
        # Ensure folder doesn't end with slash
        folder = folder_path.rstrip('/')
        full_path = f"{folder}/{filename}" if folder != '' else f"/{filename}"
        logger.info(f"Dropbox download: {full_path}")
        md, res = dbx.files_download(full_path)
        data = res.content  # bytes
        return BytesIO(data)
    except ApiError as e:
        logger.warning(f"Dropbox files_download failed for {filename} in {folder_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Dropbox unexpected error downloading {filename}: {e}")
        return None

def extract_electricity_facility_from_sql(sqlite_path):
    """
    Extract electricity facility data (kWh) from an EnergyPlus SQL file.
    Best-effort approach: picks first tabular/reportdata table and extracts numeric columns.
    """
    results = []
    if not os.path.exists(sqlite_path):
        return results
    try:
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()

        # Get list of tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        
        # Prefer tables with 'tabular' or 'report' in name
        import re
        candidates = [t for t in tables if re.search(r'(tabular|report|reportdata|report_data)', t, re.I)]
        if not candidates and tables:
            candidates = tables

        # Try each candidate until we find useful numeric column(s)
        for tbl in candidates:
            try:
                df = pd.read_sql_query(f"SELECT * FROM \"{tbl}\" LIMIT 100000", conn)
            except Exception:
                continue
            
            if df is None or df.empty:
                continue
            
            # Look for electricity-like column
            elect_cols = [c for c in df.columns if re.search(r'electric', c, re.I)]
            if elect_cols:
                col = elect_cols[0]
                try:
                    series = pd.to_numeric(df[col], errors='coerce')
                    df2 = pd.DataFrame({'electricity_kwh': series.fillna(0)})
                    results = dataframe_to_clean_dict(df2)
                    conn.close()
                    return results
                except Exception:
                    pass
            else:
                # Try first numeric column
                numeric_cols = []
                for c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        numeric_cols.append(c)
                if numeric_cols:
                    chosen = numeric_cols[0]
                    try:
                        series = pd.to_numeric(df[chosen], errors='coerce')
                        df2 = pd.DataFrame({'electricity_kwh': series.fillna(0)})
                        results = dataframe_to_clean_dict(df2)
                        conn.close()
                        return results
                    except Exception:
                        pass
        conn.close()
    except Exception as e:
        logger.warning(f"extract_electricity_facility_from_sql error: {e}")
        try:
            conn.close()
        except Exception:
            pass
    
    return results

# ===== DROPBOX-ONLY API ENDPOINTS =====

@app.route('/api/csv-list')
def get_csv_list():
    """Get list of available CSV files (agents/zones) from Dropbox"""
    try:
        if not DROPBOX_ENABLED:
            return jsonify({"status": "error", "message": "Dropbox not enabled. Set DROPBOX_TOKEN environment variable."}), 503

        files = _dropbox_list_folder(DROPBOX_AGENT_FOLDER)
        if not files:
            return jsonify({"status": "error", "message": f"No files found in Dropbox folder {DROPBOX_AGENT_FOLDER}"}), 404

        csv_files = sorted([os.path.splitext(f['name'])[0] for f in files if f['name'].lower().endswith('.csv')])

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
        logger.error(f"get_csv_list: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/energy-list')
def get_energy_list():
    """Get list of available energy files from Dropbox"""
    try:
        if not DROPBOX_ENABLED:
            return jsonify({"status": "error", "message": "Dropbox not enabled. Set DROPBOX_TOKEN environment variable."}), 503

        files = _dropbox_list_folder(DROPBOX_ENERGY_FOLDER)
        if not files:
            return jsonify({"status": "error", "message": f"No files found in Dropbox folder {DROPBOX_ENERGY_FOLDER}"}), 404

        sql_files = sorted([os.path.splitext(f['name'])[0].replace('eplusout_', '').replace('.sql', '') for f in files if (f['name'].lower().endswith('.sql') or f['name'].lower().endswith('.csv'))])

        return jsonify({
            "status": "success",
            "files": sql_files,
            "scenarios": SCENARIOS,
            "policies": POLICIES,
            "count": len(sql_files)
        })
    except Exception as e:
        logger.error(f"get_energy_list: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/energy/<scenario>/<policy>')
def get_energy_data(scenario, policy):
    """Load electricity facility data from SQL file in Dropbox energy folder (Dropbox-only)"""
    try:
        if not DROPBOX_ENABLED:
            return jsonify({"status": "error", "message": "Dropbox not enabled. Set DROPBOX_TOKEN environment variable."}), 503

        filename_sql = f'eplusout_{scenario}_{policy}.sql'
        filename_csv = f'eplusout_{scenario}_{policy}.csv'
        logger.info(f"get_energy_data: Loading {filename_sql} or {filename_csv} from Dropbox")

        data = None

        # Try cached SQL first
        cache_key = filename_sql
        if cache_key in sql_cache:
            data = sql_cache[cache_key]
            logger.info(f"Loaded {filename_sql} from cache")
        else:
            # Try downloading SQL first from Dropbox
            bio = _dropbox_download_file(DROPBOX_ENERGY_FOLDER, filename_sql)
            if bio is not None:
                # Save to a temporary local file so sqlite3 can open it
                tmp_path = os.path.join(tempfile.gettempdir(), f'dropbox_{uuid4().hex}_{filename_sql}')
                try:
                    with open(tmp_path, 'wb') as f:
                        f.write(bio.read())
                    data = extract_electricity_facility_from_sql(tmp_path)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            # If SQL not found, try CSV version from Dropbox
            if data is None:
                bio = _dropbox_download_file(DROPBOX_ENERGY_FOLDER, filename_csv)
                if bio is not None:
                    df = pd.read_csv(bio)
                    data = dataframe_to_clean_dict(df)

            # Cache if obtained
            if data is not None:
                sql_cache[cache_key] = data

        if data is None:
            error_msg = f"File not found in Dropbox for scenario={scenario}, policy={policy}"
            logger.error(error_msg)
            return jsonify({"status": "error", "message": error_msg}), 404

        # Return with aggregated statistics
        if data:
            df = pd.DataFrame(data)
            stats = {
                'total_kwh': float(df['electricity_kwh'].sum()) if not df.empty and 'electricity_kwh' in df.columns else 0,
                'average_kwh': float(df['electricity_kwh'].mean()) if not df.empty and 'electricity_kwh' in df.columns else 0,
                'max_kwh': float(df['electricity_kwh'].max()) if not df.empty and 'electricity_kwh' in df.columns else 0,
                'min_kwh': float(df['electricity_kwh'].min()) if not df.empty and 'electricity_kwh' in df.columns else 0,
                'count': len(data)
            }
        else:
            stats = {'total_kwh': 0, 'average_kwh': 0, 'max_kwh': 0, 'min_kwh': 0, 'count': 0}

        response_data = {
            "status": "success",
            "filename": filename_sql,
            "scenario": scenario,
            "policy": policy,
            "data": data,
            "rows": data,
            "count": len(data),
            "statistics": stats
        }
        return jsonify(response_data)

    except Exception as e:
        error_msg = f"Exception in get_energy_data: {e}"
        logger.exception(error_msg)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/energy-all')
def get_all_energy_data():
    """Get electricity facility data from all energy files in Dropbox (Dropbox-only)"""
    try:
        if not DROPBOX_ENABLED:
            return jsonify({"status": "error", "message": "Dropbox not enabled. Set DROPBOX_TOKEN environment variable."}), 503

        logger.info("get_all_energy_data: Loading all energy files from Dropbox")
        all_results = []

        files = _dropbox_list_folder(DROPBOX_ENERGY_FOLDER)
        names = [f['name'] for f in files if (f['name'].lower().endswith('.sql') or f['name'].lower().endswith('.csv'))]
        
        for name in names:
            basename, ext = os.path.splitext(name)
            norm = basename.replace('eplusout_', '')
            parts = norm.split('_')
            # attempt to parse scenario & policy from filename
            if len(parts) >= 2:
                scenario = parts[0]
                policy = '_'.join(parts[1:])
            else:
                scenario = norm
                policy = ''
            
            # Download and compute stats
            bio = _dropbox_download_file(DROPBOX_ENERGY_FOLDER, name)
            data = None
            if bio is not None:
                if name.lower().endswith('.sql'):
                    tmp_path = os.path.join(tempfile.gettempdir(), f'dropbox_{uuid4().hex}_{name}')
                    try:
                        with open(tmp_path, 'wb') as f:
                            f.write(bio.read())
                        data = extract_electricity_facility_from_sql(tmp_path)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                else:
                    df = pd.read_csv(bio)
                    data = dataframe_to_clean_dict(df)

            if data:
                df = pd.DataFrame(data)
                result = {
                    "filename": name,
                    "scenario": scenario,
                    "policy": policy,
                    "count": len(data),
                    "total_kwh": float(df['electricity_kwh'].sum()) if not df.empty and 'electricity_kwh' in df.columns else 0,
                    "average_kwh": float(df['electricity_kwh'].mean()) if not df.empty and 'electricity_kwh' in df.columns else 0,
                    "max_kwh": float(df['electricity_kwh'].max()) if not df.empty and 'electricity_kwh' in df.columns else 0,
                    "min_kwh": float(df['electricity_kwh'].min()) if not df.empty and 'electricity_kwh' in df.columns else 0
                }
                all_results.append(result)

        logger.info(f"Loaded {len(all_results)} energy files from Dropbox")
        return jsonify({"status": "success", "data": all_results, "count": len(all_results)})

    except Exception as e:
        logger.exception("Exception in get_all_energy_data")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/energy-csv/<filename>')
def get_energy_csv(filename):
    """Serve CSV files from Dropbox energy folder (Dropbox-only)"""
    try:
        if not DROPBOX_ENABLED:
            return jsonify({"status": "error", "message": "Dropbox not enabled. Set DROPBOX_TOKEN environment variable."}), 503

        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"status": "error", "message": "Invalid filename"}), 400

        if not filename.lower().endswith('.csv'):
            filename_csv = f"{filename}.csv"
        else:
            filename_csv = filename

        logger.info(f"get_energy_csv: Loading {filename_csv} from Dropbox")

        cache_key = f'energy::{filename_csv}'
        if cache_key in csv_cache:
            data = csv_cache[cache_key]
            logger.info(f"Loaded {filename_csv} from cache ({len(data)} rows)")
        else:
            # Download from Dropbox
            bio = _dropbox_download_file(DROPBOX_ENERGY_FOLDER, filename_csv)
            if bio is None:
                error_msg = f"File not found in Dropbox: {filename_csv}"
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 404
            
            df = pd.read_csv(bio)
            data = dataframe_to_clean_dict(df)
            csv_cache[cache_key] = data
            logger.info(f"Read and cached {filename_csv} from Dropbox ({len(data)} rows)")

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
        logger.exception("Exception in get_energy_csv")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/csv/<data_type>/<scenario>/<policy>')
def get_csv_data(data_type, scenario, policy):
    """Load specific CSV file from Dropbox agent_simulation folder (Dropbox-only)"""
    try:
        if not DROPBOX_ENABLED:
            return jsonify({"status": "error", "message": "Dropbox not enabled. Set DROPBOX_TOKEN environment variable."}), 503

        # tolerant parsing: strip trailing colon/semicolon suffixes like ":1" or ";1"
        try:
            policy = str(policy).split(':', 1)[0].split(';', 1)[0].strip()
        except Exception:
            pass

        filename = f'{data_type}_{scenario}_{policy}.csv'
        logger.info(f"get_csv_data: Loading {filename} from Dropbox")

        cache_key = filename
        if cache_key in csv_cache:
            logger.info(f"Loading {filename} from cache")
            data = csv_cache[cache_key]
        else:
            # Download from Dropbox
            bio = _dropbox_download_file(DROPBOX_AGENT_FOLDER, filename)
            if bio is None:
                error_msg = f"File not found in Dropbox: {filename}"
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 404

            df = pd.read_csv(bio, dtype=str, na_values=['', 'NaN', 'nan'])
            data = dataframe_to_clean_dict(df)
            csv_cache[cache_key] = data
            logger.info(f"Successfully loaded and cached from Dropbox: {len(data)} rows")

        response_data = {
            "status": "success",
            "filename": filename,
            "count": len(data),
            "data": data,
            "rows": data,
            "columns": list(data[0].keys()) if data else []
        }
        return jsonify(response_data)
    except Exception as e:
        logger.exception("Exception in get_csv_data")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/csv-file/<filename>')
def get_csv_by_filename(filename):
    """Load arbitrary CSV file from Dropbox agent_simulation folder (Dropbox-only)"""
    try:
        if not DROPBOX_ENABLED:
            return jsonify({"status": "error", "message": "Dropbox not enabled. Set DROPBOX_TOKEN environment variable."}), 503

        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"status": "error", "message": "Invalid filename"}), 400

        filename_csv = f'{filename}.csv' if not filename.lower().endswith('.csv') else filename
        logger.info(f"get_csv_by_filename: Loading {filename_csv} from Dropbox")

        cache_key = f'file::{filename_csv}'
        if cache_key in csv_cache:
            data = csv_cache[cache_key]
            logger.info(f"Loaded {filename_csv} from cache ({len(data)} rows)")
        else:
            # Download from Dropbox
            bio = _dropbox_download_file(DROPBOX_AGENT_FOLDER, filename_csv)
            if bio is None:
                error_msg = f"File not found in Dropbox: {filename_csv}"
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 404

            df = pd.read_csv(bio, dtype=str, na_values=['', 'NaN', 'nan'])
            data = dataframe_to_clean_dict(df)
            csv_cache[cache_key] = data
            logger.info(f"Read and cached {filename_csv} from Dropbox ({len(data)} rows)")

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
        logger.exception("Exception in get_csv_by_filename")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/csv-preview/<data_type>/<scenario>/<policy>')
def get_csv_preview(data_type, scenario, policy):
    """Get first 10 rows preview from Dropbox (Dropbox-only)"""
    try:
        if not DROPBOX_ENABLED:
            return jsonify({"status": "error", "message": "Dropbox not enabled. Set DROPBOX_TOKEN environment variable."}), 503

        # tolerant parsing: strip trailing colon/semicolon suffixes like ":1" or ";1"
        try:
            policy = str(policy).split(':', 1)[0].split(';', 1)[0].strip()
        except Exception:
            pass

        filename = f'{data_type}_{scenario}_{policy}.csv'
        logger.info(f"get_csv_preview: Loading {filename} from Dropbox")

        bio = _dropbox_download_file(DROPBOX_AGENT_FOLDER, filename)
        if bio is None:
            error_msg = f"File not found in Dropbox: {filename}"
            logger.error(error_msg)
            return jsonify({"status": "error", "message": error_msg}), 404

        df = pd.read_csv(bio, nrows=10, dtype=str, na_values=['', 'NaN', 'nan'])
        preview_data = dataframe_to_clean_dict(df)
        
        # best-effort total row count using full read
        total_rows = None
        try:
            bio.seek(0)
            total_rows = sum(1 for _ in bio.getvalue().splitlines()) - 1
        except Exception:
            total_rows = len(df)
        
        return jsonify({
            "status": "success",
            "filename": filename,
            "preview": preview_data,
            "rows": preview_data,
            "total_rows": total_rows,
            "columns": list(df.columns)
        })

    except Exception as e:
        logger.exception("Exception in get_csv_preview")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/csv-count/<filename>')
def get_csv_count(filename):
    """Get total row count of CSV file from Dropbox (Dropbox-only)"""
    try:
        if not DROPBOX_ENABLED:
            return jsonify({"status": "error", "message": "Dropbox not enabled. Set DROPBOX_TOKEN environment variable."}), 503

        filename_csv = f'{filename}.csv' if not filename.lower().endswith('.csv') else filename
        logger.info(f"get_csv_count: {filename_csv} from Dropbox")

        bio = _dropbox_download_file(DROPBOX_AGENT_FOLDER, filename_csv)
        if bio is None:
            error_msg = f"File not found in Dropbox: {filename_csv}"
            logger.error(error_msg)
            return jsonify({"status": "error", "message": error_msg}), 404

        bio.seek(0)
        df = pd.read_csv(bio)
        return jsonify({"status": "success", "filename": filename_csv, "count": len(df)})

    except Exception as e:
        logger.exception("Exception in get_csv_count")
        return jsonify({"status": "error", "message": str(e)}), 500

# Serve frontend static files AFTER API endpoints so /api/* works reliably
@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_frontend(path):
    try:
        return send_from_directory(app.static_folder, path)
    except Exception:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    logger.info("Starting Twin-B Flask Server")
    if DROPBOX_ENABLED:
        logger.info("✅ Dropbox mode: ENABLED")
        logger.info(f"   Agent folder: {DROPBOX_AGENT_FOLDER}")
        logger.info(f"   Energy folder: {DROPBOX_ENERGY_FOLDER}")
    else:
        logger.error("❌ Dropbox mode: DISABLED")
        logger.error("❌ FATAL: This application requires Dropbox integration to function.")
        logger.error("❌ Please set the DROPBOX_TOKEN environment variable before starting the server.")
        logger.error("")
        logger.error("Example (PowerShell):")
        logger.error("  $env:DROPBOX_TOKEN='<your_actual_token>'")
        logger.error("  python backend/server.py")

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)