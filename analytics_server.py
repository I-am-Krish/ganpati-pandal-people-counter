from flask import Flask, render_template, jsonify
import sqlite3
import datetime
import csv
import io
from flask import Response

app = Flask(__name__)
DB_PATH = "analytics.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('analytics.html')

@app.route('/api/stats')
def api_stats():
    """Returns the latest high-level stats."""
    try:
        conn = get_db_connection()
        # Get the latest live status
        live_status = conn.execute(
            'SELECT active_tracks, total_in, total_out FROM live_status ORDER BY id DESC LIMIT 1'
        ).fetchone()
        
        conn.close()

        if live_status:
            total_in = live_status['total_in']
            total_out = live_status['total_out']
            active_tracks = live_status['active_tracks']
        else:
            total_in = 0
            total_out = 0
            active_tracks = 0

        occupancy = max(0, total_in - total_out)

        return jsonify({
            "total_in": total_in,
            "total_out": total_out,
            "active_tracks": active_tracks,
            "occupancy": occupancy
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chart')
def api_chart():
    """Returns today's IN/OUT data grouped by hour for plotting.
    BUG 5 FIX: Uses date() directly since timestamps are now stored in local time.
    """
    try:
        conn = get_db_connection()
        
        today = datetime.date.today().isoformat()
        
        # Get counts of IN/OUT events grouped by hour for today
        query = '''
            SELECT 
                strftime('%H:00', timestamp) as hour_bucket,
                SUM(CASE WHEN event_type = 'IN' THEN 1 ELSE 0 END) as in_count,
                SUM(CASE WHEN event_type = 'OUT' THEN 1 ELSE 0 END) as out_count
            FROM events
            WHERE date(timestamp) = ?
            GROUP BY hour_bucket
            ORDER BY hour_bucket ASC
        '''
        rows = conn.execute(query, (today,)).fetchall()
        conn.close()

        labels = []
        in_data = []
        out_data = []
        
        for row in rows:
            labels.append(row['hour_bucket'])
            in_data.append(row['in_count'])
            out_data.append(row['out_count'])

        return jsonify({
            "labels": labels,
            "in_data": in_data,
            "out_data": out_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/export')
def api_export():
    """Generates and downloads a CSV report of today's traffic."""
    try:
        conn = get_db_connection()
        today = datetime.date.today().isoformat()
        query = '''
            SELECT timestamp, event_type, track_id 
            FROM events 
            WHERE date(timestamp) = ?
            ORDER BY timestamp ASC
        '''
        rows = conn.execute(query, (today,)).fetchall()
        conn.close()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Timestamp', 'Event Type', 'Track ID'])
        
        for row in rows:
            writer.writerow([row['timestamp'], row['event_type'], row['track_id']])

        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=traffic_report_{today_str}.csv"}
        )
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    # Run on port 7002, separate from the counting server
    print("Starting Analytics Server on http://0.0.0.0:7002")
    app.run(host='0.0.0.0', port=7002, debug=False, use_reloader=False)
