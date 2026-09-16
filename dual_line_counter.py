import time
import sqlite3

def signed_distance_to_line(line_start, line_end, point):
    """
    Returns the signed perpendicular distance from a point to a line.
    For a horizontal Right-to-Left line:
      Positive = ABOVE the line
      Negative = BELOW the line
    """
    ax, ay = line_start
    bx, by = line_end
    px, py = point
    
    # Direction vector of the line
    dx = bx - ax
    dy = by - ay
    
    # Length of the line segment
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return 0.0
    
    # Signed distance (cross product divided by length)
    return ((dx) * (py - ay) - (dy) * (px - ax)) / length


class SingleLineCounter:
    """
    Single-line directional people counter with hysteresis dead zone.
    
    A crossing is only registered when the person's anchor moves from
    one side of the line to the other AND is at least `dead_zone` pixels
    away from the line. This prevents jitter-based false counts.
    
    After a crossing, the track enters a cooldown period during which
    no further crossings can be registered for that ID.
    """
    def __init__(self, line_start, line_end, dead_zone=15, cooldown_frames=30, timeout_seconds=5.0, db_path="analytics.db"):
        self.line = (line_start, line_end)
        self.dead_zone = dead_zone
        self.cooldown_frames = cooldown_frames
        self.timeout_seconds = timeout_seconds
        
        self.entered_count = 2259
        self.exited_count = 1966
        
        # Dictionary mapping tracker_id -> state dict
        self.tracks = {}
        self._frame_count = 0

        # Analytics DB setup
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Table for IN/OUT events — store local time
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
                        event_type TEXT,
                        track_id INTEGER
                    )
                ''')
                # Table for live active tracks tracking — store local time
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS live_status (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
                        active_tracks INTEGER,
                        total_in INTEGER,
                        total_out INTEGER
                    )
                ''')
                conn.commit()
        except Exception as e:
            print(f"Error initializing DB: {e}")

    def _log_event(self, event_type, track_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO events (event_type, track_id) VALUES (?, ?)', (event_type, track_id))
                conn.commit()
        except Exception as e:
            print(f"Error logging event: {e}")

    def log_status(self, active_tracks):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO live_status (active_tracks, total_in, total_out) 
                    VALUES (?, ?, ?)
                ''', (active_tracks, self.entered_count, self.exited_count))
                conn.commit()
        except Exception as e:
            pass

    def _get_side(self, point):
        """
        Returns 1 (ABOVE), -1 (BELOW), or 0 (in dead zone) based on
        the signed distance from the point to the counting line.
        """
        dist = signed_distance_to_line(self.line[0], self.line[1], point)
        if dist > self.dead_zone:
            return 1   # clearly above
        elif dist < -self.dead_zone:
            return -1  # clearly below
        return 0       # in dead zone, ignore

    def update(self, detections):
        """
        Updates the counter state machine based on the latest detections.
        detections: An sv.Detections object with tracker_id populated.
        """
        current_time = time.time()
        self._frame_count += 1
        
        # Clean up stale tracks (BUG 1 FIX: lowered from 10s to 5s)
        stale_ids = [tid for tid, data in self.tracks.items() 
                     if current_time - data["last_seen"] > self.timeout_seconds]
        for tid in stale_ids:
            del self.tracks[tid]
            
        if detections.tracker_id is None or len(detections) == 0:
            return

        for i in range(len(detections)):
            tracker_id = int(detections.tracker_id[i])
            bbox = detections.xyxy[i]
            
            # Use CENTER of bbox (stable for this overhead camera angle)
            x1, y1, x2, y2 = bbox
            px = (x1 + x2) / 2.0
            py = (y1 + y2) / 2.0
            point = (px, py)
            
            current_side = self._get_side(point)
            
            if tracker_id not in self.tracks:
                # Only register if clearly on one side (not in dead zone)
                if current_side != 0:
                    self.tracks[tracker_id] = {
                        "confirmed_side": current_side,
                        "cooldown": 0,
                        "last_seen": current_time
                    }
                continue
                
            track = self.tracks[tracker_id]
            track["last_seen"] = current_time
            
            # If in dead zone, skip — don't update anything
            if current_side == 0:
                continue
            
            # If in cooldown, decrement but don't check for crossings
            if track["cooldown"] > 0:
                track["cooldown"] -= 1
                # Update confirmed_side so we have correct reference after cooldown
                track["confirmed_side"] = current_side
                continue
            
            confirmed_side = track["confirmed_side"]
            
            # Check for crossing: side must have genuinely flipped
            if confirmed_side != 0 and confirmed_side != current_side:
                if confirmed_side == 1 and current_side == -1:
                    # Crossed from above to below = ENTRY (IN)
                    self.entered_count += 1
                    track["cooldown"] = self.cooldown_frames
                    self._log_event("IN", tracker_id)
                    print(f"  *** Track {tracker_id}: ENTERED (above->below) "
                          f"point=({px:.0f},{py:.0f}) Total IN={self.entered_count}")
                elif confirmed_side == -1 and current_side == 1:
                    # Crossed from below to above = EXIT (OUT)
                    self.exited_count += 1
                    track["cooldown"] = self.cooldown_frames
                    self._log_event("OUT", tracker_id)
                    print(f"  *** Track {tracker_id}: EXITED (below->above) "
                          f"point=({px:.0f},{py:.0f}) Total OUT={self.exited_count}")
            
            # Update confirmed side
            track["confirmed_side"] = current_side
