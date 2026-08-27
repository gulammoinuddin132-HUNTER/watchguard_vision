from src.database.db_manager import DatabaseManager
import sqlite3

db = DatabaseManager()
# 1. Initial
e0 = db.get_recent_security_events(limit=50)
print('Initial events count:', len(e0))

# 2. Log 1 event
eid = db.log_security_event('TEST_EVENT', 'GREEN', 'Test event description', person_name='TestUser')
print('Logged event id:', eid)
e1 = db.get_recent_security_events(limit=50)
print('After 1 event count:', len(e1))

# 3. Clean
conn = sqlite3.connect(str(db.db_path))
conn.execute('DELETE FROM security_events')
conn.commit()
conn.close()

e_final = db.get_recent_security_events(limit=50)
print('Final events count:', len(e_final))
