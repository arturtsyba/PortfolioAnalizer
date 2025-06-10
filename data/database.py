import sqlite3

class Database:
    def __init__(self):
        self.conn = sqlite3.connect("portfolio.db")
        self.cursor = self.conn.cursor()

    def execute_query(self, query, params=(), fetch=False):
        try:
            self.cursor.execute(query, params)
            if fetch:
                result = [dict(zip([column[0] for column in self.cursor.description], row)) for row in self.cursor.fetchall()]
                return result
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            return False

    def close(self):
        self.conn.close()