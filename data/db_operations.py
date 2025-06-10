from .database import Database
from datetime import date

class DBOperations:
    def __init__(self):
        self.db = Database()
        self.initialize_database()

    def initialize_database(self):
        """Инициализация базы данных (создание таблиц, если их нет)."""
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            investment_goal TEXT,
            horizon INTEGER
        )
        """
        create_portfolios_table = """
        CREATE TABLE IF NOT EXISTS portfolios (
            portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            portfolio_name TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """
        create_assets_table = """
        CREATE TABLE IF NOT EXISTS assets (
            asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER,
            ticker TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            count_assets FLOAT,
            purchase_price REAL NOT NULL,
            purchase_date DATE NOT NULL,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
        )
        """
        self.db.execute_query(create_users_table)
        self.db.execute_query(create_portfolios_table)
        self.db.execute_query(create_assets_table)

    def add_user(self, user_id, username, investment_goal=None, horizon=None):
        query = "INSERT INTO users (user_id, username, investment_goal, horizon) VALUES (?, ?, ?, ?)"
        return self.db.execute_query(query, (user_id, username, investment_goal, horizon))

    def get_user(self, username):
        query = "SELECT user_id, username, investment_goal, horizon FROM users WHERE username = ?"
        result = self.db.execute_query(query, (username,), fetch=True)
        return result[0] if result else None

    def get_user_by_id(self, user_id):
        query = "SELECT user_id, username, investment_goal, horizon FROM users WHERE user_id = ?"
        result = self.db.execute_query(query, (user_id,), fetch=True)
        return result[0] if result else None

    def update_user(self, user_id, investment_goal=None, horizon=None):
        # Проверяем, существует ли пользователь
        user = self.get_user_by_id(user_id)
        if not user:
            print(f"User with user_id={user_id} not found, creating new user")
            # Создаём пользователя, если его нет
            username = f"user_{user_id}"  # Генерируем временное имя
            success = self.add_user(user_id, username, investment_goal, horizon)
            if not success:
                print(f"Failed to create user with user_id={user_id}")
                return False
        else:
            updates = []
            params = []
            if investment_goal is not None:
                updates.append("investment_goal = ?")
                params.append(investment_goal)
            if horizon is not None:
                updates.append("horizon = ?")
                params.append(horizon)
            if updates:
                params.append(user_id)
                query = f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?"
                print(f"Updating user: query={query}, params={params}")
                result = self.db.execute_query(query, params)
                print(f"Update result: {result}")
                return result
        return True

    def add_portfolio(self, user_id, portfolio_name):
        # Проверяем, существует ли пользователь
        user = self.get_user_by_id(user_id)
        if not user:
            print(f"User with user_id={user_id} not found, creating new user")
            username = f"user_{user_id}"
            self.add_user(user_id, username)
        query = "INSERT INTO portfolios (user_id, portfolio_name) VALUES (?, ?)"
        return self.db.execute_query(query, (user_id, portfolio_name))

    def get_portfolios(self, user_id):
        query = "SELECT portfolio_id, portfolio_name FROM portfolios WHERE user_id = ?"
        result = self.db.execute_query(query, (user_id,), fetch=True)
        return result if result else []

    def get_portfolio(self, portfolio_id):
        """Получение данных конкретного портфеля."""
        query = "SELECT portfolio_id, user_id, portfolio_name FROM portfolios WHERE portfolio_id = ?"
        result = self.db.execute_query(query, (portfolio_id,), fetch=True)
        return result[0] if result else None

    def add_asset(self, portfolio_id, ticker, asset_type, count_assets, purchase_price, purchase_date):
        query = "INSERT INTO assets (portfolio_id, ticker, asset_type, count_assets, purchase_price, purchase_date) VALUES (?, ?, ?, ?, ?, ?)"
        return self.db.execute_query(query, (portfolio_id, ticker, asset_type, count_assets, purchase_price, purchase_date))

    def get_assets(self, portfolio_id):
        query = "SELECT asset_id, ticker, asset_type, count_assets, purchase_price, purchase_date FROM assets WHERE portfolio_id = ?"
        result = self.db.execute_query(query, (portfolio_id,), fetch=True)
        if not result or result is None:
            return []
        return result

    def update_asset(self, asset_id, ticker, asset_type, count_assets, purchase_price, purchase_date):
        query = """
            UPDATE assets 
            SET ticker = ?, asset_type = ?, count_assets = ?, purchase_price = ?, purchase_date = ?
            WHERE asset_id = ?
        """
        return self.db.execute_query(query, (ticker, asset_type, count_assets, purchase_price, purchase_date, asset_id))

    def delete_asset(self, asset_id):
        query = "DELETE FROM assets WHERE asset_id = ?"
        return self.db.execute_query(query, (asset_id,))

    def delete_portfolio(self, portfolio_id):
        """Удаление портфеля и связанных активов."""
        delete_assets_query = "DELETE FROM assets WHERE portfolio_id = ?"
        self.db.execute_query(delete_assets_query, (portfolio_id,))
        delete_portfolio_query = "DELETE FROM portfolios WHERE portfolio_id = ?"
        return self.db.execute_query(delete_portfolio_query, (portfolio_id,))

    def close(self):
        self.db.close()