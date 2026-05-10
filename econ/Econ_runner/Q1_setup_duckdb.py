
import duckdb, pandas as pd, os

os.makedirs('my_benchmark/econ/data', exist_ok=True)
db_path = 'my_benchmark/econ/data/econ.duckdb'
if os.path.exists(db_path):
    os.remove(db_path)

commodities = pd.read_csv('my_benchmark/econ/data/commodities (2).csv')
factors = pd.read_csv('my_benchmark/econ/data/factors.csv')

commodities['commodity_info'] = commodities['commodity_name'].astype(str) + ' (' + commodities['category'].astype(str) + ')'
factors['factor_info'] = '[' + factors['side'].astype(str) + '] ' + factors['factor_description'].astype(str)

conn = duckdb.connect(db_path)
conn.register('c_df', commodities)
conn.register('f_df', factors)
conn.execute('CREATE TABLE Commodities AS SELECT * FROM c_df')
conn.execute('CREATE TABLE Factors AS SELECT * FROM f_df')
print('Tables:', conn.execute('SHOW TABLES').fetchall())
print('Commodities:', conn.execute('SELECT COUNT(*) FROM Commodities').fetchone())
print('Factors:', conn.execute('SELECT COUNT(*) FROM Factors').fetchone())
conn.close()
print(f'Saved -> {db_path}')
