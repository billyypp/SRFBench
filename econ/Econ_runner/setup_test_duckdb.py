
import duckdb, pandas as pd, os

data_dir = "my_benchmark/econ/data/test"
db_path = os.path.join(data_dir, "econ.duckdb")
if os.path.exists(db_path):
    os.remove(db_path)

commodities = pd.read_csv(os.path.join(data_dir, "commodities.csv"))
factors = pd.read_csv(os.path.join(data_dir, "factors.csv"))

commodities["commodity_info"] = (commodities["commodity_name"].astype(str)
                                 + " (" + commodities["category"].astype(str) + ")")
factors["factor_info"] = ("[" + factors["side"].astype(str) + "] "
                          + factors["factor_description"].astype(str))

conn = duckdb.connect(db_path)
conn.register("c_df", commodities)
conn.register("f_df", factors)
conn.execute("CREATE TABLE Commodities AS SELECT * FROM c_df")
conn.execute("CREATE TABLE Factors AS SELECT * FROM f_df")
print("Tables:", conn.execute("SHOW TABLES").fetchall())
print("Commodities:", conn.execute("SELECT COUNT(*) FROM Commodities").fetchone())
print("Factors:", conn.execute("SELECT COUNT(*) FROM Factors").fetchone())
conn.close()
print(f"Saved -> {db_path}")
