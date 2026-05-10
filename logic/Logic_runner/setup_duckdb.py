
import duckdb, os, pandas as pd

DATA = "my_benchmark/logic/data"
OUT  = f"{DATA}/logic.duckdb"
if os.path.exists(OUT):
    os.remove(OUT)

statements = pd.read_csv(f"{DATA}/folio_statements_20.csv")
formulas   = pd.read_csv(f"{DATA}/folio_formulas.csv")
labels     = pd.read_csv(f"{DATA}/folio_entailment_labels_corrected (2).csv")

con = duckdb.connect(OUT)
con.register("s", statements); con.execute("CREATE TABLE Statements AS SELECT * FROM s")
con.register("f", formulas);   con.execute("CREATE TABLE Formulas   AS SELECT * FROM f")
con.register("l", labels);     con.execute("CREATE TABLE EntailLabels AS SELECT * FROM l")
con.close()
print(f"OK -> {OUT}: {len(statements)} statements, {len(formulas)} formulas, {len(labels)} labels")
