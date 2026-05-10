
import duckdb, os, pandas as pd

DATA = "my_benchmark/paper/data"
OUT  = f"{DATA}/paper.duckdb"
if os.path.exists(OUT):
    os.remove(OUT)

claims = pd.read_csv(f"{DATA}/scifact_claims_30.csv")
papers_60 = pd.read_csv(f"{DATA}/scifact_papers_60.csv", encoding="latin-1")
papers_full = pd.read_csv(f"{DATA}/scifact_papers.csv", encoding="latin-1")
papers_full["_row_idx"] = range(len(papers_full))

con = duckdb.connect(OUT)
con.register("c", claims);    con.execute("CREATE TABLE Claims AS SELECT * FROM c")
con.register("p", papers_60); con.execute("CREATE TABLE Papers AS SELECT * FROM p")
con.register("pf", papers_full); con.execute("CREATE TABLE PapersFull AS SELECT * FROM pf")
con.close()
print(f"OK -> {OUT}: claims={len(claims)} papers={len(papers_60)} full={len(papers_full)}")
