from Bio import Entrez
import json

Entrez.email = "your@email.com"
handle = Entrez.esearch(db="pubmed", term="antimicrobial resistance Europe", retmax=200)
ids = Entrez.read(handle)["IdList"]

records = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
parsed = Entrez.read(records)

with open("data/abstracts.jsonl", "w") as f:
    for article in parsed["PubmedArticle"]:
        med = article["MedlineCitation"]
        pmid = str(med["PMID"])
        title = str(med["Article"]["ArticleTitle"])
        abstract = str(med["Article"].get("Abstract", {}).get("AbstractText", [""])[0])
        year = str(med["Article"]["Journal"]["JournalIssue"]["PubDate"].get("Year", ""))
        f.write(json.dumps({"pmid": pmid, "title": title, "abstract": abstract, "year": year}) + "\n")

print("Done")
