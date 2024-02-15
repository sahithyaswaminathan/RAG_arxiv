import arxiv
import argparse
import nltk
import PyPDF2 #pdf reading and manipulation
import os 
import json
from rake_nltk import Rake

nltk.download("stopwords")
nltk.download("punkt")

def extract_keywords(query):
    rake = Rake()
    rake.extract_keywords_from_text(query)
    keywords = rake.get_ranked_phrases()
    return " ".join(keywords)


def scrape_paper(args):
    keyword_query = extract_keywords(args.query)

    results = []
    search = arxiv.Search(query=keyword_query, 
                          max_results = args.num_result, 
                          sort_by = arxiv.SortCriteriorn.Relevance)
    papers = list(search.results())
    for i, p in enumerate(papers):
        text = ""
        file_path = f"src/data/data_{i}.pdf"
        p.download_pdf(filename=file_path)

        with open(f"src/data/data_{i}.pdf", "rb") as file:
            pdf = PyPDF2.PdfReader(file)

            for page in range(len(pdf.pages)):
                page_obj = pdf.pages[page]

                text += page_obj.extract_text() + " "

        os.unlink(file_path)
        paper_doc = {"url": p.pdf_url, "title": p.title, "text": text}
        results.append(paper_doc)
    return results
    

if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help = "type the query to search",type=str)
    parser.add_argument("--num_result", help="the number of results to return", type =int)
    args = parser.parse_args()

    results = scrape_paper(args) #pass the available query to the scrape paper
    
    #write the results in json. This json will be used by the streamlit app
    for i, r in enumerate(results):
        with open(f"src/data/data_{i}.json", "w") as f:
            json.dump(r, f)
