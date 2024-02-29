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
    print("in extract keywords")
    rake = Rake()
    rake.extract_keywords_from_text(query)
    keywords = rake.get_ranked_phrases()
    return " ".join(keywords)


def scrape_paper(args):
    print("in scrape paper")
    keyword_query = extract_keywords(args.query)
    print('keyword extracted')
    results = []
    print(args.num_result)
    print(type(args.num_result))
    search = arxiv.Search(query=keyword_query, 
                          max_results = 5, 
                          sort_by = arxiv.SortCriterion.Relevance)
    print('search done')
    papers = list(search.results())
    print(papers)
    for i, p in enumerate(papers):
        text = ""
        file_path = f"/Users/sahithya/Documents/Study Materials/Machine Learning/LLM/src/data/data_{i}.pdf"
        p.download_pdf(filename=file_path)

        with open(f"/Users/sahithya/Documents/Study Materials/Machine Learning/LLM/src/data/data_{i}.pdf", "rb") as file:
            pdf = PyPDF2.PdfReader(file)

            for page in range(len(pdf.pages)):
                page_obj = pdf.pages[page]

                text += page_obj.extract_text() + " "

        os.unlink(file_path)
        paper_doc = {"url": p.pdf_url, "title": p.title, "text": text}
        results.append(paper_doc)
    return results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help = "type the query to search",type=str)
    parser.add_argument("--num_result", help="the number of results to return", type =str)
    args = parser.parse_args()
    #print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    print("entering scrape paper")
    results = scrape_paper(args) #pass the available query to the scrape paper
    
    #write the results in json. This json will be used by the streamlit app
    for i, r in enumerate(results):
        with open(f"/Users/sahithya/Documents/Study Materials/Machine Learning/LLM/src/data/data_{i}.json", "w") as f:
            json.dump(r, f)
