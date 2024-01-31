import json
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class DocumentSearch:
    def __init__(self, documents, index_file_title, index_file_content=None):
        self.documents = self.read_json(documents)
        self.index_file_title = self.read_json(index_file_title)
        if index_file_content is not None:
            self.index_file_content=self.read_json(index_file_content)
        else:
            self.index_file_content=None
                                            

    def read_json(self, json_file):
        with open(json_file, 'r') as file:
            index = json.load(file)
        return index

    def search(self, query):
        tokens_query=[token.lower() for token in query.split()]
        docs_title=[]
        for token in tokens_query:
            if token in self.index_file_title:
                docs_title.append(list(self.index_file_title[token].keys()))
        results_title=set(docs_title[0]).intersection(*docs_title)
        if self.index_file_content is None:
            results_content=None
        else:
            docs_content=[]
            for token in tokens_query:
                if token in self.index_file_content:
                    docs_content.append(list(self.index_file_content[token].keys()))
            results_content=set(docs_content[0]).intersection(*docs_content)
        return results_title, results_content
    
    def linear_ranking(self, query):

        results_title, results_content=self.search(query)

        tokens_query=[token.lower() for token in query.split()]

        # Summing the count of the tokens in the title
        info_token_sum={}
        info_token_pos={}
        for doc in results_title:
            sum=0
            info_token_pos_doc=[]
            for token in tokens_query:
                sum+=self.index_file_title[token][doc]['count']
                info_token_pos_doc.append(self.index_file_title[token][doc]['positions'])
            info_token_sum[doc]=sum
            info_token_pos[doc]=info_token_pos_doc
        
        return info_token_sum, info_token_pos


if __name__ == "__main__":

    # Créer une instance de DocumentSearch
    document_search = DocumentSearch('documents.json', 'title_pos_index.json', 'content_pos_index.json')

    # Lire la requête depuis le terminal
    user_query = input("Entrez votre requête : ")

    # Effectuer la recherche
    search_results_title = document_search.search(user_query)

    search_results_content = document_search.search(user_query)

    print(search_results_title)
    #print(len(search_results_title))

    print(document_search.linear_ranking(user_query))

    # Afficher les résultats
    #for result in search_results:
    #    print(f"Titre : {result['title']}, URL : {result['url']}")
