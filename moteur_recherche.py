from collections import defaultdict
import json
import math
import nltk
from nltk.corpus import stopwords  # Assuming you have nltk library installed
import time

nltk.download('stopwords')

class DocumentSearch:
    def __init__(self, documents, index_file_title, index_file_content=None):
        self.documents = self.read_json(documents)
        self.index_file_title = self.read_json(index_file_title)
        if index_file_content is not None:
            self.index_file_content=self.read_json(index_file_content)
            self.avg_doc_len_content = sum(sum(entry.get('count', 0) for entry in doc_info.values()) for doc_info in self.index_file_content.values()) / len(self.documents)
        else:
            self.index_file_content=None
            self.avg_doc_len_content=None
        self.avg_doc_len_title = sum(sum(entry.get('count', 0) for entry in doc_info.values()) for doc_info in self.index_file_title.values()) / len(self.documents)
                                  

    def read_json(self, json_file):
        with open(json_file, 'r') as file:
            index = json.load(file)
        return index

    def search(self, query, type='and'):
        tokens_query=[token.lower() for token in query.split()]
        docs_title=[]
        for token in tokens_query:
            if token in self.index_file_title:
                docs_title.append(list(self.index_file_title[token].keys()))
        
        if self.index_file_content is None:
            results_content=set()
        else:
            docs_content=[]
            for token in tokens_query:
                if token in self.index_file_content:
                    docs_content.append(list(self.index_file_content[token].keys()))
            if not docs_title and not docs_content:
                print("Aucun résultat.")
                return set(), set()
            elif docs_title and not docs_content:
                results_title=self.intersection_or_union(docs_title, type)
                return results_title, set()
            elif not docs_title and docs_content:
                results_content=self.intersection_or_union(docs_content, type)
                return set(), results_content
            else:
                results_content=self.intersection_or_union(docs_content, type)

        if not docs_title:
            print("Aucun résultat.")
            return set(), set()
        results_title=self.intersection_or_union(docs_title, type)
        
        if not results_title and not results_content:
            print('Aucun résultat')
        return results_title, results_content
    
    def intersection_or_union(self, docs_field, type):
        if type.lower()=='and':
            results_field=set(docs_field[0]).intersection(*docs_field)
        if type.lower()=='or':
            results_field=set(docs_field[0]).union(*docs_field)
        return results_field
        
    def linear_ranking(self, query, type='and'):
        results_title, results_content = self.search(query, type)

        if not results_title and not results_content:
            with open('results.json', 'w') as result_file:
                json.dump({}, result_file, default=str, indent=None)
            return None
        all_results = results_title.union(results_content)

        tokens_query = [token.lower() for token in query.split()]

        info_token_sum_title, info_token_pos_title = self.nb_tokens_and_pos_in_doc(results_title, tokens_query, 'title')
        scores_title = self.ranking_pos_nb(info_token_sum_title, info_token_pos_title, tokens_query)

        sum_bm25_content = defaultdict(int)
        scores_content = None

        if self.index_file_content is not None:
            info_token_sum_content, info_token_pos_content = self.nb_tokens_and_pos_in_doc(results_content, tokens_query, 'content')
            scores_content = self.ranking_pos_nb(info_token_sum_content, info_token_pos_content, tokens_query)

            for doc in results_content:
                for token in tokens_query:
                    if doc in self.index_file_content[token]:
                        sum_bm25_content[doc] += self.calculate_bm25_score(token, doc, 'content')
        
        sum_bm25_title = defaultdict(int)
        for doc in results_title:
            for token in tokens_query:
                if doc in self.index_file_title[token]:
                    sum_bm25_title[doc] += self.calculate_bm25_score(token, doc, 'title')

        scores_final = {}

        for doc in all_results:
            score_title = scores_title.get(doc, 0)
            score_content = scores_content.get(doc, 0) if scores_content else 0
            score_bm25_title = sum_bm25_title.get(doc, 0)
            score_bm25_content = sum_bm25_content.get(doc, 0)

            scores_final[doc] = 10 * (0.7 * score_title + 0.2 * score_content) + 0.7 * score_bm25_title + 0.3 * score_bm25_content

        sorted_scores_final = dict(sorted(scores_final.items(), key=lambda item: item[1], reverse=True))

        results_dict = {doc['title']: doc['url'] for doc in self.documents if str(doc['id']) in sorted_scores_final}

        with open('results.json', 'w') as result_file:
            json.dump(results_dict, result_file, default=str, indent=None)

        return "The results are in results.json"

    def calculate_bm25_score(self, token, doc_id, field, k1=1.2, b=0.75):
        field_params = {'title': (self.index_file_title, self.avg_doc_len_title),
                        'content': (self.index_file_content, self.avg_doc_len_content)}

        index, avg_doc_len = field_params[field]

        corpus_size = len(self.documents)
        token_info = index.get(token, {})

        # token frequency
        tf = token_info.get(doc_id, {}).get('count', 0)
        # length of the document in the field affected (title or content)
        doc_len = sum(entry.get('count', 0) for entry in token_info.values())
        # documet frequency
        doc_freq = len(token_info)

        idf = math.log((corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        term1 = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (doc_len / avg_doc_len)) + tf)

        return idf * term1

    def ranking_pos_nb(self, info_token_sum, info_token_pos, tokens_query):

        for doc in info_token_sum:
            sum = 0
            info_token_pos_doc = []

            for token in tokens_query:
                try:
                    sum += self.index_file_title[token][doc]['count']
                    info_token_pos_doc.append(self.index_file_title[token][doc]['positions'])
                except:
                    continue

            info_token_sum[doc] = 0.3 * sum / len(tokens_query) + 0.7 * (1 - self.same_order(info_token_pos[doc]))

        return info_token_sum

    def nb_tokens_and_pos_in_doc(self, results, tokens_query, type):
        if type == 'title':
            index = self.index_file_title
        elif type == 'content':
            index = self.index_file_content

        info_token_sum = {}
        info_token_pos = {}

        for doc in results:
            sum = 0
            info_token_pos_doc = []

            for i, token in enumerate(tokens_query):
                try:
                    sum += index[token][doc]['count']
                    info_token_pos_doc.append(index[token][doc]['positions'])
                except:
                    continue

            if token.lower() in set(stopwords.words('french')):
                info_token_sum[doc] = (1/4) * sum
            else:
                info_token_sum[doc] = 4 * sum

            info_token_pos[doc] = info_token_pos_doc

        return info_token_sum, info_token_pos

    def same_order(self, list_positions):
        min_pos=-1
        for list_positions_doc in list_positions: #in range(1, len(list_positions)): # liste_positions_doc in list_positions:
            new_pos = [element for element in list_positions_doc if element > min_pos]
            if not new_pos:
                return False
            min_pos=min(new_pos)
        return True

if __name__ == "__main__":

    # Créer une instance de DocumentSearch
    document_search = DocumentSearch('documents.json', 'title_pos_index.json', 'content_pos_index.json')

    # Lire la requête depuis le terminal
    user_query = input("Entrez votre requête : ")

    # Effectuer la recherche
    #search_results_title = document_search.search(user_query, type='or')

    #print(search_results_title)
    #print(len(search_results_title))
    t0=time.time()

    document_search.linear_ranking(user_query, type='and')
    print(time.time()-t0)
