import json
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
import math
from collections import defaultdict
import time

#nltk.download('punkt')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
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
        if type.lower()=='and':
            results_title=set(docs_title[0]).intersection(*docs_title)
        if type.lower()=='or':
            results_title=set(docs_title[0]).union(*docs_title)
        if self.index_file_content is None:
            results_content=set()
        else:
            docs_content=[]
            for token in tokens_query:
                if token in self.index_file_content:
                    docs_content.append(list(self.index_file_content[token].keys()))
            if type.lower()=='and':
                results_content=set(docs_content[0]).intersection(*docs_content)
            if type.lower()=='or':
                results_content=set(docs_content[0]).union(*docs_content)
        return results_title, results_content
    
    def linear_ranking(self, query, type='and'):
        results_title, results_content=self.search(query, type)
        results_title=set()
        print(len(results_title), len(results_content))
        all_results=results_title.union(results_content)

        print(len(all_results))

        tokens_query=[token.lower() for token in query.split()]

        info_token_sum_title, info_token_pos_title=self.nb_tokens_and_pos_in_doc(results_title, tokens_query, 'title')
        scores_title=self.ranking_pos_nb(info_token_sum_title, info_token_pos_title,tokens_query)

        sum_bm25_content={}
        if self.index_file_content is not None:
            info_token_sum_content, info_token_pos_content=self.nb_tokens_and_pos_in_doc(results_content, tokens_query, 'content')
            scores_content=self.ranking_pos_nb(info_token_sum_content, info_token_pos_content,tokens_query)
            for doc in results_content:
                sum_bm25_content[doc]=0
                for token in tokens_query:
                    if doc in self.index_file_content[token]:
                        sum_bm25_content[doc]+=self.calculate_bm25_score(token, doc, 'content')
        else:
            for doc in results_content:
                sum_bm25_content[doc]=0

        sum_bm25_title={}
        for doc in results_title:
            sum_bm25_title[doc]=0
            for token in tokens_query:
                if doc in self.index_file_title[token]:
                    sum_bm25_title[doc]+=self.calculate_bm25_score(token, doc, 'title')


        scores_final={}

        for doc in all_results:
            try:
                score_title=scores_title[doc]
            except:
                score_title=0
            try:
                score_content=scores_content[doc]
            except:
                score_content=0
            try:
                score_bm25_title=sum_bm25_title[doc]
            except:
                score_bm25_title=0
            try:
                score_bm25_content=sum_bm25_content[doc]
            except:
                score_bm25_content=0
            scores_final[doc]=10*(0.7*score_title+0.2*score_content)+0.7*score_bm25_title+0.3*score_bm25_content
        
        sorted_scores_final = dict(sorted(scores_final.items(), key=lambda item: item[1], reverse=True))

        print(len(sorted_scores_final))
        results_dict={}
        for doc in self.documents:
            for id_doc in sorted_scores_final:
                if str(doc['id'])==id_doc:
                    results_dict[doc['title']]=doc['url']
        results_dict2 = {doc['title']: doc['url'] for doc in self.documents if str(doc['id']) in sorted_scores_final}
        print(len(results_dict2))
        print(len(results_dict))
        results_dict3 = {doc['title']: doc['url'] for id_doc, doc in zip(sorted_scores_final, self.documents)}
        print(len(results_dict3))
        with open('results.json', 'w') as result_file:
            json.dump(results_dict, result_file, indent=None)

        print("The results are in results.json")


    def calculate_bm25_score(self, token, doc_id, field, k1=1.2, b=0.75):
        if field=='title':
            index=self.index_file_title
            avg_doc_len=self.avg_doc_len_title
        if field=='content':
            index=self.index_file_content
            avg_doc_len=self.avg_doc_len_content

        corpus_size=len(self.documents)    
        token_info = index.get(token, {})
    
        tf = token_info.get(doc_id, {}).get('count', 0)
        doc_len = sum(entry.get('count', 0) for entry in token_info.values())
        doc_freq = len(token_info)

        idf = math.log((corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        term1 = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (doc_len / avg_doc_len)) + tf)

        return idf * term1


    def ranking_pos_nb(self, info_token_sum, info_token_pos, tokens_query):
        scores_doc={}
        for doc in info_token_sum:
            scores_doc[doc]=0.3*info_token_sum[doc]/len(tokens_query)+0.7*(1-self.same_order(info_token_pos[doc]))
        return scores_doc


    def nb_tokens_and_pos_in_doc(self, results, tokens_query, type):
        if type=='title':
            index=self.index_file_title
        elif type=='content':
            index=self.index_file_content
        # Summing the count of the tokens in the title
        info_token_sum={}
        info_token_pos={}
        for doc in results:
            sum=0
            info_token_pos_doc=[]
            for token in tokens_query:
                try:
                    sum+=index[token][doc]['count']
                    info_token_pos_doc.append(index[token][doc]['positions'])
                except:
                    continue
            if token.lower() in set(stopwords.words('french')):
                info_token_sum[doc]=(1/4)*sum 
            else:
                info_token_sum[doc]=4*sum
            info_token_pos[doc]=info_token_pos_doc
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

    document_search.linear_ranking(user_query, type='or')
    print(time.time()-t0)
    #print(set(document_search.calculate_len_doc().items())==set(document_search.calculate_len_doc2().items()))
    #print(document_search.calculate_len_doc().keys()==document_search.calculate_len_doc2().keys())

    #print(document_search.calculate_len_doc())
    # Afficher les résultats
    #for result in search_results:
    #    print(f"Titre : {result['title']}, URL : {result['url']}")
