import json
import nltk
import math
from collections import defaultdict
from nltk.corpus import stopwords

nltk.download('stopwords')

class DocumentSearch:
    def __init__(self, documents, index_file_title, index_file_content=None):
        self.documents = self.read_json(documents)
        self.index_file_title = self.read_json(index_file_title)
        self.index_file_content = self.read_json(index_file_content) if index_file_content else None
        self.avg_doc_len_title = self.calculate_avg_doc_len(self.index_file_title)
        self.avg_doc_len_content = self.calculate_avg_doc_len(self.index_file_content) if self.index_file_content else None

    def read_json(self, json_file):
        with open(json_file, 'r') as file:
            index = json.load(file)
        return index

    def calculate_avg_doc_len(self, index):
        if not index:
            return None
        return sum(sum(entry.get('count', 0) for entry in doc_info.values()) for doc_info in index.values()) / len(self.documents)

    def search(self, query, type='and'):
        tokens_query = [token.lower() for token in query.split()]
        docs_title = [list(self.index_file_title.get(token, {}).keys()) for token in tokens_query]

        if type.lower() == 'and':
            results_title = set.intersection(*map(set, docs_title))
        elif type.lower() == 'or':
            results_title = set.union(*map(set, docs_title))

        results_content = set()

        if self.index_file_content:
            docs_content = [list(self.index_file_content.get(token, {}).keys()) for token in tokens_query]

            if type.lower() == 'and':
                results_content = set.intersection(*map(set, docs_content))
            elif type.lower() == 'or':
                results_content = set.union(*map(set, docs_content))

        return results_title, results_content

    def linear_ranking(self, query, type='and'):
        results_title, results_content = self.search(query, type)
        all_results = results_title.union(results_content)
        tokens_query = [token.lower() for token in query.split()]

        scores_title = self.calculate_scores(self.index_file_title, self.avg_doc_len_title, results_title, tokens_query)
        sum_bm25_content = self.calculate_scores(self.index_file_content, self.avg_doc_len_content, results_content, tokens_query) if self.index_file_content else {}

        sum_bm25_title = self.calculate_scores(self.index_file_title, self.avg_doc_len_title, results_title, tokens_query)

        scores_final = defaultdict(float)

        for doc in all_results:
            try:
                scores_final[doc] = 10 * (0.7 * scores_title[doc] + 0.2 * sum_bm25_content.get(doc, 0)) + 0.7 * sum_bm25_title.get(doc, 0) + 0.3 * sum_bm25_content.get(doc, 0)
            except KeyError:
                scores_final[doc] = 0

        sorted_scores_final = dict(sorted(scores_final.items(), key=lambda item: item[1], reverse=True))

        results_dict = {doc['title']: doc['url'] for doc in self.documents if str(doc['id']) in sorted_scores_final}

        with open('results.json', 'w') as result_file:
            json.dump(results_dict, result_file, indent=None)

        print("The results are in results.json")

    def calculate_scores(self, index, avg_doc_len, results, tokens_query, field=None):
        scores = defaultdict(float)

        for doc in results:
            try:
                token_scores = (index[token][doc].get('count', 0) / len(tokens_query) for token in tokens_query)
                scores[doc] = 0.3 * sum(token_scores) + 0.7 * (1 - self.same_order(index[token][doc].get('positions', [])) for token in tokens_query)
            except KeyError:
                scores[doc] = 0

        return scores

    def nb_tokens_and_pos_in_doc(self, results, tokens_query, type):
        if type == 'title':
            index = self.index_file_title
        elif type == 'content':
            index = self.index_file_content

        info_token_sum = {}
        info_token_pos = {}

        for doc in results:
            _sum = 0
            info_token_pos_doc = []

            for token in tokens_query:
                try:
                    _sum += index[token][doc]['count']
                    info_token_pos_doc.append(index[token][doc]['positions'])
                except KeyError:
                    continue

            if token.lower() in set(stopwords.words('french')):
                info_token_sum[doc] = (1/4) * _sum
            else:
                info_token_sum[doc] = 4 * _sum

            info_token_pos[doc] = info_token_pos_doc

        return info_token_sum, info_token_pos

    def same_order(self, list_positions):
        min_pos = -1

        for list_positions_doc in list_positions:
            new_pos = [element for element in list_positions_doc if element > min_pos]

            if not new_pos:
                return False

            min_pos = min(new_pos)

        return True


if __name__ == "__main__":
    document_search = DocumentSearch('documents.json', 'title_pos_index.json', 'content_pos_index.json')
    user_query = input("Enter your query: ")
    document_search.linear_ranking(user_query, type='or')

