from collections import defaultdict
import json
import math
import nltk
from nltk.corpus import stopwords  # Assuming you have nltk library installed
import time

nltk.download('stopwords')

class DocumentSearch:
    """
    DocumentSearch class provides functionality for searching and analyzing documents using an inverted index approach.

    Parameters:
    - documents (str): Path to the JSON file containing the documents to be indexed.
    - index_file_title (str): Path to the JSON file containing the inverted index for document titles.
    - index_file_content (str, optional): Path to the JSON file containing the inverted index for document content.
        If not provided, content-related features will not be available.

    Attributes:
    - documents (list): List of documents loaded from the specified JSON file.
    - index_file_title (dict): Inverted index for document titles, loaded from the specified JSON file.
    - index_file_content (dict or None): Inverted index for document content, loaded from the specified JSON file.
        If content index is not provided, set to None.
    - avg_doc_len_title (float): Average document length calculated from the title index.
    - avg_doc_len_content (float or None): Average document length calculated from the content index.
        If content index is not provided, set to None.

    Note: The class assumes that the documents, title index, and content index follow a specific JSON format.
    """

    def __init__(self, documents, index_file_title, index_file_content=None):
        """
        Initializes the DocumentSearch object with the provided parameters.

        Parameters:
        - documents (str): Path to the JSON file containing the documents to be indexed.
        - index_file_title (str): Path to the JSON file containing the inverted index for document titles.
        - index_file_content (str, optional): Path to the JSON file containing the inverted index for document content.
          If not provided, content-related features will not be available.
        """
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
        """
        Reads a JSON file from the given file path and returns the loaded data as a dictionary.

        Parameters:
        - json_file (str): Path to the JSON file to be read.

        Returns:
        - Union[dict, None]: Loaded data as a dictionary or None if the file cannot be read or is not in a valid JSON format.
        """
        with open(json_file, 'r') as file:
            index = json.load(file)
        return index

    def search(self, query, type='and'):
        """
        Searches for documents based on the given query using an inverted index approach.

        Parameters:
        - query (str): The search query to be processed.
        - type (str, optional): The search type, either 'and' or 'or'. Defaults to 'and'.

        Returns:
        - Tuple[Set[int], Set[int]]: Two sets representing the document IDs matching the query for titles and content, respectively.
          Returns an empty set for titles and content if no results are found for each category.

        Note:
        - The search is case-insensitive, and the query is tokenized into lowercase tokens.
        - If content index is not available, only title search results will be returned.
        """
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
        """
        Performs the intersection or union operation on a list of document IDs based on the specified type.

        Parameters:
        - docs_field (List[List[int]]): List of lists containing document IDs.
        - type (str): The operation type, either 'and' or 'or'.

        Returns:
        - Set[int]: Set of document IDs resulting from the intersection or union operation.

        Note:
        - The intersection operation returns the common document IDs across all lists.
        - The union operation returns all unique document IDs present in any of the lists.
        """
        if type.lower()=='and':
            results_field=set(docs_field[0]).intersection(*docs_field)
        if type.lower()=='or':
            results_field=set(docs_field[0]).union(*docs_field)
        return results_field
        
    def linear_ranking(self, query, type='and'):
        """
        Performs linear ranking on documents based on a combination of positional and BM25 scores.

        Parameters:
        - query (str): The search query to be processed.
        - type (str, optional): The search type, either 'and' or 'or'. Defaults to 'and'.

        Returns:
        - Optional[str]: If results are found, saves the ranked results in 'results.json' and returns a success message.
                    If no results are found, returns None.

        Note:
        - The ranking combines positional scores from both title and content searches, along with BM25 scores.
        - Results are saved in 'results.json' file.
        """
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
        """
        Calculates the BM25 score for a token in a document and a specific field (title or content).

        Parameters:
        - token (str): The token for which the BM25 score is calculated.
        - doc_id (int): The document ID for which the score is calculated.
        - field (str): The field in the document (either 'title' or 'content').
        - k1 (float, optional): BM25 parameter controlling term saturation. Defaults to 1.2.
        - b (float, optional): BM25 parameter controlling the length normalization. Defaults to 0.75.

        Returns:
        - float: The calculated BM25 score for the given token, document, and field.

        Note:
        - BM25 score calculation is based on the token's term frequency (tf), document length, document frequency, and corpus size.
        """
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
        """
        Performs ranking based on positional information and token counts in documents.

        Parameters:
        - info_token_sum (Dict[int, float]): Dictionary with document IDs as keys and initial positional sum as values.
        - info_token_pos (Dict[int, List[List[int]]]): Dictionary with document IDs as keys and token positions as values.
        - tokens_query (List[str]): List of tokens in the query.

        Returns:
        - Dict[int, float]: Updated dictionary with document IDs as keys and final positional scores as values.

        Note:
        - The ranking considers both the sum of token counts and the order of token positions in documents.
        - The final positional score is a weighted combination of token count and order.
        """
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
        """
        Calculates the sum of token counts and token positions in documents based on search results and query tokens.

        Parameters:
        - results (Set[int]): Set of document IDs for which the information is calculated.
        - tokens_query (List[str]): List of tokens in the query.
        - type (str): The type of information to calculate ('title' or 'content').

        Returns:
        - Tuple[Dict[int, float], Dict[int, List[List[int]]]]: Two dictionaries representing the sum of token counts and
          token positions for each document in the specified type.

        Note:
        - The sum of token counts is weighted based on the presence of French stopwords in the query.
        """
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
        """
        Checks if the positions in the provided list of lists maintain the same order.

        Parameters:
        - list_positions (List[List[int]]): List of lists containing token positions.

        Returns:
        - bool: True if the positions maintain the same order, False otherwise.

        Note:
        - The function iterates through the list of lists, checking if each subsequent list maintains the same order.
        """
        min_pos=-1
        for list_positions_doc in list_positions: #in range(1, len(list_positions)): # liste_positions_doc in list_positions:
            new_pos = [element for element in list_positions_doc if element > min_pos]
            if not new_pos:
                return False
            min_pos=min(new_pos)
        return True
