import argparse
import time
from moteur_recherche import DocumentSearch

def parse_args():
    parser = argparse.ArgumentParser(description='Document Search')
    parser.add_argument('query',
                        type=str,
                        help='The query you are looking for.')
    parser.add_argument('documents', 
                        type=str,
                        help='Path to the JSON file containing documents information (id, url, title).')
    parser.add_argument('index_file_title',
                        type=str,
                        help='Path to the JSON file containing the token index for the title.')
    parser.add_argument('--index_file_content',
                        type=str,
                        help='Path to the JSON file containing the token index for the content.')
    parser.add_argument('--type_of_search',
                        type=str,
                        help='Type of the search: can be "and" or "or". If "and", all the documents \
                            that have all the tokens from you query are selected. Otherwise, all the \
                                documents with at least one of the token from the query is selected')
    
    return parser.parse_args()


def main():
    args=parse_args()
    # Parse command-line arguments
    document_search = DocumentSearch(args.documents, args.index_file_title, args.index_file_content)
    user_query = args.query

    t0=time.time()

    document_search.linear_ranking(user_query, type=args.type_of_search)
    print(f"Time taken to search: {time.time()-t0}")



if __name__ == "__main__":
    main()
