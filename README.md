# Search engine project README

## Introduction
This Python-based project provides a simple way to create a minimal search engine from positionnal indexes.

## Projects components

### 1. Search engine creation (`moteur_recherche.py`)

The `moteur_recherche.py` script defines the core functionality of the search engine creation. It utilizes the `nltk' library to have the set of French stopwords.

The creation of the search engine follows these steps: tokenizing the user query, using the same tokenizing as in the positional indexes (here, splitting on spaces) and the tokens from the query are searched in the indexes (title and content if there is a content index). All the documents found are then ranked thanks to a linear ranking based on the positions of the tokens and their counts, including a BM25 score. The stopwords have less weights than the other words since their count in the document is divided by 4 and the count of non-stopwords in multiplied by 2. The positions of the tokens are taken into account by checking if their positions in the document is the same as in the query. Finally, there are 2 different scores calculated for each kind of score (BM25 and positions+count scores), one for the title and the other one for the content (if the index for the content is given). The score for the title has more weight than the score for the content (70% for the title, 30% for the content).

### 2. Example usage (`main.py`)

The `main.py` script demonstrates an example of using the search engine, using an arguments parsers to facilitate interactions with the user in the Terminal.

## Setup

1. **Dependecies:** Ensure that you have the necessary Python packages installed. You can install them using the following:

    ```bash
   pip install nltk
   ```
2. **Run the Project:**

To run the search engine, you can use the following command format:

```bash
python main.py "pourquoi erreur 404" "documents.json" "title_pos_index.json" --index_file_content "content_pos_index.json" --type_of_search "and"
```
The first argument is your query, the second one is the path to the json file containing all the documents with its URL, its title and its ID, the third one is the path to the positional index for the title. The fourth argument is optionnal: it is the path to the positional index for content. The last argument is also optional and is the type of search: either the results are the documents that contain all the tokens of the query (`--type_of_search "and"` or no specification for this argument) or the all the documents with at least one of the token of the query (`--type_of_search "or"`).

You might need to change the `python` command by `python3`.

For more information on the parameters, you can use the following command:

```bash
python main.py --help
```

## Contributor

- Julia Toukal
