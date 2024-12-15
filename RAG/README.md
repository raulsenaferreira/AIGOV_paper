## Install reqs
```
pip install -r requirements.txt
```
export gemini_api_key=""
## Transofrm pdf into markdown
```
docling China_The_Personal_Information_Protection_Law.pdf
```
## Index document
- Change `WORKING_DIR` to the path where the data will be `indexed index_lightrag.py`
- Change `document` to the document to indexi `index_lightrag.py`
```
python index_lightrag.py
```
- Change `WORKING_DIR` to the path where the data is indexed `generte_answer_lightrag.py`
- Change `question` to the question you want to ask in `generte_answer_lightrag.py`
- Change `answer_path` to the question you want to ask in `generte_answer_lightrag.py`

```
python index_lightrag.py
```

the answer will be generated in a markdown format
