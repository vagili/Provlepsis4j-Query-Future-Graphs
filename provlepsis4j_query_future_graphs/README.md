# Provlepsis4j — Streamlit Full App

This is a **single-process** Streamlit implementation of the Provlepsis4j application.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Neo4j connection

Open the app and use the **Connection** section.

Typical values:
- `NEO4J_URI`: `bolt://localhost:7687`
- `NEO4J_USER`: `neo4j`
- `NEO4J_PASSWORD`: your password
- `Default database`: `neo4j` (or your dataset DB)

## Notes

- Provlepsis4j features require the following plugins installed in the Neo4j DBMS:
  - **GDS**
  - **APOC**

