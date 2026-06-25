# Quote-to-Fulfillment Agent
Draft-only LangGraph agent for print quote email threads.
It validates the sender domain in CRM before reading quote details.
It extracts size, quantity, material, deadline, and country.
It creates at most one customer draft and one internal note per thread.
Install dependencies with `python -m pip install -r requirements.txt`.
Copy `.env.example` values into your environment and set internal API URLs.
Set both `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` to enable LLM extraction.
Run the graph by importing `run_quote_agent` from `graph.py`.
Run tests with `python -m unittest discover -s tests -v`.
