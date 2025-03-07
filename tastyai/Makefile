.PHONY: install
install:
	poetry lock && poetry install

.PHONY: vectorize
vectorize:
	make install && poetry run python scripts/vectorize.py

.PHONY: run_streamlit
run_streamlit:
	make vectorize && poetry run streamlit run tastyai/chat.py


.PHONY: run_terminal
run_terminal:
	make vectorize && poetry run python tastyai/terminal.py
