.PHONY : help clean dockerize extract_citations preprecess_tei extract_metadata build_lsa_model build_lda_model build_feature_frame

help :
	@echo "Usage: make [command]"
	@echo "    clean"
	@echo "        Remove all build and Python artifacts"
	@echo "    dockerize"
	@echo "        Build docker container"
	@echo "    run"
	@echo "        Run full analysis of the repository"
	@echo "    extract_citations"
	@echo "        Extract citation sentences from raw data (tei-xml of the pdfs)"
	@echo "    preprecess_tei"
	@echo "        Preprocess tei-xml files"
	@echo "    extract_metadata"
	@echo "        Extract metadata from pdf raw data"
	@echo "    build_lsa_model"
	@echo "        Build latent semantic analysis model"
	@echo "    build_lda_model"
	@echo "        Build latent dirichlet allocation model"
	@echo "    build_feature_frame"
	@echo "        Build the feature frame for the machine learning classifier"

clean :
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

dockerize :
	docker build -t deep-cenic .

extract_citations : preprecess_tei data/interim/CITATION.csv

preprecess_tei: src/preprocess_tei.py
	docker run -ti -v "$(PWD)":/opt/workdir deep-cenic python src/preprocess_tei.py

data/interim/CITATION.csv : data/raw/ARTICLE.csv data/raw/LR_CP.csv src/citation_extraction.py
	docker run -ti -v "$(PWD)":/opt/workdir deep-cenic python src/citation_extraction.py

extract_metadata : data/interim/CP.csv data/interim/LR.csv data/interim/LR_CP.csv

data/interim/CP.csv : data/raw/ARTICLE.csv data/raw/LR_CP.csv src/cp_metadata_extraction.py
	docker run -ti -v "$(PWD)":/opt/workdir deep-cenic python src/cp_metadata_extraction.py

data/interim/LR.csv : data/raw/LR.csv data/raw/LR_CP.csv src/lr_metadata_extraction.py
	docker run -ti -v "$(PWD)":/opt/workdir deep-cenic python src/lr_metadata_extraction.py

build_lsa_model : models/lsa/title.model models/lsa/abstract.model

models/lsa/title.model : data/interim/CP.csv data/interim/LR.csv src/lsa_model.py
	docker run -ti -v "$(PWD)":/opt/workdir deep-cenic python src/lsa_model.py

models/lsa/abstract.model : data/interim/CP.csv data/interim/LR.csv src/lsa_model.py
	docker run -ti -v "$(PWD)":/opt/workdir deep-cenic python src/lsa_model.py

data/interim/LR_CP.csv : data/raw/ARTICLE.csv data/raw/LR_CP.csv data/interim/CP.csv data/interim/LR.csv build_lsa_model src/lr_cp_metadata_extraction.py
	docker run -ti -v "$(PWD)":/opt/workdir deep-cenic python src/lr_cp_metadata_extraction.py

build_lda_model : models/lda/lda.model

models/lda/lda.model : data/interim/CITATION.csv src/lda_model.py src/config.py
	docker run -ti -v "$(PWD)":/opt/workdir deep-cenic python src/lda_model.py

build_feature_frame : data/processed/FEATURE_FRAME.csv

data/processed/FEATURE_FRAME.csv : build_lda_model extract_metadata data/raw/LR_CP_CODING.csv src/feature_frame.py
	docker run -ti -v "$(PWD)":/opt/workdir deep-cenic python src/feature_frame.py

run : clean build_feature_frame
