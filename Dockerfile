FROM python:3.6
WORKDIR /opt/workdir

COPY requirements.txt ./

RUN apt-get update && apt-get install -y libxml2-dev zlibc python3-lxml \
&& pip install -U pip && pip install --no-cache-dir -r requirements.txt

RUN pip3 install nltk
RUN pip3 install -U nltk[twitter]

RUN [ "python", "-c", "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('words')" ]
