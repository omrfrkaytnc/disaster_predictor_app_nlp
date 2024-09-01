FROM python:3.11-slim
RUN /usr/local/bin/python -m pip install --upgrade pip
WORKDIR /🏠About
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT [ "streamlit", "run" ]
CMD [ "🏠About.py" ]