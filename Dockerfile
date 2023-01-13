FROM python
COPY requirements.txt /tmp/
RUN ls
RUN pip install --requirement /tmp/requirements.txt
#CMD ["Jupyter Notebook"]    