import openai
# langchain libs
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# streamlit lib
import streamlit as st

# io lib
from io import BytesIO

# join paths and importing pdf library
from PyPDF2 import PdfReader
from pdf import PDF

from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


import pdfplumber
import spacy
import pandas as pd

import re
import string

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

terms = {'Data Engineering & Warehousing ':['build', 'maintain', 'data', 'pipelines', 'engineering', 'organize', 
                                            'large','integrity','testing', 'validation', 'Postgres DBs', 'Kinesis',  
                                            'APIs', 'Gathering', 'ETL', 'Modeling','data warehouse', 'integrate', 
                                            'SQL', 'Server', 'design', 'solution', 'ad-hoc', 'streamline','extraction', 
                                            'troubleshoot', 'architecture', 'endpoints', 'development', 'lake', 'architecture',
                                            'databases', 'NoSQL', 'exploration'],      
        'Data Mining & Statistical Analysis':['SAS', 'ANOVA', 'statistical', 'methodologies', 'regression', 'data mining',
                                             'problem-solving', 'theories', 'test', 'hypotheses', 'anamoly-detection',
                                             'SPSS', 'RStudio', 'quantitative', 'analyses', 'model', 'mathematics', 'statistic',
                                             'techniques', 'Bayesian', 'research', 'sampling', 'findings', 'analysis', 'R',
                                             'conduct', 'insights', 'statistical integrity', 'math', 'identify', 'analyze',
                                             'trend', 'stake-holders', 'MATLAB', 'Inferential Statistics','Multivariate Analysis', 
                                              'Linear','Non-linear', 'mortality', 'risk factors', 'survey'],
        'Cloud & Distributed Computing':['multi-cloud', 'develop','cloud solutions', 'domain','architects', 'technical', 'cloud',
                                        'architecture','engineering', 'TOGAF','Zachman', 'Policies','Governance', 'Strategies',
                                        'AWS', 'Redshift', 'PostgresQL', 'Oracle', 'cloud based', 'OLTP''metadata','OLAP', 'GCP',
                                        'Spark', 'APIs', 'Python', 'framework', 'understanding', 'data-driven', 'Azure', 'platform',
                                        'design', 'domain', 'tool', 'trend', 'deployment', 'application','build','environment','DevOps',
                                         'pipeline', 'manage', 'server', 'services'],
        'ML & AI':['structured', 'unstructured', 'kafka', 'spark', 'datapieline', 'big data','technologies', 'hive','hadoop','PySpark', 
                   'Python', 'SQL', 'MySQL', 'databases','tools', 'AWS', 'GCP', 'information retrieval', 'machine learning', 'features', 
                   'engineering','data mining', 'data processing', 'large', 'NLP', 'text', 'analytical skills', 'deployment','Git', 'Linux', 
                   'Windows','C','C++','Java','DevOps','distributed', 'software','development','requirements', 'experience', 'Tensor Flow', 
                   'PyTorch','supervised', 'unsupervised', 'building','evaluation', 'ML libraries','frameworks', 'exploratory analyses',
                   'traditional','techniques','AI', 'algorithms', 'analyze', 'develop', 'evaluate','classification','library'],
        'Data Visualization':['analytics','BI Tools','chart','big data','business intelligence','power BI','BOBJ','visualization','data',
                              'database','data mining','data science','charts','hadoop','graphs','MS Excel','pivot-tables','machine learning',
                              'creative','nosql','nlp','predictive','insights','python','r','sql','tableau','text mining','findings', 'communication', 
                              'skills', 'statistical', 'data-driven', 'SPSS', 'Alteryx', 'business', 'identify', 'dashboard','Bash','statistical',
                             'resolve', 'translate', 'summerize', 'analyst', 'ggplot2', 'reporting', 'business', 'result']}

class RecruitAI:

    def __init__(self, openai_api_key: str, model="gpt-3.5-turbo") -> None:

        self.openai_api_key = openai_api_key
        self.model = model
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model=self.model,
            temperature=0.0
        )
    
    
    def text2pdf(self, txt_content: str, with_header: bool = True) -> None:

        """
        Convert a txt file to pdf file.

        Parameters
        ----------
        txt_content : str
            content of txt file
        
        figure_logo : bool
            if True, add logo in pdf file

        Returns
        -------
        _file : BytesIO
            pdf file
        """

        pdf = PDF(with_header=with_header)
        pdf.add_page()
        pdf.add_text(txt_content)
        _file = BytesIO(pdf.output(dest="S").encode("latin1"))

        return _file

    def get_text_from_pdf(self, pdf: st.file_uploader):

        """
        Take the text from a pdf file

        Parameters
        ----------
        pdf : st.file_uploader
            pdf file

        Returns
        -------
        text : str
            text from pdf
        """

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    

    def get_similarity(self,resume, jd):
        res = ''.join([i for i in resume if not i.isdigit()])
        res_jd=[res, jd]
        cntv = CountVectorizer()
        count_matrix = cntv.fit_transform(res_jd)
        percentage = round((cosine_similarity(count_matrix)[0][1] * 100),2)
        return percentage
    
    def jaccard_distance(self,resume,jd):
        intersection=len(set.intersection(*[set(resume),set(jd)]))
        union=len(set.union(*[set(resume),set(jd)]))
        return intersection/float(union) *100
    
    def show_skills_classification(self,text):
        text = text.lower()
        text = re.sub(r'\d+','',text)
        text = text.translate(str.maketrans('','',string.punctuation))
        de= 0
        dm = 0
        cc = 0
        mlai=0
        dv = 0
        scores = []
        
        for area in terms.keys():
                                
            if area == 'Data Engineering & Warehousing':
                for word in terms[area]:
                    if word in text:
                        de +=1
                scores.append(de)
                                
            elif area == 'Data Mining & Statistical Analysis':
                for word in terms[area]:
                    if word in text:
                        dm +=1
                scores.append(dm)
                                
            elif area == 'Cloud & Distributed Computing':
                for word in terms[area]:
                    if word in text:
                        cc +=1
                scores.append(cc)
                            
            elif area == 'ML & AI':
                for word in terms[area]:
                    if word in text:
                        mlai +=1
                scores.append(mlai)
                                
            else:
                for word in terms[area]:
                    if word in text:
                        dv +=1
                scores.append(dv)
                    
        summary = pd.DataFrame(scores,index=terms.keys(),columns=['score']).sort_values(by='score',ascending=False) 
        summary.index.name = 'Five Concentration of Data Science'        
        return(summary)
    

    def show_pie_chart(self,summary):
        pie = plt.figure(figsize=(7,7))
        plt.pie(summary['score'], labels=summary.index, explode = (0.07,0,0,0,0), autopct='%1.0f%%',shadow=True,startangle=90)
        plt.title('Classification of Skills')
        plt.axis('equal')
        st.pyplot(pie)
    
    def parse_resume(self,text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        education = []
        work_experience = []
        skills = []
        
        edu_keywords = ["education", "qualifications", "academic"]
        work_keywords = ["experience", "employment", "work history", "professional"]
        skill_keywords = ["skills", "technologies", "proficiencies","technical skills", "TECHNICAL STRENGTH", "STRENGTH"]
        
        current_section = None
        
        for line in text.split("\n"):
            line_lower = line.lower()
            
            if any(keyword in line_lower for keyword in edu_keywords):
                current_section = "education"
            elif any(keyword in line_lower for keyword in work_keywords):
                current_section = "work_experience"
            elif any(keyword in line_lower for keyword in skill_keywords):
                current_section = "skills"
            elif current_section == "education":
                education.append(line)
            elif current_section == "work_experience":
                work_experience.append(line)
            elif current_section == "skills":
                skills.append(line)
        
        return {
            "education": education,
            "work_experience": work_experience,
            "skills": skills
        }

    def generate_summary(self,parsed_data):
        summary = (
            f"**Education:**\n{parsed_data['education']}\n\n"
            f"**Work Experience:**\n{parsed_data['work_experience']}\n\n"
            f"**Skills:**\n{parsed_data['skills']}\n"
        )
        return summary


    def create_tables(self,parsed_data):
        education_df = pd.DataFrame(parsed_data['education'], columns=["Education"])
        work_experience_df = pd.DataFrame(parsed_data['work_experience'], columns=["Work Experience"])
        skills_df = pd.DataFrame(parsed_data['skills'], columns=["Skills"])
        return education_df, work_experience_df, skills_df

    def get_prompt(
        self,
        requirements: str,
        curriculum: str
    ) -> list:

        """
        Get the prompt for the analysis of the curriculum

        Parameters
        ----------
        requirements : str
            requirements of the job

        curriculum : str
            curriculum of the candidate

        Returns
        -------
        prompt : str

        """

        messages = []

        prompt_system = """
Você é o melhor recrutador de todos os tempos. Você está analisando um currículo de um candidato para uma vaga de emprego.
Por mais que esta instrução esteja em português, você pode receber um currículo em outra língua que não seja
português ou inglês, com isso, ao final, você deve gerar os resultados em inglês sempre.
Esta vaga poderá ser de diversas áreas e para diversos cargos.
Você deve exclusivamente se basear nos requisitos passados abaixo. Os requisitos poderão ser a própria descrição da vaga
ou algumas exigências que o candidato deve ter para ocupar a vaga ou ambos.
Primeiro, você deve criar uma etapa fazendo um resumo das qualidades do candidato e destacar pontos que são de extremo
interesse da vaga. Pode ser que o currículo tenha caracterísiticas a mais do que é pedido, se esses requisitos forem interessantes
para a vaga, vale a pena destacar esses pontos. Após a etapa anterior, você deve dar pontuações para cada característica que você observar no currículo do
candidato e dar uma pontuação de 0 a 10, sendo 0 para o candidato que não atende a característica e 10 para o candidato que atende perfeitamente 
a característica, nessa etapa, você exclusivamente parear com os requisitos da vaga, devolvendo o nome da característica
da vaga e a pontuação do candidato para essa característica, sem mais e nem menos.
Ao final, você deverá dar uma nota final geral (também entre 0 a 10) deste candidato se baseando nas pontuações anteriores.

O resultado deve ser da forma:

Nome do Candidato: Resumo do candidato.

Requisitos:
As notas para cada requisito irão vir aqui.

Resultado Final:
Nota geral final irá vir aqui.
"""

        prompt_human = f"""
Requisitos:
{requirements}

Currículo do Candidato:
{curriculum}
"""

        messages.append(SystemMessage(content=prompt_system))
        messages.append(HumanMessage(content=prompt_human))

        return messages

    def get_recruit_results(self, messages: list):
        return self.llm(messages=messages)
    
    # Function to create word cloud
    # def create_wordcloud(self,text):
    #     wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    #     plt.figure(figsize=(10, 5))
    #     plt.imshow(wordcloud, interpolation='bilinear')
    #     plt.axis('off')
    #     st.pyplot()
    # def get_similarity(selfresume, jd):
    #     res = ''.join([i for i in resume if not i.isdigit()])
    #     res_jd=[res, jd]
    #     cntv = CountVectorizer()
    #     count_matrix = cntv.fit_transform(res_jd)
    #     percentage = round((cosine_similarity(count_matrix)[0][1] * 100),2)
    #     return percentage

    
