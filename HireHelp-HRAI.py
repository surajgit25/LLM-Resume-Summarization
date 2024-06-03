import streamlit as st
import base64
from recruitai import RecruitAI
from datetime import datetime
from PyPDF2 import PdfReader

import pandas as pd
import matplotlib.pyplot as plt







now = datetime.now()
time = now.strftime("%H:%M:%S")
st.set_page_config(page_title='RecruitAI')

with st.sidebar:
    openai_api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai_api_key.startswith('sk-') and len(openai_api_key) == 51):
        st.warning('Please enter your OPENAI API KEY!')
    else:
        recruit_ai = RecruitAI(openai_api_key=openai_api_key)
        st.success('OPENAI API KEY provided')

try:
    rad = st.sidebar.radio(
        'Browse the items',
        [
          
            "Analysis",
        ]
    )

    if rad == "Analysis":
        st.markdown("""Below, choose a job description and a curriculum vitae file,
                    respectively.
                    """)

        col1, col2 = st.columns(2)

        with col1:
            job_description = st.file_uploader(
                "Choose a job description file",
                type=['pdf'],
                accept_multiple_files=False
                )
            if job_description:
                jobdescription_filename= job_description.name

        with col2:
            curriculum = st.file_uploader(
                "Choose a cv file",
                type=['pdf'],
                accept_multiple_files=False
                )
            if curriculum:
                curriculum_filename= curriculum.name

        if job_description and curriculum:
            job_description_str = recruit_ai.get_text_from_pdf(job_description)
            curriculum_str = recruit_ai.get_text_from_pdf(curriculum)
            skill_classification= recruit_ai.show_skills_classification(recruit_ai.get_text_from_pdf(curriculum))
            st.subheader(" Skill Classification")
            st.dataframe(skill_classification)
            recruit_ai.show_pie_chart(skill_classification)
            parsed_data = recruit_ai.parse_resume(curriculum_str)
            section_table = recruit_ai.generate_summary(parsed_data)
            education_df, work_experience_df, skills_df =recruit_ai.create_tables(parsed_data)

            per=recruit_ai.get_similarity(curriculum_str,job_description_str)
            st.subheader("\n 1. Dissimmilarity Measure Between Resume and Job Description: ")
            st.subheader(per) 
            
            b=recruit_ai.jaccard_distance(curriculum_str,job_description_str) 
            st.subheader("\n 2. Similarity Score Between Resume  and Job Description:")
            st.subheader(b)

                   

            if st.button("Analyze"):

                messages = recruit_ai.get_prompt(
                    requirements=job_description_str,
                    curriculum=curriculum_str
                    )
                with st.status("Analyzing resume....."):

                    response = recruit_ai.llm(messages)
                response_pdf_bytes = recruit_ai.text2pdf(
                    txt_content=response.content,
                    with_header=True
                    )
        
                final_named_pdf = "results.pdf"
                with open(final_named_pdf, "wb") as f:
                    f.write(response_pdf_bytes.getbuffer())
                with open(final_named_pdf, "rb") as f:
                    bytes = f.read()
                    b64 = base64.b64encode(bytes).decode()
                    pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="700" height="800" type="application/pdf"></iframe>'

                    st.markdown(pdf_display, unsafe_allow_html=True)
                
                    download_link = f'<a href="/path/to/{final_named_pdf}" download="{final_named_pdf}">Download PDF</a>'
                st.download_button(label="Export_Report",
                    data=response_pdf_bytes,
                    file_name=jobdescription_filename + curriculum_filename +"_Result_" + time +".pdf",
                    mime='application/octet-stream')
                
            if st.button("Show Tables"):
                st.subheader("Education")
                st.dataframe(education_df)

                st.subheader("Work Experience")
                st.dataframe(work_experience_df)

                st.subheader("Skills")
                st.dataframe(skills_df)  
                
                   
                    


except Exception as e:
    st.warning(
        str(e)
        )

    
 

        