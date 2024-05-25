import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
import pandas as pd

if 'df' not in st.session_state:
    st.session_state.df = None  # This would store your dataframe

if 'lida' not in st.session_state:
    st.session_state.lida = None  # This would store your LIDA instance
if 'summary' not in st.session_state:
    st.session_state.summary = None  # This would store your LIDA instance

if 'textgen_config' not in st.session_state:
    st.session_state.textgen_config = None  # This would store your LIDA instance

if 'visualizations' not in st.session_state:
    st.session_state.visualizations = []  # This would store your visualizations

if 'goals' not in st.session_state:
    st.session_state.goals = []  # This would store your goals
if 'selected_goal_index' not in st.session_state:
    st.session_state.selected_goal_index = 0  # This stores the index of the currently selected goal
if 'goal_questions' not in st.session_state:
    st.session_state.goal_questions = []  # This stores the questions of the goals

if 'selected_goal' not in st.session_state:
    st.session_state['selected_goal'] = None
if 'selected_goal_object' not in st.session_state:
    st.session_state['selected_goal_object'] = None






def on_goal_select():
    # Update the index based on the new selection in the dropdown
    st.session_state.selected_goal_index = st.session_state.goal_questions.index(st.session_state.current_goal)
    # This explicitly updates selected_goal based on the new index
    st.session_state.selected_goal = st.session_state.goal_questions[st.session_state.selected_goal_index]
    st.session_state.selected_goal_object = st.session_state.goals[st.session_state.selected_goal_index]

           


# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="Exploratory Analysis with LIDA",
    page_icon="📊",
)
models = ["gpt-3.5-turbo-1106"]


num_visualizations=1
selected_library = "seaborn"
selected_method_label="llm"

summarization_methods = [
        {"label": "llm",
         "description":
         "Uses the LLM to generate annotate the default summary, adding details such as semantic types for columns and dataset description"},

     ]

st.write("## Exploratory Data Analysis with LIDA 📊  :bulb: :computer:")
intro_text="""
            Automatic generation of goals and visualizations for your data
            
            * [LIDA Homepage](https://github.com/microsoft/lida)
            * [LIDA GitHub](https://github.com/lida-project)
            * [indapa-lida Streamlit app GitHub](https://github.com/indapa/lida_indapa/)
            * [Demo screencast of this app](https://youtu.be/Ml74vZONdEI)


            ### Tips for using your own data

            
            - Ensure your data is tidy (one variable per column, one observation per row)
            - Ensure your data has a header row with column names
            - Ensure your data is in a CSV format

           The paper [Data organization in spreadsheets](https://www.tandfonline.com/doi/full/10.1080/00031305.2017.1375989) provides a good overview of tidy data principles.

           
           ### Questions or feedback? 
           
           Open an issue on [Github](https://github.com/indapa/lida_indapa/issues)

            """
st.write(intro_text)
lida=None

openai_key=st.secrets['OPENAI_API_KEY']



with st.sidebar:
    with st.form(key='form_one'):
    
        selected_model = st.selectbox(
            'Choose a model',
            options=models,
            index=0,
            key='model')
            
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            key='temp')
        st.markdown(
            f"Lower temperature makes goals more focused. Higher temperature makes them more complex.")
        
        num_goals = st.slider(
                "Number of goals to generate",
                min_value=1,
                max_value=10,
                value=4,
                key='num_goals')
        
        
    
        user_goal = st.text_input("Describe Your Goal:", key='user_goal')

        persona=st.text_input("Persona", key='persona', value="A data scientist")
        st.markdown(
            f"Persona is the role or identity that can help generate goals relevant to the target audience. For example, the application defaults to a 'A data scientist'")
            

        use_cache = st.checkbox("Use cache", value=True, key='use_cache')
        
        
        uploaded_file = st.file_uploader("Choose a file", type='csv')
        submit_button = st.form_submit_button(label='Submit')


if submit_button:

    
    
    #read in the data
    st.session_state.df = pd.read_csv(uploaded_file)
    #st.write(st.session_state.df.head())
    selected_method = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["label"]

    
    #make lida and text config objects and add to session state along with other parameters
    st.session_state.lida = Manager(text_gen=llm("openai", api_key=openai_key))
    st.session_state.textgen_config = TextGenerationConfig(
            n=num_visualizations,
            temperature=st.session_state.temp,
            model=st.session_state.model,
            use_cache=st.session_state.use_cache)
    
    
    
    #make summary object and add to session state
    # **** lida.summarize *****
    st.session_state.summary = st.session_state.lida.summarize(
        data=st.session_state.df,
        summary_method=selected_method,
        textgen_config=st.session_state.textgen_config)
    
    
    ###########################################################
    
    

    # make lida goals and add to session state
    st.session_state.goals = st.session_state.lida.goals(st.session_state.summary,
                                                          n=num_goals, 
                                                          persona=st.session_state.persona,
                                                          textgen_config=st.session_state.textgen_config)
    st.session_state.goal_questions = [goal.question for goal in st.session_state.goals]
    st.session_state.selected_goal_index = 0
    st.session_state.selected_goal = st.session_state.goal_questions[st.session_state.selected_goal_index]
    st.session_state.selected_goal_object = st.session_state.goals[st.session_state.selected_goal_index]

    #append user goal
    if st.session_state.user_goal is not None:
        new_goal = Goal(index=0,question=st.session_state.user_goal, visualization=st.session_state.user_goal, rationale="User Goal")
        st.session_state.goals.append(new_goal)
        st.session_state.goal_questions.append(new_goal.question)

if st.session_state.df is not None:
    st.write("## Data")
    st.write(st.session_state.df.head())
    
#display the summary based on session state
if st.session_state.summary:
    st.write("## Summary")
    #st.write(st.session_state.summary)
    if "dataset_description" in st.session_state.summary:
        st.write(st.session_state.summary["dataset_description"])

    if "fields" in st.session_state.summary:
        
        fields = st.session_state.summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["column"] = field["column"]
            # flatted_fields["dtype"] = field["dtype"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
                # flatted_fields = {**flatted_fields, **field["properties"]}
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)
        st.write(nfields_df)
    else:
        
        st.write(str(st.session_state.summary))

#display the goals based on session state
if st.session_state.goals:
    st.write("## Goals")
    selected_goal = st.selectbox(
        'Choose a generated goal',
        options=st.session_state.goal_questions,
        index=st.session_state.selected_goal_index,
        key='current_goal',
        on_change=on_goal_select)
    #st.write(f"Selected goal is: {selected_goal}")



if st.session_state.summary and st.session_state.selected_goal:
    #st.write("## Visualization")
    #st.write(f"Selected goal index: {st.session_state.selected_goal_index}")
    #st.write(f"Selected goal question: {st.session_state.selected_goal}") 
    #st.write(f"Selected goal object: {st.session_state.selected_goal_object}")
    st.session_state.visualizations = st.session_state.lida.visualize(
        summary=st.session_state.summary,
        goal=st.session_state.selected_goal_object,
        textgen_config=st.session_state.textgen_config,
        library=selected_library)
    
    if len(st.session_state.visualizations) == 0:
        st.info("No visualizations found for the selected goal")
    else:

        viz_titles = [f'Visualization {i+1}' for i in range(len(st.session_state.visualizations))]
        
        #selected_viz_title = st.selectbox('Choose a visualization', options=viz_titles, index=0)
        selected_viz_title=viz_titles[0]
        selected_viz=st.session_state.visualizations[viz_titles.index(selected_viz_title)]

        if selected_viz.raster:
            from PIL import Image
            import io
            import base64

            imgdata=base64.b64decode(selected_viz.raster)
            img = Image.open(io.BytesIO(imgdata))
            st.image(img, caption=selected_viz_title, use_column_width=True)

            st.write("### Visualization Code")
            st.code(selected_viz.code)

    