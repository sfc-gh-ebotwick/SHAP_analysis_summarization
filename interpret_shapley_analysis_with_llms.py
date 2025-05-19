from typing import Dict, List, Optional
import _snowflake
import json
import streamlit as st
import time
import shap
import pandas as pd

#Visualization imports
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

#Snowflake imports
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import complete
from snowflake.ml.registry import Registry


session = get_active_session()

#default db/schema/stage/semantic_model_file
DATABASE = "E2E_SNOW_MLOPS_DB"
SCHEMA = "STREAMLIT"
STAGE = "SEMANTIC"
FILE = "mortgage_lending_prediction.yaml"

db_list = [DATABASE]
schema_list = [SCHEMA]
stage_list = [STAGE]
yaml_list = [FILE]

SUMMARIZATION_LLM = st.sidebar.selectbox('Select your Summarization LLM:',(
                "snowflake-llama-3.3-70b",
                "llama3.1-70b",
                "llama4-maverick",
                "llama4-scout",   
                "claude-3-5-sonnet",
                "gemma-7b",
                "jamba-1.5-mini",
                "jamba-1.5-large",
                "jamba-instruct",
                "llama2-70b-chat",
                "llama3-8b",
                "llama3-70b",
                "llama3.1-8b",
                "llama3.1-405b",
                "llama3.2-1b",
                "llama3.2-3b",
                "llama3.3-70b",
                "mistral-large",
                "mistral-large2",
                "mistral-7b",
                "mixtral-8x7b",
                "reka-core",
                "reka-flash",
                "snowflake-arctic",
                "snowflake-llama-3.1-405b"), key="model_name")

# databases = session.sql('SHOW DATABASES').collect()
# for db in databases:
#     if db.name not in db_list:
#         db_list.append(db.name)
            
DATABASE = st.sidebar.selectbox('Which Database do you want to use?', 
                                db_list, 
                                placeholder= DATABASE,
                                key = "Database")


# schemas = session.sql(f'SHOW SCHEMAS in {DATABASE}').collect()
# for schema in schemas:
#     if schema.name not in schema_list:
#         schema_list.append(schema.name)

SCHEMA = st.sidebar.selectbox('Which SCHEMA do you want to use?',                              
                             schema_list, 
                             placeholder = SCHEMA,
                             key = "Schema")


# stages = session.sql(f'SHOW STAGES in {DATABASE}.{SCHEMA}').collect()
# for stage in stages:
#     if stage.name not in stage_list:
#         stage_list.append(stage.name)

STAGE = st.sidebar.selectbox('Which STAGE do you want to use?', 
                             stage_list, 
                             placeholder = STAGE,
                             key = "Stage")


if STAGE:
    yaml_files = [row[0][row[0].find('/')+1:] for row in session.sql(f'LS @{DATABASE}.{SCHEMA}.{STAGE}').select('"name"').collect()]
    FILE = st.sidebar.selectbox('Which File do you want to use?', yaml_files, key = "File")

class CortexAnalyst():
    def call_analyst_api(self,prompt: str) -> dict:

        """Calls the REST API and returns the response."""
        request_body = {
            "messages": st.session_state.messages,
            "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
        }
        resp = _snowflake.send_snow_api_request(
            "POST",
            f"/api/v2/cortex/analyst/message",
            {},
            {},
            request_body,
            {},
            30000,
        )
        if resp["status"] < 400:
            return json.loads(resp["content"])
        else:
            st.session_state.messages.pop()
            raise Exception(
                f"Failed request with status {resp['status']}: {resp}"
            )
            
    def process_api_response(self, prompt: str):
        """Processes a message and adds the response to the chat."""
        st.session_state.messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                # response = "who had the most rec yards week 10"
                response = self.call_analyst_api(prompt=prompt)
                request_id = response["request_id"]
                content = response["message"]["content"]
                st.session_state.messages.append(
                    {**response['message'], "request_id": request_id}
                )
                final_return = self.process_sql(content=content, request_id=request_id)  # type: ignore[arg-type]
                
        return final_return
        
    def process_sql(self,
        content: List[Dict[str, str]],
        request_id: Optional[str] = None,
        message_index: Optional[int] = None):
        """Displays a content item for a message."""
        message_index = message_index or len(st.session_state.messages)
        sql_markdown = 'No SQL returned!'
        if request_id:
            with st.expander("Request ID", expanded=False):
                st.markdown(request_id)
        for item in content:
            if item["type"] == "text":
                st.markdown(item["text"])
            elif item["type"] == "suggestions":
                with st.expander("Suggestions", expanded=True):
                    for suggestion_index, suggestion in enumerate(item["suggestions"]):
                        if st.button(suggestion, key=f"{message_index}_{suggestion_index}"):
                            st.session_state.active_suggestion = suggestion
            elif item["type"] == "sql":
                sql_markdown = self.execute_sql(sql = item["statement"])

        return sql_markdown

    # @st.cache_data
    def execute_sql(self, sql: str):
        with st.expander("SQL Query", expanded=False):
            st.code(sql, language="sql")
        with st.expander("Results", expanded=True):
            with st.spinner("Running SQL..."):
                session = get_active_session()

                df = session.sql(sql).to_pandas()

                    # .to_pandas()
                if len(df.index) > 1:
                    data_tab, line_tab, bar_tab = st.tabs(
                        ["Data", "Line Chart", "Bar Chart"]
                    )
                    # data_tab.dataframe(df.astype(str).T)
                    data_tab.dataframe(df)
                    if len(df.columns) > 1:
                        df = df.set_index(df.columns[0])
                    with line_tab:
                        st.line_chart(df)
                    with bar_tab:
                        st.bar_chart(df)
                else:
                    # st.dataframe(df.astype(str).T)
                    st.dataframe(df)

        return df

    def summarize_sql_results(self, prompt: str) -> str:
        sql_result = self.process_api_response(prompt).to_markdown()
        # st.write(f"Summarizing result using {SUMMARIZATION_LLM}...")
        summarized_result = complete(SUMMARIZATION_LLM, 
                                     f'''Summarize the following input prompt and corresponding SQL result 
                                     from markdown into a succint human readable summary. 
                                     Original prompt - {prompt}
                                     Sql result markdown - {sql_result}''')
        st.write(f"**{summarized_result}**")
        return summarized_result


    def plot_shap_values(self, shap_values):
        sv = shap_values[["Feature", "Influence"]].dropna(axis=0) 
        # Normalize absolute values for color intensity
        norm = mcolors.Normalize(vmin=0, vmax=sv['Influence'].abs().max())
    
        # Create dynamic color list
        colors = []
        for val in sv['Influence']:
            if val < 0:
                cmap = cm.Reds
                color = cmap(norm(abs(val)))
            else:
                cmap = cm.Greens
                color = cmap(norm(val))
            colors.append(mcolors.to_hex(color))  # Convert RGBA to hex
    
        # Set dark background
        plt.style.use('dark_background')
        sns.set_style("dark", {"axes.facecolor": "#2b2b2b"})
    
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Plot
        sns.barplot(
            data=sv,
            x='Influence',
            y='Feature',
            palette=colors,
            edgecolor=None,
            ax=ax
        )
    
        # Add zero line
        ax.axvline(0, color='lightgray', linewidth=1, linestyle='--')
    
        # Labels and titles
        ax.set_title('Tornado Chart of Feature Impact', fontsize=16, weight='bold', color='white', pad=15)
        ax.set_xlabel('Impact', fontsize=12, color='white')
        ax.set_ylabel('')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, color='black')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, color='black')
    
        # Style cleanup
        sns.despine(left=True, bottom=True)
        ax.grid(axis='x', linestyle=':', linewidth=0.5, color='gray', alpha=0.4)
    
        plt.tight_layout()
        return fig


    def explain_result(self, prompt):

        sql_result = self.process_api_response(prompt)
        # hardcode_sql = f"SELECT * FROM E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.DEMO_MORTGAGE_LENDING_TEST_1  LIMIT 1" 
        # WHERE LOAN_ID = {sql_result.LOAN_ID[0]}"        
        # hardcode = session.sql(hardcode_sql).to_pandas()
        
        # Define model name
        model_name = f"MORTGAGE_LENDING_MLOPS_1"
        
        # Create a registry to log the model to
        model_registry = Registry(session=session, 
                                  database_name=DATABASE, 
                                  schema_name="MLOPS_SCHEMA")
        
        st.write("Fetching model from registry...")
        
        model = model_registry.get_model(model_name).version('XGB_BASE')
        
        st.write("Computing shapley values...")
        # try:
        
        #Compute shap vals
        all_data = model.run(session.create_dataframe(sql_result), function_name="explain").to_pandas()
        
        #format output df
        shap_cols = [col for col in all_data.columns if col.endswith("_explanation")]
        excluded_cols = ['TIMESTAMP', 'XGB_OPTIMIZED_PREDICTION', 'MORTGAGERESPONSE']  # skip timestamp and optimized prediction
        actual_cols = [col for col in all_data.columns if col not in shap_cols and col not in excluded_cols]
        # Get the single row
        row = all_data.loc[0]
        
        # Build long-form rows
        data = []
        for col in actual_cols:
            shap_col = f"{col}_explanation"
            shap_val = row[shap_col] if shap_col in shap_cols else None
            data.append({
                "Feature": col,
                "Value": row[col],
                "Influence": shap_val
            })
        
        feature_shap_df = pd.DataFrame(data).sort_values(by="Influence", ascending=False, na_position='first').reset_index(drop=True)

        
        #Display and visualize data
        st.dataframe(feature_shap_df)
        st.pyplot(self.plot_shap_values(feature_shap_df))

        st.write("Summarizing results...")
        summarize_shap_prompt = f"""
        The following dataframe represents actual values (Value column) and associated shapley values 
        (Influence column) for features used in a ML model to determine whether an applicant is predicted to 
        pay off a mortgage loan on time or not. Please look at the input data and influence data
        and briefly summarize the key takeaways in terms of which features were most influential
        on the loan being approved (xgb_base_prediction =1) or denied (xgb_base_prediction =0)
 
        Note that these are local shap values (not global) and so values will show the impact for
        that particular feature value but do not have indications on the global impact of that feature
        across all loans.

        Make the summary brief and human readable and focus on how the key feature values paired with high/low 
        shapley values to form the prediction. Group the features by high positive influence and low positive influence 
        and only make brief mention of features with shapley values close to 0. 

        Whenever possible use markdown formatting to highlight column names etc for a very neat and clear summary
        
        Dataset: 
        {feature_shap_df.to_markdown()}"""
        
    
        summarized_results = complete(SUMMARIZATION_LLM, summarize_shap_prompt)

        st.write(summarized_results)

        return summarized_results

        # except:
        #     st.write("Failed to generate shapley values! Ensure your data is in the right format for this model.")
        #     return "Unable to generate explainability report!"
        

#instantiate class
CA = CortexAnalyst()


def show_conversation_history() -> None:
    for message_index, message in enumerate(st.session_state.messages):
        chat_role = "assistant" if message["role"] == "analyst" else "user"
        with st.chat_message(chat_role):
               try:
                   CA.process_sql(
                        content=message["content"],
                        request_id=message.get("request_id"),
                        message_index=message_index,
                    )
               except: 
                   st.write("No history found!")


def reset() -> None:
    st.session_state.messages = []
    st.session_state.suggestions = []
    st.session_state.active_suggestion = None


st.title(":sleuth_or_spy: SHAP + LLM Loan Prediction Explorer :sleuth_or_spy:")

st.markdown(f"Semantic Model: `{FILE}`")

if "messages" not in st.session_state:
    reset()

with st.sidebar:
    if st.button("Reset conversation"):
        reset()

show_conversation_history()

if user_input := st.chat_input("What is your question?"):
    st.empty()
    CA.explain_result(prompt=user_input)
    st.empty()
    
if st.session_state.active_suggestion:
    CA.process_api_response(prompt=st.session_state.active_suggestion)
    st.session_state.active_suggestion = None
