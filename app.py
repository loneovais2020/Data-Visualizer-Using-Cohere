import streamlit as st 
import os
import glob
from langchain_cohere.chat_models import ChatCohere
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool, AgentExecutor
from langchain_experimental.utilities import PythonREPL
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from langchain.callbacks.base import BaseCallbackHandler
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import socket
import json
from datetime import datetime
import re



def get_conversation_context(conversation_id, file_path='conversation_history.json'):
    """Retrieves previous conversation context for the current session."""
    try:
        with open(file_path, 'r') as f:
            # Check if the file is empty
            if os.stat(file_path).st_size == 0:
                return ""  # Return an empty string if the file is empty

            history = json.load(f)
        
        # Filter history for current conversation
        current_context = [
            entry for entry in history 
            if entry['conversation_id'] == conversation_id
        ]
        
        # Format context for the agent
        context_text = ""
        if current_context:
            context_text = "\n\nPrevious conversation context:\n"
            for entry in current_context[-3:]:  # Last 3 interactions
                context_text += f"\nUser: {entry['user_query']}\n"
                context_text += f"Assistant: {entry['agent_data']}\n"
                if entry.get('charts_generated'):
                    context_text += f"Charts generated: {', '.join(entry['charts_generated'])}\n"
        
        return context_text
    except json.JSONDecodeError:
        # Handle JSON decode error if the file is empty or malformed
        return ""
    except Exception as e:
        st.warning(f"Could not load conversation history: {str(e)}")
        return ""



# Load environment variables
load_dotenv()

# Add flag to control API key source
USE_ENV_KEYS = True  # Set to True to use .env keys, False to use user input

# Initialize session state for API keys
if 'cohere_api_key' not in st.session_state:
    st.session_state['cohere_api_key'] = os.getenv('COHERE_API_KEY') if USE_ENV_KEYS else ''
if 'tavily_api_key' not in st.session_state:
    st.session_state['tavily_api_key'] = os.getenv('TAVILY_API_KEY') if USE_ENV_KEYS else ''
if 'data_source' not in st.session_state:
    st.session_state['data_source'] = 'Web Search'
if 'conversation_id' not in st.session_state:
    st.session_state['conversation_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")

# Add page configuration
st.set_page_config(
    page_title="AI Chart Generator",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main title and description
st.markdown("<h1 class='main-header'>🤖 AI-Powered Chart Generator</h1>", unsafe_allow_html=True)
st.markdown("""
    Transform your data queries into beautiful visualizations using AI! 
    Simply describe what you want to visualize, and let our AI create the perfect chart for you.
    """)

# Add a divider
st.divider()

# Enhance the sidebar
st.sidebar.image("https://img.icons8.com/clouds/200/analytics.png", use_column_width=True)
st.sidebar.markdown("## ⚙️ Configuration")

if not USE_ENV_KEYS:
    st.sidebar.markdown("""
        ### 🔑 API Keys
        Please enter your API keys below to use the application.
        You can get them from:
        - [Cohere](https://cohere.com/)
        - [Tavily](https://tavily.com/)
    """)
    # Sidebar for API key inputs
    cohere_api_key = st.sidebar.text_input("Enter Cohere API Key:", type="password", value=st.session_state['cohere_api_key'])
    tavily_api_key = st.sidebar.text_input("Enter Tavily API Key:", type="password", value=st.session_state['tavily_api_key'])

    if st.sidebar.button("Submit API Keys"):
        if cohere_api_key and tavily_api_key:
            # Store API keys in session state
            st.session_state['cohere_api_key'] = cohere_api_key
            st.session_state['tavily_api_key'] = tavily_api_key
            st.sidebar.success("API keys saved successfully!")
        else:
            st.sidebar.error("Please enter both API keys.")

# Main content area
if st.session_state['cohere_api_key'] and st.session_state['tavily_api_key']:
    # Initialize the Cohere model
    chat = ChatCohere(model="command-r-plus", temperature=0.7, api_key=st.session_state['cohere_api_key'])

    # Initialize internet search tool
    internet_search = TavilySearchResults(api_key=st.session_state['tavily_api_key'])
    internet_search.name = "internet_search"
    internet_search.description = "Returns a list of relevant documents from the internet."

    # Pydantic model for internet search
    class TavilySearchInput(BaseModel):
        query: str = Field(description="Internet query engine.")

    internet_search.args_schema = TavilySearchInput

    # Initialize Python REPL tool
    python_repl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        description="Executes python code and returns the result.",
        func=python_repl.run,
    )

    repl_tool.name = "python_interpreter"

    # Pydantic model for Python REPL
    class ToolInput(BaseModel):
        code: str = Field(description="Python code execution.")

        # Convert to a class method
        @classmethod
        def model_json_schema(cls):
            return cls.schema_json()

    repl_tool.args_schema = ToolInput

    # Add data source selection
    st.markdown("<h2 class='sub-header'>📊 Choose Your Data Source</h2>", unsafe_allow_html=True)
    data_source = st.radio(
        "Select data source:",
        ['Web Search', 'Upload Dataset'],
        key='data_source',
        horizontal=True
    )

    uploaded_file = None
    file_path = None

    if data_source == 'Upload Dataset':
        uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel)", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            # Create 'files' directory if it doesn't exist
            os.makedirs('files', exist_ok=True)
            
            # Clean up previous files in the 'files' directory
            for file in glob.glob("files/*"):
                try:
                    os.remove(file)
                except Exception as e:
                    st.warning(f"Could not remove file {file}: {e}")
            
            # Save the file
            file_path = os.path.join('files', uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded successfully: {uploaded_file.name}")

            # Display a preview of the dataset
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                
                st.markdown("<h3 class='sub-header'>🔍 Dataset Preview</h3>", unsafe_allow_html=True)
                st.dataframe(df.head())  # Display the first few rows of the dataset
            except Exception as e:
                st.error(f"Error reading the file: {e}")

    # Modify tools list based on data source
    tools = [repl_tool]
    if data_source == 'Web Search':
        tools.append(internet_search)

    # Create 'charts' directory if it doesn't exist
    os.makedirs('charts', exist_ok=True)

    # Create the agent
    prompt_template = """You are a data visualization expert. The user has requested: '{input}'

    {context}

    CRITICAL WORKFLOW:
    1. ALWAYS start by inspecting the data:
    ```python
    df = pd.read_excel('{file_path}')
    print("Available columns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    ```

    2. After inspection:
    - Use ONLY the columns that actually exist in the dataset
    - If the user requests columns that don't exist:
        a) Find similar or related columns that DO exist
        b) Inform the user about which columns you're using instead
        c) Explain why you made these choices

    3. IMPORTANT - Chart Saving Instructions:
    - ALWAYS save charts to the 'charts' directory using:
        ```python
        # Create directory if it doesn't exist
        import os
        os.makedirs('charts', exist_ok=True)
        
        # Save the chart
        plt.savefig('charts/output.png', bbox_inches='tight', dpi=300)
        plt.close()  # Close the figure to free memory
        ```
    - Ensure the chart is saved BEFORE the script ends
    - Always use plt.close() after saving to prevent memory leaks
    - Use descriptive filenames (e.g., 'charts/sales_by_region.png')

    4. Error Recovery:
    - If you encounter a KeyError:
        a) Print df.columns to verify available columns
        b) Look for similar column names (case-sensitive check)
        c) Suggest alternatives based on available data
    - If you encounter other errors:
        a) Print df.info() to check data types
        b) Handle missing values appropriately
        c) Clean data as needed

    5. Communication:
    - Always explain what you're doing and why
    - If you make substitutions, explain your choices
    - If you can't create exactly what was requested, explain why and what you're creating instead

    Remember:
    - The user may not know the exact column names
    - Similar names might exist (e.g., "Units" vs "UnitsSold")
    - Case sensitivity matters
    - Always verify column existence before using
    - ALWAYS save charts in the 'charts' directory

    Create the most appropriate visualization based on the available data, even if it differs from the exact request."""

    context = ""
    if data_source == 'Upload Dataset' and file_path:
        context = get_conversation_context(st.session_state['conversation_id'])

    prompt = ChatPromptTemplate.from_template(prompt_template.format(input="{input}", context=context, file_path=file_path or ""))

    agent = create_cohere_react_agent(
        llm=chat,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    st.markdown("<h2 class='sub-header'>🎯 What would you like to visualize?</h2>", unsafe_allow_html=True)
    st.markdown("""
        Examples:
        - "Create a chart showing global temperature changes over the last 50 years"
        - "Visualize the market share of top 5 smartphone brands in 2023"
        - "Plot the population growth of major cities in the last decade"
    """)
    
    user_input = st.text_area(
        "Enter your query:",
        height=100,
        placeholder="Describe the chart you want to create..."
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit_button = st.button("🎨 Generate Chart", use_container_width=True)

    if submit_button:
        # Create a placeholder for the streaming output
        output_placeholder = st.empty()
        
        with st.spinner("🤖 AI is working its magic..."):
            try:
                # Clean up previous files
                # Delete previous charts
                for file in glob.glob("charts/*.png"):
                    try:
                        os.remove(file)
                    except Exception as e:
                        st.warning(f"Could not remove chart file {file}: {e}")
                # Save new uploaded file if exists
                if uploaded_file is not None:
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                # Create a callback handler to capture the output
                def clean_agent_output(output_text):
                    """Clean markdown image links and URLs from the output text."""
                    # Remove markdown image links ![...](...) and similar patterns
                    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', output_text)
                    # Remove bare URLs to images
                    cleaned = re.sub(r'https?://\S+?(?:jpg|jpeg|png|gif)(?:\s|$)', '', cleaned)
                    # Remove references to charts/directory
                    cleaned = re.sub(r'(?i)charts/[\w\-]+\.png', '', cleaned)
                    # Remove any remaining markdown links [...](...) if they point to images
                    cleaned = re.sub(r'\[.*?\]\(.*?(?:jpg|jpeg|png|gif)\)', '', cleaned)
                    # Remove any lines starting with "!" (common for image markdown)
                    cleaned = re.sub(r'^!.*$', '', cleaned, flags=re.MULTILINE)
                    # Clean up any double spaces or empty lines
                    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
                    cleaned = re.sub(r'  +', ' ', cleaned)
                    
                    return cleaned.strip()

                class StreamHandler(BaseCallbackHandler):
                    def __init__(self):
                        self.output = []
                        self.placeholder = st.empty()

                    def on_llm_start(self, *args, **kwargs):
                        self.output.append("🤔 Thinking...")
                        self._update_output()

                    def on_llm_end(self, *args, **kwargs):
                        # Remove the "Thinking..." message
                        if self.output and "🤔 Thinking..." in self.output[-1]:
                            self.output.pop()
                        self._update_output()

                    def on_tool_start(self, *args, **kwargs):
                        self.output.append("🔧 Working on it...")
                        self._update_output()

                    def on_tool_end(self, *args, **kwargs):
                        # Remove the "Working on it..." message
                        if self.output and "🔧 Working on it..." in self.output[-1]:
                            self.output.pop()
                        self._update_output()

                    def on_agent_finish(self, finish, **kwargs):
                        # Get the output and clean it
                        output = finish.return_values.get('output', '')
                        cleaned_output = clean_agent_output(output)
                        
                        # Format the final output nicely
                        self.output.append(f"""
🎉 **Task Completed!**
{cleaned_output}
---""")
                        self._update_output()

                    def _update_output(self):
                        # Add custom CSS for better formatting
                        css = """
                        <style>
                        .code-block {
                            background-color: #f6f8fa;
                            border-radius: 6px;
                            padding: 16px;
                            margin: 8px 0;
                        }
                        .error-block {
                            background-color: #ffebe9;
                            border: 1px solid #ff8182;
                            border-radius: 6px;
                            padding: 16px;
                            margin: 8px 0;
                        }
                        .output-container {
                            border-left: 3px solid #1E88E5;
                            padding-left: 16px;
                            margin: 16px 0;
                        }
                        </style>
                        """
                        # Join all output with proper spacing and render as markdown
                        formatted_output = "\n\n".join(self.output)
                        self.placeholder.markdown(css + formatted_output, unsafe_allow_html=True)

                # Initialize the callback handler
                stream_handler = StreamHandler()

                # Modify the agent executor to include callbacks
                agent_executor = AgentExecutor(
                    agent=agent, 
                    tools=tools, 
                    verbose=True,
                    callbacks=[stream_handler]
                )

                # Add retry decorator to handle network issues
                @retry(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=4, max=10),
                    reraise=True
                )
                def execute_agent_with_retry(agent_executor, input_dict):
                    try:
                        return agent_executor.invoke(input_dict)
                    except socket.gaierror as e:
                        st.error("Network connection error. Please check your internet connection.")
                        raise
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
                        raise

                # Add this function to handle conversation storage
                def manage_conversation_history(user_query, agent_response, file_path='conversation_history.json'):
                    """Manages the conversation history in a JSON file."""
                    try:
                        # Extract the text content from the agent response
                        if isinstance(agent_response, dict):
                            agent_data = agent_response.get('output', str(agent_response))
                        else:
                            agent_data = str(agent_response)
                        
                        # Get list of generated charts (only filenames, not full paths)
                        charts_generated = [os.path.basename(f) for f in glob.glob("charts/*.png")]
                        
                        # Create the history structure with serializable data
                        conversation_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "conversation_id": st.session_state['conversation_id'],
                            "user_query": user_query,
                            "agent_data": agent_data,
                            "charts_generated": charts_generated
                        }
                        
                        # Load existing history or create new
                        try:
                            with open(file_path, 'r') as f:
                                history = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            history = []
                        
                        # Add new entry
                        history.append(conversation_entry)
                        
                        # Save updated history
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(history, f, indent=2, ensure_ascii=False)
                            
                        return history
                    except Exception as e:
                        st.warning(f"Could not save conversation history: {str(e)}")
                        return []

                def get_conversation_context(conversation_id, file_path='conversation_history.json'):
                    """Retrieves previous conversation context for the current session."""
                    try:
                        with open(file_path, 'r') as f:
                            history = json.load(f)
                        
                        # Filter history for current conversation
                        current_context = [
                            entry for entry in history 
                            if entry['conversation_id'] == conversation_id
                        ]
                        
                        # Format context for the agent
                        context_text = ""
                        if current_context:
                            context_text = "\n\nPrevious conversation context:\n"
                            for entry in current_context[-3:]:  # Last 3 interactions
                                context_text += f"\nUser: {entry['user_query']}\n"
                                context_text += f"Assistant: {entry['agent_data']}\n"
                                if entry.get('charts_generated'):
                                    context_text += f"Charts generated: {', '.join(entry['charts_generated'])}\n"
                        
                        return context_text
                    except Exception as e:
                        st.warning(f"Could not load conversation history: {str(e)}")
                        return ""

                try:
                    # Execute the agent with retry logic
                    response = execute_agent_with_retry(agent_executor, {
                        "input": user_input, 
                        "file_path": file_path,
                        "context": context
                    })
                    
                    # Extract output from response if needed
                    output = response.get('output', str(response)) if isinstance(response, dict) else str(response)
                    
                    # Save conversation history
                    manage_conversation_history(user_input, output)
                    
                    st.success("Done!")
                    
                    # Enhance the results display
                    st.markdown("<h2 class='sub-header'>📊 Generated Chart</h2>", unsafe_allow_html=True)
                    # Display the newly generated chart(s)
                    new_png_files = glob.glob("charts/*.png")  # Look in charts directory
                    if not new_png_files:
                        # If no files found in charts directory, check current directory as fallback
                        new_png_files = glob.glob("*.png")
                    if not new_png_files:
                        st.warning("No charts were generated.")
                    for png_file in new_png_files:
                        st.image(png_file, caption=png_file)

                    st.markdown("<h2 class='sub-header'>📚 References</h2>", unsafe_allow_html=True)
                    st.markdown("Data sources used to create this visualization:")

                    # Show the file used if data source is 'Upload Dataset'
                    if data_source == 'Upload Dataset' and uploaded_file is not None:
                        st.markdown(f"- **File Used:** {uploaded_file.name}")

                    else:
                        # Initialize an empty list to collect all URLs
                        urls = []

                        # Iterate through the 'citations' in the response
                        for citation in response.get('citations', []):
                            for document in citation.documents:
                                if 'url' in document:
                                    urls.append(document['url'])

                        # Print all collected URLs
                        if urls:
                            for url in urls:
                                st.write(url)
                        else:
                            st.warning("No references were found in the response.")

                except Exception as e:
                    st.error(f"🚨 Failed after retries: {str(e)}")
                    st.warning("Please try again or check your internet connection.")

            except KeyError as ke:
                st.error(f"Missing expected key in the response: {ke}")
            except Exception as e:
                st.error(f"🚨 An error occurred: {str(e)}")
else:
    st.warning("👋 Welcome! Please configure your API keys in the sidebar to get started.")

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ using Streamlit, Cohere, and Tavily</p>
        <p>© 2024 AI Chart Generator</p>
    </div>
""", unsafe_allow_html=True)

# Add this in the sidebar section
with st.sidebar.expander("💬 Conversation History", expanded=False):
    try:
        with open('conversation_history.json', 'r') as f:
            history = json.load(f)
        current_history = [
            entry for entry in history 
            if entry['conversation_id'] == st.session_state['conversation_id']
        ]
        
        for entry in current_history:
            st.markdown(f"**User:** {entry['user_query']}")
            st.markdown(f"**Assistant:** {entry['agent_data']}")
            if entry.get('charts_generated'):
                st.markdown(f"*Charts: {', '.join(entry['charts_generated'])}*")
            st.divider()
    except (FileNotFoundError, json.JSONDecodeError):
        st.info("No conversation history yet.")