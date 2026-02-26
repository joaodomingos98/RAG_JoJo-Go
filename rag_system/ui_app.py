import streamlit as st
from main_app import run_rag_pipeline

# ========================================
# STREAMLIT PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="JoJo-Go",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ JoJo-Go")
st.caption("Ask questions about company policies, HR, IT, and more.")

# ========================================
# INITIALIZE SESSION STATE (Memory)
# ========================================
# Notice we are now storing "citations" and "avatar" in the dictionary
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "Information Retrieval Copilot",
            "avatar": "🤖",
            "content": "Hello! I am your AI information retrieval assistant. How can I help you today?",
            "citations": []
        }
    ]

# ========================================
# DISPLAY CHAT HISTORY
# ========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

        # If the message has citations stored in memory, display them in an expander
        if message.get("citations"):
            with st.expander("📚 Source Documents"):
                for doc in message["citations"]:
                    title = doc['metadata'].get('title', 'Untitled')
                    category = doc['metadata'].get('category', 'General')
                    similarity = doc.get('similarity', 0)
                    st.caption(f"- **{title}** (Category: {category}) | Match: {similarity:.2f}")

# ========================================
# CHAT INPUT & PROCESSING
# ========================================
if prompt := st.chat_input("📝 Type your question here..."):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Set the AI MAGIC custom role and emoji
    with st.chat_message("Information Retrieval Copilot", avatar="🤖"):

        # Stop Generation Button
        # If clicked while the app is running, Streamlit halts and re-runs the script
        if st.button("🛑 Stop Generating Response"):
            st.stop()

        with st.spinner("Searching documents and thinking..."):

            # Unpack both the string response AND the dictionary of search results
            response, search_results = run_rag_pipeline(prompt)

            st.markdown(response)

            # Display current citations immediately
            if search_results:
                with st.expander("📚 Source Documents"):
                    for doc in search_results:
                        title = doc['metadata'].get('title', 'Untitled')
                        category = doc['metadata'].get('category', 'General')
                        similarity = doc.get('similarity', 0)
                        st.caption(f"- **{title}** (Category: {category}) | Match: {similarity:.2f}")

    # Save the AI's response and citations to memory
    st.session_state.messages.append({
        "role": "JoJo-Go",
        "avatar": "⚡",
        "content": response,
        "citations": search_results
    })

# ========================================
# SIDEBAR (Controls)
# ========================================
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("Local RAG Pipeline Connected")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {
                "role": "Information Retrieval Copilot",
                "avatar": "🤖",
                "content": "Hello! I am your AI information retrieval assistant. How can I help you today?",
                "citations": []
            }
        ]
        st.rerun()