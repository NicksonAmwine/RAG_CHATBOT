import streamlit as st

def render_chat_interface(similarity_search_func, generate_answer_func):
    """Render the main chat interface."""
    # Display current document status
    if not st.session_state.document_processed:
        st.warning("Please process a document first using the sidebar.")
        return
    
    if st.session_state.chat_history:
        # Container for the chat messages
        chat_container = st.container()
        
        with chat_container:
            for i, chat in enumerate(st.session_state.chat_history):
                col1, col2 = st.columns([1, 4])
                with col2:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #0288d1; 
                            color: #ffffff;
                            padding: 10px; 
                            border-radius: 10px; 
                            margin: 5px 0;
                            border-left: 4px solid #01579b;
                        ">
                            {chat['user']}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #424242; 
                            color: #ffffff;
                            padding: 10px; 
                            border-radius: 10px; 
                            margin: 5px 0;
                            border-left: 4px solid #212121;
                        ">
                            {chat['bot']}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    st.caption(f"Based on {chat['context_chunks']} relevant chunks")

    else:
        st.info("Start a conversation by asking a question about your document!")
    
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        user_query = st.text_input(
            "Type your question here...", 
            key=f"query_input_{len(st.session_state.chat_history)}",
            placeholder="Ask anything about your document...",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_button = st.button("Send", type="primary", use_container_width=True)
    
    with col3:
        clear_button = st.button("Clear", use_container_width=True)

    # Handle query submission
    if submit_button and user_query.strip():
        handle_user_query(user_query, similarity_search_func, generate_answer_func)
    
    # Handle clear button
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()

def handle_user_query(user_query, similarity_search_func, generate_answer_func):
    """Handle user query submission."""
    with st.spinner("🔍 Searching..."):
        # Retrieve relevant chunks
        if st.session_state.collection is not None:
            retrieved_docs = similarity_search_func(st.session_state.collection, user_query, k=8)
            
            if retrieved_docs:
                context = "\n\n".join(retrieved_docs)
                answer = generate_answer_func(context, user_query)
                
                st.session_state.chat_history.append({
                    "user": user_query,
                    "bot": answer,
                    "context_chunks": len(retrieved_docs)
                })
                
                st.rerun()
            else:
                st.error("No relevant information found in the document.")
        else:
            st.error("No document collection found. Please process a document first.")