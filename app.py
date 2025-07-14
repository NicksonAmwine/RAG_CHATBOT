import streamlit as st
import time

def render_chat_interface():
    """Render the enhanced chat interface."""
    
    # Document status display
    if not st.session_state.document_processed:
        st.markdown("""
        <div class="status-indicator status-warning">
            ⚠️ No document loaded - Please process a document from the sidebar to start chatting
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Chat history display
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
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
                        {chat['human']}
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
                        {chat['ai']}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.caption(f"Based on {chat['context_chunks']} relevant chunks")
    else:
        # Welcome message with better styling
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(145deg, #f8f9ff 0%, #e8f2ff 100%); border-radius: 20px; margin: 2rem 0;">
            <h3 style="color: #667eea; margin-bottom: 1rem;">🚀 Ready to explore your document!</h3>
            <p style="color: #666; font-size: 1.1rem;">Ask me anything about the content, and I'll provide detailed answers based on the document context.</p>
            <div style="margin-top: 1.5rem;">
                <span style="display: inline-block; background: #667eea; color: white; padding: 0.5rem 1rem; border-radius: 20px; margin: 0.25rem;">💡 Try: "What is this document about?"</span>
                <span style="display: inline-block; background: #764ba2; color: white; padding: 0.5rem 1rem; border-radius: 20px; margin: 0.25rem;">🔍 Try: "Summarize the main points"</span>
                <span style="display: inline-block; background: #f5576c; color: white; padding: 0.5rem 1rem; border-radius: 20px; margin: 0.25rem;">❓ Try: "Find information about..."</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        user_query = st.text_input(
            "Type your question here...", 
            key=f"query_input_{len(st.session_state.chat_history)}",
            placeholder="💭 Ask anything about your document...",
            label_visibility="collapsed"
        )
    with col2:
        submit_button = st.button(
            "Send", 
            type="primary", 
            use_container_width=True
            )
    
    with col3:
        clear_button = st.button(
            "Clear", 
            use_container_width=True
            )

    # Handle query submission with enhanced feedback
    if submit_button and user_query.strip():
        return {"action": "submit", "query": user_query}
    
    # Handle clear button with confirmation
    if clear_button:
        return {"action": "clear"}
    
    return None