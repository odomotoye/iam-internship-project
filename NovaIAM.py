import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import groq
from groq import Groq

# os.environ['GROQ_API_KEY'] = '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'

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="IAM Form Recommender", page_icon="🛡️", layout="centered")
st.title("🛡️ NovaIAM Request Assistant")
st.write(
    """
    Enter an Identity & Access Management request below.  
    The system will analyze the request and recommend the correct form with a direct link.
    """
)

# ============================================
# LOAD RESOURCES
# ============================================
@st.cache_resource
def load_resources():
    base_path = "/mount/src/iam-internship-project"
    df_forms = pd.read_csv(f"{base_path}/cluster_forms_to_links.csv")
    
    try:
        df_examples = pd.read_csv(f"{base_path}/cluster_examples.csv", on_bad_lines="skip")
    except Exception as e:
        st.warning(f"Could not load cluster_examples.csv: {e}")
        df_examples = None

    try:
        df_embeds = pd.read_parquet(f"{base_path}/cluster_keyword_embeddings.parquet")
    except Exception:
        df_embeds = None

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Combine subcategory + keywords for embeddings
    form_texts = df_forms["subcategory"].astype(str) + " \n" + df_forms["keywords"].astype(str)
    keyword_embeddings = encoder.encode(form_texts.tolist())

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    return df_forms, df_examples, df_embeds, keyword_embeddings, encoder, client

df_forms, df_examples, df_embeds, keyword_embeddings, encoder, client = load_resources()

# ============================================
# CLUSTER SELECTION
# ============================================
def find_best_cluster(user_text):
    user_emb = encoder.encode([user_text])
    sims = cosine_similarity(user_emb, keyword_embeddings)[0]

    top_k = np.argsort(sims)[::-1][:3]
    best_idx = int(top_k[0])
    row = df_forms.iloc[best_idx]

    top_matches = []
    for idx in top_k:
        r = df_forms.iloc[int(idx)]
        top_matches.append({
            "index": int(idx),
            "name": str(r["subcategory"]),
            "form_link": str(r["form_link"]),
            "similarity": float(sims[int(idx)])
        })

    return {
        "cluster_id": int(row.get("hdbscan_cluster", -1)),
        "subcategory": row["subcategory"],
        "form_link": row["form_link"],
        "similarity_score": float(sims[best_idx]),
        "top_matches": top_matches,
        "forms_count": len(df_forms),
        "embeddings_count": getattr(keyword_embeddings, 'shape', (None,))[0] if keyword_embeddings is not None else 0,
    }

# ============================================
# LLM RESPONSE
# ============================================
def generate_llm_response(user_text, subcategory, form_link):
    system_prompt = f"""
    You are an IAM request assistant. A user provided a request.
    Your job: recommend the correct IAM form.

    ALWAYS use the official form link provided (never hallucinate URLs).

    Return a friendly, concise response with:
    - Recommended IAM subcategory
    - The exact clickable link from the system
    A brief explanation why, starting on the next line after the link.

    IMPORTANT: Make sure there is a line break (\n) after the link before the explanation begins.
    """

    try:
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"User Request: {user_text}\n\n"
                        f"Matched Subcategory: {subcategory}\n"
                        f"Form Link: {form_link}\n"
                    )
                }
            ],
            temperature=0.2,
            max_tokens=300,
        )

        return completion.choices[0].message.content

    except groq.AuthenticationError:
        st.error("Groq authentication failed: invalid API key.")
        return "Error: Groq authentication failed."

    except groq.GroqError as e:
        msg = str(e)
        st.error(f"LLM error: {msg}")
        return f"Error: {msg}"

    except Exception as e:
        st.error(f"LLM request failed: {e}")
        return f"Error: {e}"

# ============================================
# STREAMLIT UI
# ============================================
user_text = st.text_area("Describe your IAM request here:", height=160)

if st.button("Get Recommendation"):
    if not user_text.strip():
        st.warning("Please enter a request.")
    else:
        with st.spinner("Analyzing request..."):
            result = find_best_cluster(user_text)
            llm_output = generate_llm_response(
                user_text=user_text,
                subcategory=result["subcategory"],
                form_link=result["form_link"]
            )

        st.success("Recommendation Ready!")

        st.write("### 📝 Suggested IAM Form")
        st.write(llm_output)

        # Extra details
        with st.expander("🔍 How was this determined?"):
            st.write(f"**Cluster ID:** {result['cluster_id']}")
            st.write(f"**Similarity Score:** {result['similarity_score']:.4f}")
            st.write("Embedding-based cluster assignment using SBERT + cosine similarity.")

            st.write("**Top Matches & Links:**")
            for m in result.get('top_matches', []):
                st.markdown(f"- [{m['name']}]({m['form_link']}) (index {m['index']}): {m['similarity']:.4f}")

st.write("---")
st.caption("Powered by SBERT embeddings + Groq LLM · IAM POC")
